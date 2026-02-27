"""
REVE EEG Foundation Model Adapter for Mobile-2 Downstream Tasks.

Wraps the pretrained REVE model (brain-bzh/reve-base, 69.2M params) as a
frozen feature extractor and adds task-specific heads for:
  1. Source localization (3D MNI regression)
  2. EZ region classification (Temporal/Frontal/Parieto-Occipital)
  3. Stimulation intensity classification (Low/High)

REVE expects: (B, C, T) EEG at 200Hz + (B, C, 3) electrode positions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance (Lin et al., 2017)."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class REVEAdapter(nn.Module):
    """REVE backbone + task-specific head for Mobile-2."""

    def __init__(
        self,
        task: str = "source_localization",
        reve_model_name: str = "brain-bzh/reve-base",
        reve_positions_name: str = "brain-bzh/reve-positions",
        hf_token: Optional[str] = None,
        feature_dim: int = 512,
        freeze_backbone: bool = True,
        unfreeze_last_n: int = 0,
        # Head configs
        loc_hidden_dim: int = 512,
        loc_output_dim: int = 3,
        region_classes: int = 3,
        intensity_classes: int = 2,
        dropout: float = 0.1,
        # Loss
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        # Augmentation
        aug_time_shift: bool = True,
        aug_amplitude_scale: bool = True,
        aug_amplitude_range: float = 0.2,
        # Electrode positions fallback
        electrode_positions: Optional[np.ndarray] = None,
        electrode_names: Optional[list] = None,
    ):
        super().__init__()
        self.task = task
        self.feature_dim = feature_dim
        self._electrode_positions = electrode_positions  # (C, 3) fallback
        self._electrode_names = electrode_names
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.aug_time_shift = aug_time_shift
        self.aug_amplitude_scale = aug_amplitude_scale
        self.aug_amplitude_range = aug_amplitude_range
        self.backbone_frozen = freeze_backbone and (unfreeze_last_n == 0)

        # Source normalization stats (set via set_target_stats)
        self._target_mean: Optional[torch.Tensor] = None
        self._target_std: Optional[torch.Tensor] = None

        # Class weights for weighted CE / focal loss (set via set_class_weights)
        self._class_weights: Optional[torch.Tensor] = None

        # Load REVE backbone
        from transformers import AutoModel

        self.pos_bank = AutoModel.from_pretrained(
            reve_positions_name, trust_remote_code=True, token=hf_token,
        )
        self.reve = AutoModel.from_pretrained(
            reve_model_name, trust_remote_code=True, token=hf_token,
        )

        if freeze_backbone:
            for param in self.reve.parameters():
                param.requires_grad = False
            for param in self.pos_bank.parameters():
                param.requires_grad = False

        # Unfreeze last N transformer layers for domain adaptation
        if unfreeze_last_n > 0 and hasattr(self.reve, 'transformer'):
            layers = list(self.reve.transformer.layers)
            n_unfrozen = min(unfreeze_last_n, len(layers))
            for layer in layers[-n_unfrozen:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"    Unfroze last {n_unfrozen}/{len(layers)} REVE transformer layers")

        # Resolve electrode positions once at init
        self._resolved_positions = self._resolve_positions()

        # Task head
        if task == "source_localization":
            self.head = nn.Sequential(
                nn.Linear(feature_dim, loc_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(loc_hidden_dim, loc_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(loc_hidden_dim, loc_output_dim),
            )
        elif task == "ez_region":
            # Deeper head for hard 3-class problem
            self.head = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, region_classes),
            )
        elif task == "stim_intensity":
            # Also deeper head for stim_intensity
            self.head = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, intensity_classes),
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    def set_class_weights(self, weights: torch.Tensor):
        """Set class weights for weighted cross-entropy / focal loss."""
        self._class_weights = weights

    def set_target_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Set target normalization stats for source localization."""
        self._target_mean = mean
        self._target_std = std

    def denormalize_prediction(self, pred: torch.Tensor) -> torch.Tensor:
        """Denormalize source localization output to MNI coordinates."""
        if self._target_mean is not None and self._target_std is not None:
            return pred * self._target_std.to(pred.device) + self._target_mean.to(pred.device)
        return pred

    def _resolve_positions(self) -> Optional[torch.Tensor]:
        """Get electrode positions. Prioritize raw positions for non-standard montages."""
        # Priority 1: raw positions from TSV (works for Mobile-2's e1-e256 naming)
        if self._electrode_positions is not None:
            pos = torch.tensor(self._electrode_positions, dtype=torch.float32)
            n_found = (pos.abs().sum(dim=-1) > 0).sum().item()
            print(f"Found {n_found} positions out of {pos.shape[0]} channels (raw TSV)")
            return pos

        # Priority 2: name-based lookup via REVE position bank
        if self._electrode_names is not None:
            try:
                positions = self.pos_bank(self._electrode_names)
                if positions is not None and positions.shape[0] > 0:
                    n_found = positions.shape[0]
                    print(f"Found {n_found} positions out of {len(self._electrode_names)} channels (pos_bank)")
                    return positions
            except Exception:
                pass

        return None

    def _augment_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation during training.

        Args:
            eeg: (B, C, T) raw EEG tensor

        Returns:
            Augmented (B, C, T) tensor
        """
        # Time-shift augmentation: random circular shift per sample
        if self.aug_time_shift:
            B, C, T = eeg.shape
            max_shift = max(1, T // 10)  # up to 10% of signal length
            shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=eeg.device)
            augmented = torch.zeros_like(eeg)
            for i in range(B):
                augmented[i] = torch.roll(eeg[i], shifts=shifts[i].item(), dims=-1)
            eeg = augmented

        # Amplitude scaling: random ±20% gain per channel
        if self.aug_amplitude_scale:
            B, C, T = eeg.shape
            scale = 1.0 + (torch.rand(B, C, 1, device=eeg.device) * 2 - 1) * self.aug_amplitude_range
            eeg = eeg * scale

        return eeg

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg: (B, C, T) at 200Hz, zero-padded to >= 1 second (200 samples)

        Returns:
            Task output: (B, 3) for regression, (B, n_classes) for classification
        """
        B = eeg.size(0)

        # Apply augmentation during training only
        if self.training:
            eeg = self._augment_eeg(eeg)

        # Get positions - (B, C, 3)
        if self._resolved_positions is not None:
            positions = self._resolved_positions.to(eeg.device)
            positions = positions.unsqueeze(0).expand(B, -1, -1)
        else:
            raise RuntimeError(
                "No electrode positions available. Provide electrode_positions "
                "or electrode_names at init."
            )

        # REVE forward — conditionally allow gradients for unfrozen layers
        if self.backbone_frozen:
            with torch.no_grad():
                features = self.reve(eeg, positions)
            if isinstance(features, tuple):
                features = features[0]
            if features.dim() == 4:
                features = features.mean(dim=(1, 2))
            elif features.dim() == 3:
                features = features.mean(dim=1)
            features = features.detach().float()
        else:
            features = self.reve(eeg, positions)
            if isinstance(features, tuple):
                features = features[0]
            if features.dim() == 4:
                features = features.mean(dim=(1, 2))
            elif features.dim() == 3:
                features = features.mean(dim=1)
            features = features.float()

        return self.head(features)

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute task-specific loss and metrics."""
        if self.task == "source_localization":
            distances = torch.norm(pred - target, dim=-1)  # (B,)
            loss = distances.mean()
            metrics = {
                "mean_error_mm": distances.mean().item(),
                "median_error_mm": distances.median().item(),
                "within_20mm": (distances < 20.0).float().mean().item(),
            }
        else:
            # Classification (ez_region or stim_intensity)
            # Move class weights to the right device
            weight = None
            if self._class_weights is not None:
                weight = self._class_weights.to(pred.device)

            if self.use_focal_loss:
                focal = FocalLoss(gamma=self.focal_gamma, weight=weight)
                loss = focal(pred, target)
            else:
                loss = F.cross_entropy(pred, target, weight=weight)

            preds = pred.argmax(dim=-1)
            acc = (preds == target).float().mean().item()
            metrics = {"accuracy": acc}

        return loss, metrics
