"""
Source Localization Head for Mobile-2 HD-EEG Benchmark.

Takes LaBraM-encoded EEG representation and predicts the 3D MNI coordinate
of the stimulated SEEG electrode position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .eeg_agent import LaBraMEncoder


class SourceLocalizationHead(nn.Module):
    """
    Predicts 3D MNI coordinates from EEG representations.

    Task A: HD-EEG (256ch) -> predict stimulated SEEG electrode position
    Loss: L2 distance to ground truth SEEG position in MNI space
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 3,  # MNI x, y, z
        dropout: float = 0.1,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) EEG representation

        Returns:
            (B, 3) predicted MNI coordinates
        """
        return self.head(x)


class Mobile2Model(nn.Module):
    """
    Full model for Mobile-2 source localization.

    Uses the LaBraM encoder (Path B of EEG Agent) plus a localization head.
    """

    def __init__(
        self,
        labram_hidden_dim: int = 200,
        labram_num_layers: int = 12,
        labram_num_heads: int = 8,
        labram_patch_size: int = 200,
        labram_max_channels: int = 256,
        projection_dim: int = 256,
        loc_hidden_dim: int = 512,
        loc_output_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.labram = LaBraMEncoder(
            hidden_dim=labram_hidden_dim,
            num_layers=labram_num_layers,
            num_heads=labram_num_heads,
            patch_size=labram_patch_size,
            max_channels=labram_max_channels,
            dropout=dropout,
        )

        self.proj = nn.Sequential(
            nn.Linear(labram_hidden_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.loc_head = SourceLocalizationHead(
            input_dim=projection_dim,
            hidden_dim=loc_hidden_dim,
            output_dim=loc_output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        raw_eeg: torch.Tensor,
        channel_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            raw_eeg: (B, C, T) at target sample rate
            channel_ids: (B, C) channel indices

        Returns:
            (B, 3) predicted MNI coordinates
        """
        hidden = self.labram(raw_eeg, channel_ids)
        projected = self.proj(hidden)
        return self.loc_head(projected)

    def compute_loss(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        L2 localization loss.

        Args:
            pred_coords: (B, 3) predicted MNI coordinates
            true_coords: (B, 3) ground truth MNI coordinates

        Returns:
            loss: scalar L2 loss
            metrics: dict with mean_error_mm
        """
        # L2 distance per sample
        distances = torch.norm(pred_coords - true_coords, dim=-1)  # (B,)
        loss = distances.mean()

        metrics = {
            "mean_error_mm": distances.mean().item(),
            "median_error_mm": distances.median().item(),
            "within_20mm": (distances < 20.0).float().mean().item(),
        }

        return loss, metrics

    def load_eeg_agent_weights(self, eeg_agent_state_dict: dict):
        """Load LaBraM weights from a trained EEG Agent."""
        labram_keys = {k: v for k, v in eeg_agent_state_dict.items() if k.startswith("labram.")}
        proj_keys = {k.replace("signal_proj.", "proj."): v
                     for k, v in eeg_agent_state_dict.items() if k.startswith("signal_proj.")}
        self.load_state_dict({**labram_keys, **proj_keys}, strict=False)
