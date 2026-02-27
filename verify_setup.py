"""Quick verification script to check if all dependencies are installed."""

import sys


def check_imports():
    """Check if all required packages can be imported."""
    print("Checking imports...")
    errors = []

    try:
        import torch
        print(f"  torch {torch.__version__}")
    except ImportError as e:
        errors.append(f"  torch: {e}")

    try:
        import transformers
        print(f"  transformers {transformers.__version__}")
    except ImportError as e:
        errors.append(f"  transformers: {e}")

    try:
        import pandas as pd
        print(f"  pandas {pd.__version__}")
    except ImportError as e:
        errors.append(f"  pandas: {e}")

    try:
        from PIL import Image
        print("  pillow OK")
    except ImportError as e:
        errors.append(f"  pillow: {e}")

    try:
        import sentence_transformers
        print("  sentence-transformers OK")
    except ImportError as e:
        errors.append(f"  sentence-transformers: {e}")

    if errors:
        print("\nMissing:")
        for e in errors:
            print(e)
        return False
    return True


def check_project():
    """Check project modules."""
    print("\nChecking project imports...")

    try:
        from configs.default import Config
        print("  configs.default OK")
    except ImportError as e:
        print(f"  configs.default FAIL: {e}")
        return False

    try:
        from models.text_agent import TextAgent
        print("  models.text_agent OK")
    except ImportError as e:
        print(f"  models.text_agent FAIL: {e}")
        return False

    try:
        from models.mri_agent import MRIAgent
        print("  models.mri_agent OK")
    except ImportError as e:
        print(f"  models.mri_agent FAIL: {e}")
        return False

    try:
        from models.orchestrator import OrchestratorAgent
        print("  models.orchestrator OK")
    except ImportError as e:
        print(f"  models.orchestrator FAIL: {e}")
        return False

    try:
        from models.report_parser import parse_to_label_indices
        print("  models.report_parser OK")
    except ImportError as e:
        print(f"  models.report_parser FAIL: {e}")
        return False

    try:
        from data.dataset import load_label_maps, load_classification_csv
        print("  data.dataset OK")
    except ImportError as e:
        print(f"  data.dataset FAIL: {e}")
        return False

    return True


def check_data():
    """Check datasets."""
    import os
    print("\nChecking data...")

    files = [
        "external_data/MME/classification_gold.csv",
        "external_data/MME/label_maps.json",
    ]
    for f in files:
        if os.path.exists(f):
            print(f"  {os.path.basename(f)} OK")
        else:
            print(f"  {os.path.basename(f)} MISSING")


def main():
    print("=" * 50)
    print("MEAF v2 Setup Verification")
    print("=" * 50)

    ok = check_imports() and check_project()
    check_data()

    if ok:
        print("\nAll checks passed.")
    else:
        print("\nSome checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
