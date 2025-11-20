import os
from pathlib import Path
import sys

# Base directory of the project (folder where this file lives)
BASE_DIR = Path(__file__).resolve().parent

# Default local paths (for running on your own machine / Jupyter)
DEFAULT_DATASET_ROOT = BASE_DIR / "dataset_ecssd"
DEFAULT_MODEL_DIR = BASE_DIR / "models"

# Some common hyperparameters
IMAGE_SIZE = 128
BATCH_SIZE = 8
NUM_EPOCHS = 20
NUM_WORKERS = 2


def running_in_colab() -> bool:
    """Heuristic: are we running inside a Colab notebook?"""
    return "google.colab" in sys.modules


def get_paths():
    """
    Returns dataset_root and model_dir depending on environment.
    - On Colab: use /content/... paths
    - Else: use project-local paths
    """
    if running_in_colab():
        dataset_root = Path("/content/dataset_ecssd")
        model_dir = Path("/content/drive/MyDrive/SOD")
    else:
        dataset_root = DEFAULT_DATASET_ROOT
        model_dir = DEFAULT_MODEL_DIR

    return dataset_root, model_dir
