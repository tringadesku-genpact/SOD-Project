from pathlib import Path

# This file controls where data/models/outputs live.

# Folder where the repo itself lives
BASE_DIR = Path(__file__).resolve().parent

# If we're in Colab with Drive mounted, use that.
# You can change "SOD" to whatever folder name you like in MyDrive.
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/SOD")

if COLAB_DRIVE_ROOT.exists():
    STORAGE_ROOT = COLAB_DRIVE_ROOT
else:
    # Fallback: store everything next to the project (local machine)
    STORAGE_ROOT = BASE_DIR

# Paths for dataset, models, and outputs
DATASET_DIR = STORAGE_ROOT / "dataset_ecssd"
MODELS_DIR  = STORAGE_ROOT / "models"
OUTPUTS_DIR = STORAGE_ROOT / "outputs"  # you can keep this in BASE_DIR if you prefer

def get_paths():
    """Return (dataset_root, model_dir) as strings for compatibility."""
    return str(DATASET_DIR), str(MODELS_DIR)

# Hyperparameters
IMAGE_SIZE    = 224   # higher res for better detail
BATCH_SIZE    = 8
NUM_EPOCHS    = 25
LEARNING_RATE = 1e-3

# DataLoader workers (0 is safest on Windows)
NUM_WORKERS   = 0

# Train/val/test split
TRAIN_FRACTION = 0.7
VAL_FRACTION   = 0.15

# Seed
SEED = 42
