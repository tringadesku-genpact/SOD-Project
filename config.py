from pathlib import Path

# Repo folder
BASE_DIR = Path(__file__).resolve().parent

COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/SOD")

if COLAB_DRIVE_ROOT.exists():
    STORAGE_ROOT = COLAB_DRIVE_ROOT
else:
    # Fallback: store everything next to the project (local machine)
    STORAGE_ROOT = BASE_DIR

# Paths for dataset, models, and outputs
DATASET_DIR = STORAGE_ROOT / "dataset_ecssd"
MODELS_DIR  = STORAGE_ROOT / "models"
OUTPUTS_DIR = STORAGE_ROOT / "outputs" 

def get_paths():
    return str(DATASET_DIR), str(MODELS_DIR)

# Hyperparameters
IMAGE_SIZE    = 128   # switch to 224 for testing
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
