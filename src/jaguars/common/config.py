from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
TEST_FIXTURES_DIR = DATA_DIR / "test_fixtures"

# FiftyOne Dataset Names
JID_MASTER_DATASET = "JID_Master_Dataset"
DATASET_REID_CROPS = "JID_ReID_Crops"

# FiftyOne Group Field Configuration
GROUP_FIELD_NAME = "group"
DEFAULT_GROUP_SLICE = "image"

# Processing
DEFAULT_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
DEFAULT_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
