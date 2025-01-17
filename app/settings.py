from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# ML Model config
DETECTION_MODEL = root_path / 'data' / 'models' / 'BestFinal.pt'

# API Settings
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB
SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png",
                         "image/bmp", "image/webp", "image/gif", "image/tiff"]
SUPPORTED_VIDEO_TYPES = ["video/mp4", "video/x-msvideo"]

# Logging config
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'