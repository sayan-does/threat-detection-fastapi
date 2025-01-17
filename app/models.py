from pathlib import Path
import sys
from pydantic import BaseModel
from typing import List, Optional

class DetectionResult(BaseModel):
    """Single detection result from the model"""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

class DetectionResponse(BaseModel):
    """Response model for detection endpoints"""
    detections: List[DetectionResult]
    processing_time: float
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
SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/bmp", "image/webp"]
SUPPORTED_VIDEO_TYPES = ["video/mp4", "video/x-msvideo"]

# Logging config
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'