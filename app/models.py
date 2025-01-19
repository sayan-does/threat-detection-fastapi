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
    image_data: Optional[str] = None  # Base64 encoded image data
    video_data: Optional[str] = None  # Base64 encoded video data

# Path configurations
file_path = Path(__file__).resolve()
root_path = file_path.parent

if root_path not in sys.path:
    sys.path.append(str(root_path))

# ML Model config
DETECTION_MODEL = root_path / 'data' / 'models' / 'BestFinal.pt'

# API Settings
MAX_UPLOAD_SIZE = 1000 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/bmp", "image/webp"]
SUPPORTED_VIDEO_TYPES = ["video/mp4", "video/x-msvideo"]

# Logging config
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
