# app/services.py
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import time
from ultralytics import YOLO
from typing import List, Optional, Union
import logging

from app.models import DetectionResponse, DetectionResult

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.model = None

    def initialize_model(self, model_path: str):
        try:
            logger.info(f"Loading model from {model_path}")
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            self.model = YOLO(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _process_results(self, results) -> DetectionResponse:
       
        try:
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    detection = DetectionResult(
                        class_id=int(box.cls),
                        class_name=result.names[int(box.cls)],
                        confidence=float(box.conf),
                        bbox=box.xyxy[0].tolist()
                    )
                    detections.append(detection)

            return DetectionResponse(
                detections=detections,
                processing_time=results[0].speed.get('inference', 0)
            )
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            raise

    def detect_image(self, image: Union[np.ndarray, Image.Image], conf: float) -> DetectionResponse:
        try:
            if self.model is None:
                raise RuntimeError("Model not initialized")

            # PIL Image to numpy array
            if isinstance(image, Image.Image):
                image = np.array(image)

            # image is in correct format
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)

            logger.info(f"Running inference on image with shape: {image.shape}")
            results = self.model.predict(image, conf=conf)
            logger.info("Inference completed")
            
            return self._process_results(results)
            
        except Exception as e:
            logger.error(f"Error in detect_image: {str(e)}")
            raise