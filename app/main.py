# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
from typing import List, Optional
import logging
import traceback

from app.models import DetectionResponse, DetectionResult
from app.services import ModelService
from app.settings import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Threat Detection API",
    description="API for detecting threats in images and videos using YOLOv8",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model service
model_service = None


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global model_service
    try:
        logger.info("Initializing model service...")
        model_service = ModelService()
        model_path = Path(DETECTION_MODEL)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model_service.initialize_model(str(model_path))
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        logger.error(traceback.format_exc())
        raise


@app.get("/")
async def root():
    """Root endpoint to check API status"""
    return {"status": "online", "message": "Threat Detection API is running"}


def process_image(contents: bytes) -> np.ndarray:
    """Process uploaded image bytes into numpy array"""
    try:
        # First try using PIL
        image = Image.open(io.BytesIO(contents))
        image = image.convert('RGB')
        return np.array(image)
    except Exception as pil_error:
        logger.warning(f"PIL failed to open image: {str(pil_error)}")
        try:
            # Fallback to OpenCV
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV failed to decode image")
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as cv_error:
            logger.error(f"OpenCV failed to open image: {str(cv_error)}")
            raise ValueError(
                "Failed to process image with both PIL and OpenCV")


@app.post("/detect/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = 0.5
):
    """Detect threats in an uploaded image"""
    global model_service

    try:
        logger.info(f"Processing image detection request: {file.filename}")

        # Validate model service
        if model_service is None:
            raise RuntimeError("Model service not initialized")

        # Validate file type
        if file.content_type not in SUPPORTED_IMAGE_TYPES:
            logger.info(f"File type: {file.content_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Supported types: {SUPPORTED_IMAGE_TYPES}"
            )

        # Read and process image
        try:
            contents = await file.read()
            logger.info(f"Image size: {len(contents)} bytes")

            # Process image using our helper function
            image = process_image(contents)

            if image is None or image.size == 0:
                raise ValueError("Failed to load image")

            logger.info(f"Successfully loaded image with shape: {image.shape}")

        except Exception as e:
            logger.error(f"Error reading image: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=400, detail=f"Invalid image file: {str(e)}")

        # Validate confidence
        if not 0 <= confidence <= 1:
            raise HTTPException(
                status_code=400,
                detail="Confidence must be between 0 and 1"
            )

        # Perform detection
        logger.info("Performing detection...")
        results = model_service.detect_image(image, confidence)
        logger.info(
            f"Detection completed. Found {len(results.detections)} threats")

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    global model_service
    try:
        return {
            "model_path": str(DETECTION_MODEL),
            "status": "loaded" if model_service and model_service.model else "not_loaded",
            "initialized": model_service is not None
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
