from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask 
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
from fastapi.responses import FileResponse
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
import base64
import os

from app.models import DetectionResponse, DetectionResult
from app.services import ModelService
from app.settings import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api.log')
    ]
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
    global model_service
    try:
        logger.info("Initializing model service...") #from yolo, putting in my wights
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
    try:
        # First try using PIL
        image = Image.open(io.BytesIO(contents))
        image = image.convert('RGB')
        return np.array(image)
    except Exception as pil_error:
        logger.warning(f"PIL failed to open image: {str(pil_error)}")
        try:
            # open cv trial
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV failed to decode image")
            # convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as cv_error:
            logger.error(f"OpenCV failed to open image: {str(cv_error)}")
            raise ValueError(
                "Failed to process image with both PIL and OpenCV")


@app.post("/detect/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = 0.5,
    return_image: bool = True  # Add optional parameter to control image return
):
    """
    Detect threats in an uploaded image and optionally return the processed image
    """
    try:
        logger.info(f"Processing image detection request: {file.filename}")

        # check model service
        if model_service is None:
            raise RuntimeError("Model service not initialized")

        # Validate file type
        if file.content_type not in SUPPORTED_IMAGE_TYPES:
            logger.warning(f"Unsupported file type: {file.content_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Supported types: {SUPPORTED_IMAGE_TYPES}"
            )

        # read and process image
        try:
            contents = await file.read()
            logger.info(f"Read image data: {len(contents)} bytes")

            image = process_image(contents)
            logger.info(f"Processed image shape: {image.shape}")

            if image is None or image.size == 0:
                raise ValueError("Failed to load image")

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process image: {str(e)}"
            )

        # confidence
        if not 0 <= confidence <= 1:
            raise HTTPException(
                status_code=400,
                detail="Confidence must be between 0 and 1"
            )

        # detection
        try:
            logger.info("Running detection...")
            results = model_service.detect_image(image, confidence)
            logger.info(
                f"Detection completed with {len(results.detections)} results")

        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Detection failed: {str(e)}"
            )

        try:
            image_rgb = image.copy()
            for detection in results.detections:
                x1, y1, x2, y2 = detection.bbox
                cv2.rectangle(
                    image_rgb,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    image_rgb,
                    f"{detection.class_name} ({detection.confidence:.2f})",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

            # Convert to BGR for OpenCV operations
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Only encode image if return_image is True
            image_data = None
            if return_image:
                _, buffer = cv2.imencode(".jpg", image_bgr)
                image_data = base64.b64encode(buffer.tobytes()).decode('utf-8')

            return DetectionResponse(
                detections=results.detections,
                processing_time=results.processing_time,
                image_data=image_data
            )

        except Exception as e:
            logger.error(f"Error drawing detections: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process detection results: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/detect/image/download", response_class=FileResponse)
async def detect_image_download(
    file: UploadFile = File(...),
    confidence: float = 0.5
):
    """Detect threats in an uploaded image and return a downloadable image"""
    try:
        logger.info(f"Processing image download request: {file.filename}")

        # Validate model service
        if model_service is None:
            raise RuntimeError("Model service not initialized")

        # Validate file type
        if file.content_type not in SUPPORTED_IMAGE_TYPES:
            logger.warning(f"Unsupported file type: {file.content_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Supported types: {SUPPORTED_IMAGE_TYPES}"
            )

        # Read and process image
        try:
            contents = await file.read()
            logger.info(f"Read image data: {len(contents)} bytes")

            image = process_image(contents)
            logger.info(f"Processed image shape: {image.shape}")

            if image is None or image.size == 0:
                raise ValueError("Failed to load image")

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process image: {str(e)}"
            )

        # Validate confidence
        if not 0 <= confidence <= 1:
            raise HTTPException(
                status_code=400,
                detail="Confidence must be between 0 and 1"
            )

        # Perform detection
        try:
            logger.info("Running detection...")
            results = model_service.detect_image(image, confidence)
            logger.info(
                f"Detection completed with {len(results.detections)} results")

        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Detection failed: {str(e)}"
            )

        # Draw bounding boxes and save image
        try:
            image_rgb = image.copy()
            for detection in results.detections:
                x1, y1, x2, y2 = detection.bbox
                cv2.rectangle(
                    image_rgb,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    image_rgb,
                    f"{detection.class_name} ({detection.confidence:.2f})",
                    (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

            # Convert to BGR for OpenCV operations
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Create a temporary file with a context manager
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, image_bgr)
                logger.info(
                    f"Saved processed image to temporary file: {temp_path}")

            # Return the file response with background cleanup task
            return FileResponse(
                temp_path,
                media_type='image/jpeg',
                filename=f'processed_{file.filename}',
                background=BackgroundTask(lambda: os.unlink(temp_path))
            )

        except Exception as e:
            logger.error(f"Error saving detection results: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save detection results: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )



# @app.post("/detect/video/download", response_class=FileResponse)
# async def detect_video_download(
#     file: UploadFile = File(...),
#     confidence: float = 0.5
# ):
#     """Detect threats in an uploaded video and return a downloadable video"""
#     global model_service

#     try:
#         logger.info(f"Processing video detection request: {file.filename}")

#         # Validate model service
#         if model_service is None:
#             raise RuntimeError("Model service not initialized")

#         # Validate file type
#         if file.content_type not in SUPPORTED_VIDEO_TYPES:
#             logger.info(f"File type: {file.content_type}")
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Invalid file type. Supported types: {SUPPORTED_VIDEO_TYPES}"
#             )

#         # Read and process video
#         try:
#             contents = await file.read()
#             logger.info(f"Video size: {len(contents)} bytes")

#             # Process video using our helper function
#             temp_file = tempfile.NamedTemporaryFile(suffix='.mp4')
#             temp_file.write(contents)
#             temp_file.seek(0)

#             cap = cv2.VideoCapture(temp_file.name)
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 # Perform detection
#                 results = model_service.detect_image(frame, confidence)

#                 # Draw bounding boxes on the frame
#                 for detection in results.detections:
#                     x1, y1, x2, y2 = detection.bbox
#                     cv2.rectangle(frame, (int(x1), int(y1)),
#                                   (int(x2), int(y2)), (0, 255, 0), 2)
#                     cv2.putText(frame, f"{detection.class_name} ({detection.confidence:.2f})", (int(
#                         x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#                 out.write(frame)

#             cap.release()
#             out.release()
#             cv2.destroyAllWindows()

#             # Return the modified video as a downloadable file
#             return FileResponse('output.mp4', media_type='video/mp4', filename='output.mp4')

#         except Exception as e:
#             logger.error(f"Error reading video: {str(e)}")
#             logger.error(traceback.format_exc())
#             raise HTTPException(
#                 status_code=400, detail=f"Invalid video file: {str(e)}")

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error processing video: {str(e)}")
#         logger.error(traceback.format_exc())
#         raise HTTPException(
#             status_code=500,
#             detail=f"Internal server error: {str(e)}")


# @app.get("/model/info")
# async def get_model_info():
#     """Get information about the loaded model"""
#     global model_service
#     try:
#         return {
#             "model_path": str(DETECTION_MODEL),
#             "status": "loaded" if model_service and model_service.model else "not_loaded",
#             "initialized": model_service is not None
#         }
#     except Exception as e:
#         logger.error(f"Error getting model info: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
