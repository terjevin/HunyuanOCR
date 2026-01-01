"""FastAPI Worker Server for OCR processing using PaddleOCR."""
import os
import sys
import base64
import time
import logging
from io import BytesIO
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import psutil
import numpy as np

# Add parent directory to path for shared imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.models import OCRImageRequest, OCRResult, OCRBox, WorkerHealth
from shared.config import (
    WORKER_HOST, WORKER_PORT, OCR_CPU_THREADS, ENABLE_MKLDNN, TEMP_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
ocr_engine = None
start_time = time.time()
total_requests = 0
worker_name = os.getenv("WORKER_NAME", f"worker-{os.getpid()}")

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)


def initialize_ocr():
    """Initialize PaddleOCR engine with CPU optimization.
    
    Note: PaddleOCR language is set at initialization and cannot be changed
    dynamically without reinitialization. The default language is 'en', which
    works well for Latin-based scripts. For other languages, set the OCR_LANGUAGE
    environment variable before starting the worker.
    """
    global ocr_engine
    try:
        from paddleocr import PaddleOCR
        
        ocr_language = os.getenv("OCR_LANGUAGE", "en")
        
        logger.info("Initializing PaddleOCR with CPU optimization...")
        logger.info(f"Language: {ocr_language}, CPU Threads: {OCR_CPU_THREADS}, MKLDNN: {ENABLE_MKLDNN}")
        
        ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang=ocr_language,
            use_gpu=False,
            enable_mkldnn=ENABLE_MKLDNN,
            cpu_threads=OCR_CPU_THREADS,
            show_log=False,
            det_db_box_thresh=0.3,
            det_db_thresh=0.3,
        )
        
        logger.info(f"PaddleOCR initialized successfully with language: {ocr_language}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize PaddleOCR: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info(f"Starting OCR Worker Server: {worker_name}")
    success = initialize_ocr()
    if not success:
        logger.error("Failed to initialize OCR engine. Worker may not function properly.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down OCR Worker Server")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="OCR Worker Server",
    description="Worker node for distributed OCR processing using PaddleOCR",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint with resource monitoring."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = process.memory_percent()
        uptime = time.time() - start_time
        
        health = WorkerHealth(
            status="healthy" if ocr_engine is not None else "unhealthy",
            worker_name=worker_name,
            cpu_percent=round(cpu_percent, 2),
            memory_percent=round(memory_percent, 2),
            memory_mb=round(memory_mb, 2),
            uptime_seconds=round(uptime, 2),
            total_requests=total_requests,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        return health
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr")
async def process_ocr(request: OCRImageRequest):
    """Process a single image with OCR."""
    global total_requests
    total_requests += 1
    
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR engine not initialized")
    
    start_time_ms = time.time() * 1000
    
    try:
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            # Convert to numpy array for PaddleOCR
            img_array = np.array(image)
            
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
        # Perform OCR
        try:
            # PaddleOCR expects numpy array
            result = ocr_engine.ocr(img_array, cls=True)
            
            if result is None or len(result) == 0 or result[0] is None:
                # No text detected
                processing_time = (time.time() * 1000) - start_time_ms
                return OCRResult(
                    page_number=request.page_number,
                    boxes=[],
                    text="",
                    word_count=0,
                    char_count=0,
                    line_count=0,
                    avg_confidence=0.0,
                    processing_time_ms=round(processing_time, 2),
                    worker_name=worker_name
                )
            
            # Parse results
            boxes = []
            all_text = []
            confidences = []
            
            for line in result[0]:
                if line is None:
                    continue
                    
                coords = line[0]  # Bounding box coordinates
                text_info = line[1]  # (text, confidence)
                
                text = text_info[0]
                confidence = text_info[1]
                
                # Convert numpy arrays to lists for JSON serialization
                coords_list = [[int(x), int(y)] for x, y in coords]
                
                boxes.append(OCRBox(
                    coords=coords_list,
                    text=text,
                    confidence=float(confidence)
                ))
                
                all_text.append(text)
                confidences.append(confidence)
            
            # Combine text
            combined_text = "\n".join(all_text)
            
            # Calculate statistics
            word_count = sum(len(text.split()) for text in all_text)
            char_count = sum(len(text) for text in all_text)
            line_count = len(all_text)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            processing_time = (time.time() * 1000) - start_time_ms
            
            return OCRResult(
                page_number=request.page_number,
                boxes=boxes,
                text=combined_text,
                word_count=word_count,
                char_count=char_count,
                line_count=line_count,
                avg_confidence=round(avg_confidence, 4),
                processing_time_ms=round(processing_time, 2),
                worker_name=worker_name
            )
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            processing_time = (time.time() * 1000) - start_time_ms
            
            return OCRResult(
                page_number=request.page_number,
                boxes=[],
                text="",
                word_count=0,
                char_count=0,
                line_count=0,
                avg_confidence=0.0,
                processing_time_ms=round(processing_time, 2),
                worker_name=worker_name,
                error=f"OCR processing failed: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in OCR processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "OCR Worker Server",
        "worker_name": worker_name,
        "status": "running",
        "ocr_engine": "PaddleOCR",
        "uptime_seconds": round(time.time() - start_time, 2)
    }


if __name__ == "__main__":
    logger.info(f"Starting OCR Worker Server: {worker_name}")
    logger.info(f"Host: {WORKER_HOST}, Port: {WORKER_PORT}")
    logger.info(f"Temp Directory: {TEMP_DIR}")
    
    uvicorn.run(
        app,
        host=WORKER_HOST,
        port=WORKER_PORT,
        log_level="info"
    )
