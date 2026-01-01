"""FastAPI Coordinator Server for distributed OCR processing."""
import os
import sys
import base64
import time
import uuid
import asyncio
import logging
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import httpx
from pdf2image import convert_from_bytes
from PIL import Image

# Add parent directory to path for shared imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.models import (
    OCRImageRequest, OCRResult, OCRJobResponse, JobMetrics,
    WorkerMetrics, WorkersStatusResponse
)
from shared.config import (
    WORKER_NODES, COORDINATOR_HOST, COORDINATOR_PORT,
    MAX_RETRIES, RETRY_DELAY_SECONDS, DEFAULT_DPI, TEMP_DIR,
    WORKER_REQUEST_TIMEOUT
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OCR Coordinator Server",
    description="Coordinator for distributed OCR processing across multiple workers",
    version="1.0.0"
)

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)


async def check_worker_health(client: httpx.AsyncClient, worker: dict) -> dict:
    """Check health status of a worker."""
    try:
        url = f"http://{worker['host']}:{worker['port']}/health"
        response = await client.get(url, timeout=5.0)
        
        if response.status_code == 200:
            health_data = response.json()
            return {
                "name": worker["name"],
                "host": worker["host"],
                "port": worker["port"],
                "status": "healthy",
                "health": health_data
            }
        else:
            return {
                "name": worker["name"],
                "host": worker["host"],
                "port": worker["port"],
                "status": "unhealthy",
                "error": f"HTTP {response.status_code}"
            }
    except Exception as e:
        return {
            "name": worker["name"],
            "host": worker["host"],
            "port": worker["port"],
            "status": "unreachable",
            "error": str(e)
        }


async def send_ocr_request(
    client: httpx.AsyncClient,
    worker: dict,
    image_base64: str,
    language: str,
    page_number: int,
    retry_count: int = 0
) -> OCRResult:
    """Send OCR request to a worker with retry logic."""
    url = f"http://{worker['host']}:{worker['port']}/ocr"
    
    request_data = OCRImageRequest(
        image_base64=image_base64,
        language=language,
        page_number=page_number
    )
    
    try:
        response = await client.post(
            url,
            json=request_data.model_dump(),
            timeout=WORKER_REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            result_data = response.json()
            return OCRResult(**result_data)
        else:
            error_msg = f"Worker {worker['name']} returned HTTP {response.status_code}"
            logger.error(f"{error_msg} for page {page_number}")
            
            if retry_count < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY_SECONDS * (2 ** retry_count))  # Exponential backoff
                logger.info(f"Retrying page {page_number} (attempt {retry_count + 2}/{MAX_RETRIES + 1})")
                return await send_ocr_request(client, worker, image_base64, language, page_number, retry_count + 1)
            
            return OCRResult(
                page_number=page_number,
                error=error_msg,
                worker_name=worker['name']
            )
            
    except Exception as e:
        error_msg = f"Worker {worker['name']} error: {str(e)}"
        logger.error(f"{error_msg} for page {page_number}")
        
        if retry_count < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY_SECONDS * (2 ** retry_count))  # Exponential backoff
            logger.info(f"Retrying page {page_number} (attempt {retry_count + 2}/{MAX_RETRIES + 1})")
            return await send_ocr_request(client, worker, image_base64, language, page_number, retry_count + 1)
        
        return OCRResult(
            page_number=page_number,
            error=error_msg,
            worker_name=worker['name']
        )


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


async def process_pdf_pages(
    images: List[Image.Image],
    language: str,
    job_id: str
) -> List[OCRResult]:
    """Distribute PDF pages across workers using round-robin."""
    total_pages = len(images)
    logger.info(f"Job {job_id}: Processing {total_pages} pages")
    
    # Filter healthy workers
    async with httpx.AsyncClient() as client:
        worker_health_checks = [check_worker_health(client, worker) for worker in WORKER_NODES]
        health_results = await asyncio.gather(*worker_health_checks)
    
    healthy_workers = [w for w in health_results if w["status"] == "healthy"]
    
    if not healthy_workers:
        logger.error(f"Job {job_id}: No healthy workers available")
        return [
            OCRResult(page_number=i+1, error="No healthy workers available")
            for i in range(total_pages)
        ]
    
    logger.info(f"Job {job_id}: {len(healthy_workers)} healthy workers available")
    
    # Prepare tasks with round-robin distribution
    tasks = []
    async with httpx.AsyncClient() as client:
        for i, image in enumerate(images):
            worker_idx = i % len(healthy_workers)
            worker = healthy_workers[worker_idx]
            
            # Find original worker config
            worker_config = next((w for w in WORKER_NODES if w['name'] == worker['name']), None)
            
            image_base64 = image_to_base64(image)
            page_number = i + 1
            
            task = send_ocr_request(client, worker_config, image_base64, language, page_number)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
    
    return results


def calculate_metrics(
    job_id: str,
    filename: str,
    results: List[OCRResult],
    start_time: datetime,
    file_size_bytes: int
) -> JobMetrics:
    """Calculate comprehensive metrics for the job."""
    end_time = datetime.utcnow()
    total_duration_ms = (end_time - start_time).total_seconds() * 1000
    
    # Basic counts
    total_pages = len(results)
    pages_processed = sum(1 for r in results if r.error is None)
    pages_failed = total_pages - pages_processed
    
    # Text statistics
    total_words = sum(r.word_count for r in results if r.error is None)
    total_chars = sum(r.char_count for r in results if r.error is None)
    total_lines = sum(r.line_count for r in results if r.error is None)
    
    # Confidence
    confidences = [r.avg_confidence for r in results if r.error is None and r.avg_confidence > 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    # Timing
    processing_times = [r.processing_time_ms for r in results if r.error is None]
    avg_page_time_ms = sum(processing_times) / len(processing_times) if processing_times else 0.0
    
    # Worker distribution
    ocr_rounds_per_worker: Dict[str, int] = {}
    worker_metrics_dict: Dict[str, Dict] = {}
    
    for result in results:
        if result.worker_name:
            # Count pages per worker
            ocr_rounds_per_worker[result.worker_name] = ocr_rounds_per_worker.get(result.worker_name, 0) + 1
            
            # Collect worker metrics
            if result.worker_name not in worker_metrics_dict:
                worker_metrics_dict[result.worker_name] = {
                    "worker_name": result.worker_name,
                    "pages_processed": 0,
                    "pages_failed": 0,
                    "total_time_ms": 0.0,
                    "errors": []
                }
            
            if result.error:
                worker_metrics_dict[result.worker_name]["pages_failed"] += 1
                worker_metrics_dict[result.worker_name]["errors"].append(
                    f"Page {result.page_number}: {result.error}"
                )
            else:
                worker_metrics_dict[result.worker_name]["pages_processed"] += 1
                worker_metrics_dict[result.worker_name]["total_time_ms"] += result.processing_time_ms
    
    # Convert to WorkerMetrics objects
    worker_metrics = []
    for wm_dict in worker_metrics_dict.values():
        avg_time = 0.0
        if wm_dict["pages_processed"] > 0:
            avg_time = wm_dict["total_time_ms"] / wm_dict["pages_processed"]
        
        worker_metrics.append(WorkerMetrics(
            worker_name=wm_dict["worker_name"],
            pages_processed=wm_dict["pages_processed"],
            pages_failed=wm_dict["pages_failed"],
            total_time_ms=round(wm_dict["total_time_ms"], 2),
            avg_time_ms=round(avg_time, 2),
            errors=wm_dict["errors"]
        ))
    
    # Collect errors
    errors = [f"Page {r.page_number}: {r.error}" for r in results if r.error]
    
    # Determine status
    if pages_failed == 0:
        status = "success"
    elif pages_processed > 0:
        status = "partial_success"
    else:
        status = "failed"
    
    return JobMetrics(
        job_id=job_id,
        filename=filename,
        total_pages=total_pages,
        pages_processed=pages_processed,
        pages_failed=pages_failed,
        total_words=total_words,
        total_chars=total_chars,
        total_lines=total_lines,
        avg_confidence=round(avg_confidence, 4),
        start_time=start_time.isoformat() + "Z",
        end_time=end_time.isoformat() + "Z",
        total_duration_ms=round(total_duration_ms, 2),
        avg_page_time_ms=round(avg_page_time_ms, 2),
        file_size_bytes=file_size_bytes,
        file_size_mb=round(file_size_bytes / 1024 / 1024, 2),
        ocr_rounds_per_worker=ocr_rounds_per_worker,
        worker_metrics=worker_metrics,
        status=status,
        errors=errors
    )


@app.get("/health")
async def health_check():
    """Coordinator health check."""
    return {
        "status": "healthy",
        "service": "OCR Coordinator",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/workers/status")
async def workers_status():
    """Get status of all workers."""
    async with httpx.AsyncClient() as client:
        worker_health_checks = [check_worker_health(client, worker) for worker in WORKER_NODES]
        results = await asyncio.gather(*worker_health_checks)
    
    healthy_count = sum(1 for r in results if r["status"] == "healthy")
    
    return WorkersStatusResponse(
        total_workers=len(WORKER_NODES),
        healthy_workers=healthy_count,
        workers=results
    )


@app.post("/ocr/pdf")
async def process_pdf(
    file: UploadFile = File(...),
    language: str = Form("en"),
    dpi: int = Form(DEFAULT_DPI)
):
    """Process a PDF file with distributed OCR."""
    job_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    logger.info(f"Job {job_id}: Processing PDF '{file.filename}'")
    
    try:
        # Read PDF file
        pdf_bytes = await file.read()
        file_size_bytes = len(pdf_bytes)
        
        logger.info(f"Job {job_id}: Converting PDF to images (DPI: {dpi})")
        
        # Convert PDF to images
        try:
            images = convert_from_bytes(pdf_bytes, dpi=dpi)
            logger.info(f"Job {job_id}: Converted to {len(images)} images")
        except Exception as e:
            logger.error(f"Job {job_id}: PDF conversion failed: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to convert PDF: {str(e)}")
        
        # Process pages
        results = await process_pdf_pages(images, language, job_id)
        
        # Calculate metrics
        metrics = calculate_metrics(job_id, file.filename, results, start_time, file_size_bytes)
        
        # Generate combined text
        combined_text = "\n\n".join([
            f"--- Page {r.page_number} ---\n{r.text}"
            for r in results if r.error is None and r.text
        ])
        
        # Generate markdown
        combined_markdown = "\n\n".join([
            f"## Page {r.page_number}\n\n{r.text}"
            for r in results if r.error is None and r.text
        ])
        
        logger.info(f"Job {job_id}: Completed - {metrics.status}")
        
        return OCRJobResponse(
            metrics=metrics,
            results=results,
            combined_text=combined_text,
            combined_markdown=combined_markdown
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job {job_id}: Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/image")
async def process_image(
    file: UploadFile = File(...),
    language: str = Form("en")
):
    """Process a single image with OCR."""
    job_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    logger.info(f"Job {job_id}: Processing image '{file.filename}'")
    
    try:
        # Read image file
        image_bytes = await file.read()
        file_size_bytes = len(image_bytes)
        
        # Open image
        try:
            image = Image.open(BytesIO(image_bytes))
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
        except Exception as e:
            logger.error(f"Job {job_id}: Failed to open image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Process image
        results = await process_pdf_pages([image], language, job_id)
        
        # Calculate metrics
        metrics = calculate_metrics(job_id, file.filename, results, start_time, file_size_bytes)
        
        # Get result
        result = results[0] if results else OCRResult(page_number=1, error="No result returned")
        
        logger.info(f"Job {job_id}: Completed - {metrics.status}")
        
        return OCRJobResponse(
            metrics=metrics,
            results=[result],
            combined_text=result.text if result.error is None else "",
            combined_markdown=f"# {file.filename}\n\n{result.text}" if result.error is None else ""
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job {job_id}: Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "OCR Coordinator Server",
        "status": "running",
        "workers": len(WORKER_NODES),
        "endpoints": [
            "POST /ocr/pdf - Process PDF file",
            "POST /ocr/image - Process single image",
            "GET /workers/status - Check worker status",
            "GET /health - Health check"
        ]
    }


if __name__ == "__main__":
    logger.info("Starting OCR Coordinator Server")
    logger.info(f"Host: {COORDINATOR_HOST}, Port: {COORDINATOR_PORT}")
    logger.info(f"Workers: {len(WORKER_NODES)}")
    
    uvicorn.run(
        app,
        host=COORDINATOR_HOST,
        port=COORDINATOR_PORT,
        log_level="info"
    )
