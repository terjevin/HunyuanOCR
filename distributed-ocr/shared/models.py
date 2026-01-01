"""Pydantic models for distributed OCR system."""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class OCRBox(BaseModel):
    """Bounding box for detected text."""
    coords: List[List[int]] = Field(..., description="Bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]")
    text: str = Field(..., description="Detected text")
    confidence: float = Field(..., description="Detection confidence score")


class OCRResult(BaseModel):
    """Result from OCR processing of a single image."""
    page_number: int = Field(..., description="Page number (1-indexed)")
    boxes: List[OCRBox] = Field(default_factory=list, description="Detected text boxes")
    text: str = Field(default="", description="Combined text from all boxes")
    word_count: int = Field(default=0, description="Number of words detected")
    char_count: int = Field(default=0, description="Number of characters detected")
    line_count: int = Field(default=0, description="Number of lines detected")
    avg_confidence: float = Field(default=0.0, description="Average confidence score")
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")
    worker_name: Optional[str] = Field(None, description="Name of worker that processed this page")
    error: Optional[str] = Field(None, description="Error message if processing failed")


class OCRImageRequest(BaseModel):
    """Request for processing a single image."""
    image_base64: str = Field(..., description="Base64-encoded image data")
    language: str = Field(default="en", description="OCR language (en, ch, etc.)")
    page_number: int = Field(default=1, description="Page number for tracking")


class WorkerMetrics(BaseModel):
    """Metrics for a single worker."""
    worker_name: str = Field(..., description="Worker identifier")
    pages_processed: int = Field(default=0, description="Number of pages successfully processed")
    pages_failed: int = Field(default=0, description="Number of pages that failed")
    total_time_ms: float = Field(default=0.0, description="Total processing time in milliseconds")
    avg_time_ms: float = Field(default=0.0, description="Average processing time per page")
    errors: List[str] = Field(default_factory=list, description="List of error messages")


class JobMetrics(BaseModel):
    """Comprehensive metrics for an OCR job."""
    job_id: str = Field(..., description="Unique job identifier")
    filename: str = Field(..., description="Original filename")
    total_pages: int = Field(..., description="Total number of pages")
    pages_processed: int = Field(default=0, description="Number of pages successfully processed")
    pages_failed: int = Field(default=0, description="Number of pages that failed")
    
    total_words: int = Field(default=0, description="Total word count across all pages")
    total_chars: int = Field(default=0, description="Total character count across all pages")
    total_lines: int = Field(default=0, description="Total line count across all pages")
    avg_confidence: float = Field(default=0.0, description="Average confidence score across all pages")
    
    start_time: str = Field(..., description="Job start timestamp (ISO 8601)")
    end_time: Optional[str] = Field(None, description="Job end timestamp (ISO 8601)")
    total_duration_ms: float = Field(default=0.0, description="Total job duration in milliseconds")
    avg_page_time_ms: float = Field(default=0.0, description="Average processing time per page")
    
    file_size_bytes: int = Field(default=0, description="Original file size in bytes")
    file_size_mb: float = Field(default=0.0, description="Original file size in megabytes")
    
    ocr_rounds_per_worker: Dict[str, int] = Field(default_factory=dict, description="Pages processed per worker")
    worker_metrics: List[WorkerMetrics] = Field(default_factory=list, description="Per-worker breakdown")
    
    status: str = Field(..., description="Job status: success/partial_success/failed")
    errors: List[str] = Field(default_factory=list, description="List of error messages")


class OCRJobResponse(BaseModel):
    """Response for OCR job."""
    metrics: JobMetrics = Field(..., description="Job metrics")
    results: List[OCRResult] = Field(default_factory=list, description="Per-page OCR results")
    combined_text: str = Field(default="", description="Combined text from all pages")
    combined_markdown: str = Field(default="", description="Combined text in markdown format")


class WorkerHealth(BaseModel):
    """Worker health status."""
    status: str = Field(..., description="Health status: healthy/unhealthy")
    worker_name: str = Field(..., description="Worker identifier")
    cpu_percent: float = Field(..., description="CPU usage percentage")
    memory_percent: float = Field(..., description="Memory usage percentage")
    memory_mb: float = Field(..., description="Memory usage in MB")
    uptime_seconds: float = Field(..., description="Uptime in seconds")
    total_requests: int = Field(default=0, description="Total requests processed")
    timestamp: str = Field(..., description="Timestamp (ISO 8601)")


class WorkersStatusResponse(BaseModel):
    """Response for workers status check."""
    total_workers: int = Field(..., description="Total number of configured workers")
    healthy_workers: int = Field(default=0, description="Number of healthy workers")
    workers: List[Dict] = Field(default_factory=list, description="Worker status details")
