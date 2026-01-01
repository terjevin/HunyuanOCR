"""System configuration for distributed OCR pipeline."""
import os
from typing import List

# Worker Configuration
WORKER_NODES: List[dict] = [
    {"host": "158.39.75.13", "port": 8001, "name": "linux1"},
    {"host": "158.37.66.212", "port": 8001, "name": "linux2"},
    {"host": "158.39.75.48", "port": 8001, "name": "linux3"},
]

# Coordinator Configuration
COORDINATOR_HOST = os.getenv("COORDINATOR_HOST", "0.0.0.0")
COORDINATOR_PORT = int(os.getenv("COORDINATOR_PORT", "8000"))

# Worker Configuration
WORKER_HOST = os.getenv("WORKER_HOST", "0.0.0.0")
WORKER_PORT = int(os.getenv("WORKER_PORT", "8001"))

# Resource Limits
MAX_RAM_GB = int(os.getenv("MAX_RAM_GB", "22"))
MAX_CPU_CORES = int(os.getenv("MAX_CPU_CORES", "12"))

# OCR Configuration
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "en")
OCR_CPU_THREADS = int(os.getenv("OCR_CPU_THREADS", "10"))
ENABLE_MKLDNN = os.getenv("ENABLE_MKLDNN", "true").lower() == "true"

# Temporary Storage
TEMP_DIR = os.getenv("TEMP_DIR", "/dev/shm/ocr_temp")

# Retry Configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", "2"))

# Timeout Configuration
WORKER_REQUEST_TIMEOUT = int(os.getenv("WORKER_REQUEST_TIMEOUT", "120"))  # seconds

# PDF Configuration
DEFAULT_DPI = int(os.getenv("DEFAULT_DPI", "200"))
