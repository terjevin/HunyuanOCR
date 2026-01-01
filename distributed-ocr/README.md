# Distributed OCR Pipeline System

A complete distributed OCR pipeline system that runs on CPU-only VMs without GPU requirements. Uses PaddleOCR for text recognition and distributes workload across multiple worker nodes.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Applications                      │
│                    (PDF/Image Upload via REST API)              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Coordinator Node (linux3)                     │
│                        158.39.75.48:8000                        │
│                                                                  │
│  • PDF-to-Image Conversion (pdf2image)                          │
│  • Round-Robin Load Balancing                                   │
│  • Result Aggregation                                           │
│  • Comprehensive Metrics Collection                             │
│  • Retry Logic (3 attempts with exponential backoff)           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
                    ▼           ▼           ▼
         ┌──────────────┬──────────────┬──────────────┐
         │  Worker 1    │  Worker 2    │  Worker 3    │
         │   linux1     │   linux2     │   linux3     │
         │158.39.75.13  │158.37.66.212 │158.39.75.48  │
         │    :8001     │    :8001     │    :8001     │
         │              │              │              │
         │  PaddleOCR   │  PaddleOCR   │  PaddleOCR   │
         │  + MKLDNN    │  + MKLDNN    │  + MKLDNN    │
         │  CPU Optim   │  CPU Optim   │  CPU Optim   │
         └──────────────┴──────────────┴──────────────┘
              │                │                │
              └────────────────┴────────────────┘
                               │
                               ▼
                    Processed OCR Results
                    (Text + Metrics)
```

### System Components

- **1 Coordinator Node** (linux3 - 158.39.75.48:8000)
  - Handles PDF upload and validation
  - Converts PDF pages to images using pdf2image/poppler
  - Distributes pages across workers using round-robin
  - Aggregates results and generates comprehensive metrics
  - Provides REST API for client interactions

- **3 Worker Nodes** (including coordinator host)
  - **linux1** (158.39.75.13:8001) - Worker only
  - **linux2** (158.37.66.212:8001) - Worker only
  - **linux3** (158.39.75.48:8001) - Coordinator + Worker

## Resource Limits per Worker

- **Max RAM**: 22GB (enforced via systemd LimitAS)
- **Max CPU cores**: 12 (enforced via systemd CPUQuota=1200%)
- **Temporary Storage**: `/dev/shm/ocr_temp` (tmpfs for fast I/O)
- **OCR CPU Threads**: 10 threads per worker
- **MKLDNN**: Enabled for CPU optimization

## Features

### 1. PDF Processing Pipeline
- Accept PDF uploads via REST API
- Convert PDF pages to images at configurable DPI (default: 200)
- Distribute pages across available workers using round-robin
- Aggregate results into combined text and markdown output
- Comprehensive error handling with retry logic

### 2. Worker Functionality
- PaddleOCR with MKL-DNN optimization for CPU performance
- Health check endpoint with resource monitoring
- Base64 image input, structured JSON output
- Automatic resource limiting via systemd

### 3. Comprehensive Metrics
Each job returns detailed metrics including:
- `job_id`, `filename`, `total_pages`, `pages_processed`, `pages_failed`
- `total_words`, `total_chars`, `total_lines`, `avg_confidence`
- `start_time`, `end_time`, `total_duration_ms`, `avg_page_time_ms`
- `file_size_bytes`, `file_size_mb`
- `ocr_rounds_per_worker` - Pages processed per worker
- `worker_metrics` - Per-worker breakdown (pages, time, errors)
- `status` - success/partial_success/failed
- `errors` - List of error messages

### 4. Error Handling
- Retry failed pages up to 3 times with exponential backoff
- Graceful handling of worker failures
- Automatic failover to healthy workers
- Comprehensive error messages in response

## Prerequisites

### System Requirements
- **OS**: Ubuntu 24.04 LTS (or compatible)
- **Python**: 3.12+
- **RAM**: 22GB per worker node
- **CPU**: Multi-core processor (12 cores recommended)
- **Storage**: 10GB free space + tmpfs support
- **Network**: All nodes must be able to communicate

### Software Dependencies
- Python 3.12+
- poppler-utils (for PDF conversion)
- System packages: `python3.12`, `python3.12-venv`, `python3-pip`

## Quick Start

### Option 1: Automated Deployment (Recommended)

1. **Ensure SSH access** to all nodes:
   ```bash
   # Test SSH access
   ssh ubuntu@158.39.75.13 "echo 'Connection successful'"
   ssh ubuntu@158.37.66.212 "echo 'Connection successful'"
   ssh ubuntu@158.39.75.48 "echo 'Connection successful'"
   ```

2. **Run the deployment script**:
   ```bash
   cd distributed-ocr
   ./deploy.sh
   ```

   The script will:
   - Install all system dependencies
   - Set up Python virtual environments
   - Deploy worker and coordinator services
   - Create systemd services with resource limits
   - Start all services automatically

3. **Verify deployment**:
   ```bash
   # Check coordinator health
   curl http://158.39.75.48:8000/health
   
   # Check all workers status
   curl http://158.39.75.48:8000/workers/status
   ```

### Option 2: Docker Deployment

1. **Build and run with Docker Compose**:
   ```bash
   cd distributed-ocr
   docker-compose -f docker/docker-compose.yml up -d
   ```

2. **Check status**:
   ```bash
   docker-compose -f docker/docker-compose.yml ps
   curl http://localhost:8000/health
   curl http://localhost:8000/workers/status
   ```

### Option 3: Manual Installation

See the [Manual Installation](#manual-installation) section below.

## API Documentation

### Coordinator Endpoints

#### 1. Process PDF File
```bash
POST /ocr/pdf
```

**Parameters:**
- `file` (form-data, required): PDF file to process
- `language` (form-data, optional): OCR language (default: "en")
- `dpi` (form-data, optional): Image DPI for conversion (default: 200)

**Example:**
```bash
curl -X POST "http://158.39.75.48:8000/ocr/pdf" \
  -F "file=@document.pdf" \
  -F "language=en" \
  -F "dpi=200"
```

**Response:**
```json
{
  "metrics": {
    "job_id": "uuid",
    "filename": "document.pdf",
    "total_pages": 10,
    "pages_processed": 10,
    "pages_failed": 0,
    "total_words": 5000,
    "total_chars": 25000,
    "total_lines": 500,
    "avg_confidence": 0.9567,
    "start_time": "2026-01-01T10:00:00Z",
    "end_time": "2026-01-01T10:02:30Z",
    "total_duration_ms": 150000,
    "avg_page_time_ms": 15000,
    "file_size_bytes": 1048576,
    "file_size_mb": 1.0,
    "ocr_rounds_per_worker": {
      "linux1": 4,
      "linux2": 3,
      "linux3": 3
    },
    "worker_metrics": [...],
    "status": "success",
    "errors": []
  },
  "results": [...],
  "combined_text": "...",
  "combined_markdown": "..."
}
```

#### 2. Process Single Image
```bash
POST /ocr/image
```

**Parameters:**
- `file` (form-data, required): Image file to process
- `language` (form-data, optional): OCR language (default: "en")

**Example:**
```bash
curl -X POST "http://158.39.75.48:8000/ocr/image" \
  -F "file=@page.png" \
  -F "language=en"
```

#### 3. Check Workers Status
```bash
GET /workers/status
```

**Example:**
```bash
curl http://158.39.75.48:8000/workers/status
```

**Response:**
```json
{
  "total_workers": 3,
  "healthy_workers": 3,
  "workers": [
    {
      "name": "linux1",
      "host": "158.39.75.13",
      "port": 8001,
      "status": "healthy",
      "health": {
        "status": "healthy",
        "worker_name": "linux1",
        "cpu_percent": 15.5,
        "memory_percent": 25.3,
        "memory_mb": 5632.5,
        "uptime_seconds": 3600,
        "total_requests": 150
      }
    }
  ]
}
```

#### 4. Health Check
```bash
GET /health
```

**Example:**
```bash
curl http://158.39.75.48:8000/health
```

### Worker Endpoints

#### 1. Process Image (Internal Use)
```bash
POST /ocr
```

**Request Body:**
```json
{
  "image_base64": "base64_encoded_image_data",
  "language": "en",
  "page_number": 1
}
```

#### 2. Worker Health Check
```bash
GET /health
```

**Example:**
```bash
curl http://158.39.75.13:8001/health
```

## Configuration Options

### Environment Variables

#### Shared Configuration
- `MAX_RAM_GB`: Maximum RAM per worker (default: 22)
- `MAX_CPU_CORES`: Maximum CPU cores per worker (default: 12)
- `TEMP_DIR`: Temporary storage directory (default: /dev/shm/ocr_temp)

#### Worker Configuration
- `WORKER_HOST`: Worker bind address (default: 0.0.0.0)
- `WORKER_PORT`: Worker port (default: 8001)
- `WORKER_NAME`: Worker identifier
- `OCR_LANGUAGE`: Default OCR language (default: en)
- `OCR_CPU_THREADS`: CPU threads for OCR (default: 10)
- `ENABLE_MKLDNN`: Enable MKLDNN optimization (default: true)

#### Coordinator Configuration
- `COORDINATOR_HOST`: Coordinator bind address (default: 0.0.0.0)
- `COORDINATOR_PORT`: Coordinator port (default: 8000)
- `DEFAULT_DPI`: Default DPI for PDF conversion (default: 200)
- `MAX_RETRIES`: Maximum retry attempts (default: 3)
- `RETRY_DELAY_SECONDS`: Initial retry delay (default: 2)
- `WORKER_REQUEST_TIMEOUT`: Timeout for worker requests in seconds (default: 120)

### Modifying Worker Nodes

Edit `shared/config.py` to change worker configuration:

```python
WORKER_NODES: List[dict] = [
    {"host": "158.39.75.13", "port": 8001, "name": "linux1"},
    {"host": "158.37.66.212", "port": 8001, "name": "linux2"},
    {"host": "158.39.75.48", "port": 8001, "name": "linux3"},
]
```

## Multi-language OCR Support

PaddleOCR supports 100+ languages. Common language codes:

- `en` - English
- `ch` - Chinese (Simplified)
- `fr` - French
- `german` - German
- `korean` - Korean
- `japan` - Japanese
- `devanagari` - Hindi/Sanskrit

For Norwegian, Swedish, Danish, and other Latin-script languages, use `en` as PaddleOCR handles Latin characters effectively.

**Important Note**: PaddleOCR language is set at worker initialization and cannot be changed dynamically. To process documents in different languages:
1. Deploy separate worker pools with different language settings, OR
2. Restart workers with a different `OCR_LANGUAGE` environment variable

To set worker language, edit the systemd service file or set the environment variable:
```bash
Environment="OCR_LANGUAGE=ch"  # For Chinese
Environment="OCR_LANGUAGE=fr"  # For French
```

**Example (Chinese document):**
```bash
curl -X POST "http://158.39.75.48:8000/ocr/pdf" \
  -F "file=@chinese_doc.pdf" \
  -F "language=ch" \
  -F "dpi=300"
```

Note: The `language` parameter in the API request is for informational purposes. The actual OCR language is determined by the worker's `OCR_LANGUAGE` environment variable.

## Manual Installation

### On Each Worker Node

1. **Install system dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3.12 python3.12-venv python3-pip poppler-utils
   ```

2. **Create directories**:
   ```bash
   sudo mkdir -p /opt/distributed-ocr
   sudo mkdir -p /var/log/distributed-ocr
   sudo mkdir -p /dev/shm/ocr_temp
   sudo chown -R $USER:$USER /opt/distributed-ocr
   sudo chown -R $USER:$USER /var/log/distributed-ocr
   ```

3. **Copy application files**:
   ```bash
   cd /opt/distributed-ocr
   # Copy shared/ and worker/ directories here
   ```

4. **Create virtual environment**:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   pip install -r worker/requirements.txt
   ```

5. **Create systemd service** (`/etc/systemd/system/ocr-worker.service`):
   ```ini
   [Unit]
   Description=OCR Worker Service
   After=network.target

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/opt/distributed-ocr/worker
   Environment="WORKER_NAME=linux1"
   Environment="WORKER_HOST=0.0.0.0"
   Environment="WORKER_PORT=8001"
   ExecStart=/opt/distributed-ocr/venv/bin/python worker_server.py
   Restart=always
   RestartSec=10
   LimitAS=23622320128
   CPUQuota=1200%
   StandardOutput=append:/var/log/distributed-ocr/worker.log
   StandardError=append:/var/log/distributed-ocr/worker.error.log

   [Install]
   WantedBy=multi-user.target
   ```

6. **Start service**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable ocr-worker
   sudo systemctl start ocr-worker
   sudo systemctl status ocr-worker
   ```

### On Coordinator Node

1. **Follow steps 1-4 from worker installation**

2. **Install coordinator dependencies**:
   ```bash
   source venv/bin/activate
   pip install -r coordinator/requirements.txt
   ```

3. **Create systemd service** (`/etc/systemd/system/ocr-coordinator.service`):
   ```ini
   [Unit]
   Description=OCR Coordinator Service
   After=network.target

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/opt/distributed-ocr/coordinator
   Environment="COORDINATOR_HOST=0.0.0.0"
   Environment="COORDINATOR_PORT=8000"
   ExecStart=/opt/distributed-ocr/venv/bin/python main_server.py
   Restart=always
   RestartSec=10
   LimitAS=23622320128
   CPUQuota=1200%
   StandardOutput=append:/var/log/distributed-ocr/coordinator.log
   StandardError=append:/var/log/distributed-ocr/coordinator.error.log

   [Install]
   WantedBy=multi-user.target
   ```

4. **Start service**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable ocr-coordinator
   sudo systemctl start ocr-coordinator
   sudo systemctl status ocr-coordinator
   ```

## Performance Expectations

### Typical Performance Metrics

- **Single Page Processing**: 10-20 seconds per page (CPU-only)
- **100-page PDF**: 15-30 minutes (distributed across 3 workers)
- **Memory Usage**: 4-8GB per worker during processing
- **CPU Utilization**: 80-100% on allocated cores
- **Throughput**: ~15-20 pages/minute (with 3 workers)

### Performance Optimization Tips

1. **Adjust DPI**: Lower DPI (150) for faster processing, higher DPI (300) for better accuracy
2. **CPU Threads**: Increase `OCR_CPU_THREADS` if more cores available
3. **Add Workers**: Scale horizontally by adding more worker nodes
4. **Optimize Images**: Pre-process images to reduce size/complexity
5. **Use tmpfs**: Ensure `/dev/shm` is used for temporary storage

## Troubleshooting

### Common Issues

#### 1. Worker Not Responding
```bash
# Check worker status
ssh ubuntu@worker-host "sudo systemctl status ocr-worker"

# Check worker logs
ssh ubuntu@worker-host "sudo tail -f /var/log/distributed-ocr/worker.log"

# Restart worker
ssh ubuntu@worker-host "sudo systemctl restart ocr-worker"
```

#### 2. Memory Issues
```bash
# Check memory usage
ssh ubuntu@worker-host "free -h"

# Check worker memory
ssh ubuntu@worker-host "ps aux | grep worker_server"

# Increase memory limit in systemd service
# Edit: LimitAS=33622320128  # 32GB
```

#### 3. PDF Conversion Fails
```bash
# Verify poppler-utils is installed
which pdftoppm

# Test PDF conversion manually
pdf2image document.pdf
```

#### 4. PaddleOCR Initialization Fails
```bash
# Check PaddleOCR installation
source /opt/distributed-ocr/venv/bin/activate
python -c "from paddleocr import PaddleOCR; print('OK')"

# Reinstall if necessary
pip install --force-reinstall paddleocr paddlepaddle
```

#### 5. Network Connectivity Issues
```bash
# Test connectivity from coordinator to workers
curl http://158.39.75.13:8001/health
curl http://158.37.66.212:8001/health
curl http://158.39.75.48:8001/health

# Check firewall rules
sudo ufw status
sudo ufw allow 8000/tcp  # Coordinator
sudo ufw allow 8001/tcp  # Workers
```

### Log Files

- **Coordinator logs**: `/var/log/distributed-ocr/coordinator.log`
- **Coordinator errors**: `/var/log/distributed-ocr/coordinator.error.log`
- **Worker logs**: `/var/log/distributed-ocr/worker.log`
- **Worker errors**: `/var/log/distributed-ocr/worker.error.log`

### Health Monitoring

```bash
# Create monitoring script
cat > monitor.sh << 'EOF'
#!/bin/bash
echo "=== OCR System Health Check ==="
echo ""
echo "Coordinator:"
curl -s http://158.39.75.48:8000/health | jq .
echo ""
echo "Workers:"
curl -s http://158.39.75.48:8000/workers/status | jq .
EOF

chmod +x monitor.sh
./monitor.sh
```

## Examples

See `curl_examples.sh` for comprehensive usage examples.

### Basic PDF Processing
```bash
curl -X POST "http://158.39.75.48:8000/ocr/pdf" \
  -F "file=@document.pdf" \
  -F "language=en" \
  -F "dpi=200"
```

### Extract Only Text
```bash
curl -X POST "http://158.39.75.48:8000/ocr/pdf" \
  -F "file=@document.pdf" \
  -F "language=en" | jq -r '.combined_text'
```

### Get Metrics Only
```bash
curl -X POST "http://158.39.75.48:8000/ocr/pdf" \
  -F "file=@document.pdf" \
  -F "language=en" | jq '.metrics'
```

### Process High-Quality Scan
```bash
curl -X POST "http://158.39.75.48:8000/ocr/pdf" \
  -F "file=@scan.pdf" \
  -F "language=en" \
  -F "dpi=300"
```

## Security Considerations

1. **Network Security**: Use firewall rules to restrict access
2. **Authentication**: Add API authentication for production use
3. **File Validation**: System validates PDF/image files before processing
4. **Resource Limits**: Systemd enforces memory and CPU limits
5. **Temporary Files**: Automatically cleaned up after processing

## License

This distributed OCR system is provided as-is for use with the HunyuanOCR project.

## Support

For issues, questions, or contributions:
- Create an issue in the GitHub repository
- Check the troubleshooting guide above
- Review log files for detailed error messages

## Acknowledgments

- **PaddleOCR**: Optical character recognition engine
- **FastAPI**: Web framework for building APIs
- **pdf2image**: PDF to image conversion library
- **Poppler**: PDF rendering library
