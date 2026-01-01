#!/bin/bash

# Distributed OCR System - Automated Deployment Script
# This script deploys the coordinator and workers to multiple nodes via SSH

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
COORDINATOR_NODE="158.39.75.48"
WORKER_NODES=("158.39.75.13" "158.37.66.212" "158.39.75.48")
WORKER_NAMES=("linux1" "linux2" "linux3")
SSH_USER="${SSH_USER:-ubuntu}"
DEPLOY_DIR="/opt/distributed-ocr"
VENV_DIR="${DEPLOY_DIR}/venv"
LOG_DIR="/var/log/distributed-ocr"

echo -e "${GREEN}=== Distributed OCR System Deployment ===${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Function to deploy worker
deploy_worker() {
    local host=$1
    local worker_name=$2
    
    print_status "Deploying worker to ${worker_name} (${host})..."
    
    # Create deployment script
    cat > /tmp/deploy_worker_${worker_name}.sh << 'DEPLOY_SCRIPT'
#!/bin/bash
set -e

DEPLOY_DIR="/opt/distributed-ocr"
VENV_DIR="${DEPLOY_DIR}/venv"
LOG_DIR="/var/log/distributed-ocr"
WORKER_NAME="WORKER_NAME_PLACEHOLDER"

echo "Setting up worker: ${WORKER_NAME}..."

# Create directories
sudo mkdir -p ${DEPLOY_DIR}
sudo mkdir -p ${LOG_DIR}
sudo mkdir -p /dev/shm/ocr_temp
sudo chown -R $USER:$USER ${DEPLOY_DIR}
sudo chown -R $USER:$USER ${LOG_DIR}

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3-pip poppler-utils

# Create virtual environment
echo "Creating virtual environment..."
python3.12 -m venv ${VENV_DIR}
source ${VENV_DIR}/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install paddleocr paddlepaddle fastapi uvicorn psutil pillow numpy pydantic

echo "Worker setup completed for ${WORKER_NAME}"
DEPLOY_SCRIPT
    
    # Replace placeholder with actual worker name
    sed -i "s/WORKER_NAME_PLACEHOLDER/${worker_name}/g" /tmp/deploy_worker_${worker_name}.sh
    
    # Copy and execute deployment script
    scp /tmp/deploy_worker_${worker_name}.sh ${SSH_USER}@${host}:/tmp/
    ssh ${SSH_USER}@${host} "bash /tmp/deploy_worker_${worker_name}.sh"
    
    # Copy application files
    print_status "Copying application files to ${worker_name}..."
    ssh ${SSH_USER}@${host} "mkdir -p ${DEPLOY_DIR}/shared ${DEPLOY_DIR}/worker"
    
    scp shared/*.py ${SSH_USER}@${host}:${DEPLOY_DIR}/shared/
    scp worker/*.py ${SSH_USER}@${host}:${DEPLOY_DIR}/worker/
    scp worker/requirements.txt ${SSH_USER}@${host}:${DEPLOY_DIR}/worker/
    
    # Create systemd service
    print_status "Creating systemd service for ${worker_name}..."
    
    cat > /tmp/ocr-worker.service << SERVICE_FILE
[Unit]
Description=OCR Worker Service - ${worker_name}
After=network.target

[Service]
Type=simple
User=${SSH_USER}
WorkingDirectory=${DEPLOY_DIR}/worker
Environment="WORKER_NAME=${worker_name}"
Environment="WORKER_HOST=0.0.0.0"
Environment="WORKER_PORT=8001"
Environment="TEMP_DIR=/dev/shm/ocr_temp"
ExecStart=${VENV_DIR}/bin/python worker_server.py
Restart=always
RestartSec=10

# Resource limits
LimitAS=23622320128
CPUQuota=1200%

# Logging
StandardOutput=append:${LOG_DIR}/worker.log
StandardError=append:${LOG_DIR}/worker.error.log

[Install]
WantedBy=multi-user.target
SERVICE_FILE
    
    scp /tmp/ocr-worker.service ${SSH_USER}@${host}:/tmp/
    ssh ${SSH_USER}@${host} "sudo mv /tmp/ocr-worker.service /etc/systemd/system/ && \
                             sudo systemctl daemon-reload && \
                             sudo systemctl enable ocr-worker && \
                             sudo systemctl restart ocr-worker"
    
    print_status "Worker ${worker_name} deployed successfully"
    
    # Clean up temp files
    rm /tmp/deploy_worker_${worker_name}.sh
    rm /tmp/ocr-worker.service
}

# Function to deploy coordinator
deploy_coordinator() {
    local host=$COORDINATOR_NODE
    
    print_status "Deploying coordinator to ${host}..."
    
    # Create deployment script
    cat > /tmp/deploy_coordinator.sh << 'DEPLOY_SCRIPT'
#!/bin/bash
set -e

DEPLOY_DIR="/opt/distributed-ocr"
VENV_DIR="${DEPLOY_DIR}/venv"
LOG_DIR="/var/log/distributed-ocr"

echo "Setting up coordinator..."

# Create directories
sudo mkdir -p ${DEPLOY_DIR}
sudo mkdir -p ${LOG_DIR}
sudo mkdir -p /dev/shm/ocr_temp
sudo chown -R $USER:$USER ${DEPLOY_DIR}
sudo chown -R $USER:$USER ${LOG_DIR}

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3-pip poppler-utils

# Create virtual environment (reuse if exists from worker deployment)
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv ${VENV_DIR}
fi
source ${VENV_DIR}/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install fastapi uvicorn httpx pdf2image pillow pydantic psutil

echo "Coordinator setup completed"
DEPLOY_SCRIPT
    
    # Copy and execute deployment script
    scp /tmp/deploy_coordinator.sh ${SSH_USER}@${host}:/tmp/
    ssh ${SSH_USER}@${host} "bash /tmp/deploy_coordinator.sh"
    
    # Copy application files
    print_status "Copying application files to coordinator..."
    ssh ${SSH_USER}@${host} "mkdir -p ${DEPLOY_DIR}/shared ${DEPLOY_DIR}/coordinator"
    
    scp shared/*.py ${SSH_USER}@${host}:${DEPLOY_DIR}/shared/
    scp coordinator/*.py ${SSH_USER}@${host}:${DEPLOY_DIR}/coordinator/
    scp coordinator/requirements.txt ${SSH_USER}@${host}:${DEPLOY_DIR}/coordinator/
    
    # Create systemd service
    print_status "Creating systemd service for coordinator..."
    
    cat > /tmp/ocr-coordinator.service << SERVICE_FILE
[Unit]
Description=OCR Coordinator Service
After=network.target

[Service]
Type=simple
User=${SSH_USER}
WorkingDirectory=${DEPLOY_DIR}/coordinator
Environment="COORDINATOR_HOST=0.0.0.0"
Environment="COORDINATOR_PORT=8000"
Environment="TEMP_DIR=/dev/shm/ocr_temp"
ExecStart=${VENV_DIR}/bin/python main_server.py
Restart=always
RestartSec=10

# Resource limits
LimitAS=23622320128
CPUQuota=1200%

# Logging
StandardOutput=append:${LOG_DIR}/coordinator.log
StandardError=append:${LOG_DIR}/coordinator.error.log

[Install]
WantedBy=multi-user.target
SERVICE_FILE
    
    scp /tmp/ocr-coordinator.service ${SSH_USER}@${host}:/tmp/
    ssh ${SSH_USER}@${host} "sudo mv /tmp/ocr-coordinator.service /etc/systemd/system/ && \
                             sudo systemctl daemon-reload && \
                             sudo systemctl enable ocr-coordinator && \
                             sudo systemctl restart ocr-coordinator"
    
    print_status "Coordinator deployed successfully"
    
    # Clean up temp files
    rm /tmp/deploy_coordinator.sh
    rm /tmp/ocr-coordinator.service
}

# Function to check service status
check_status() {
    local host=$1
    local service=$2
    local name=$3
    
    print_status "Checking status of ${name}..."
    ssh ${SSH_USER}@${host} "sudo systemctl status ${service} --no-pager | head -20" || true
}

# Main deployment flow
main() {
    echo "Deployment Configuration:"
    echo "  Coordinator: ${COORDINATOR_NODE}"
    echo "  Workers: ${WORKER_NODES[@]}"
    echo "  SSH User: ${SSH_USER}"
    echo ""
    
    read -p "Continue with deployment? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled"
        exit 1
    fi
    
    # Deploy workers
    for i in "${!WORKER_NODES[@]}"; do
        deploy_worker "${WORKER_NODES[$i]}" "${WORKER_NAMES[$i]}"
        echo ""
    done
    
    # Deploy coordinator
    deploy_coordinator
    echo ""
    
    # Wait for services to start
    print_status "Waiting for services to start..."
    sleep 10
    
    # Check status
    echo ""
    print_status "Checking service status..."
    echo ""
    
    for i in "${!WORKER_NODES[@]}"; do
        check_status "${WORKER_NODES[$i]}" "ocr-worker" "${WORKER_NAMES[$i]} worker"
        echo ""
    done
    
    check_status "${COORDINATOR_NODE}" "ocr-coordinator" "Coordinator"
    echo ""
    
    print_status "${GREEN}Deployment completed!${NC}"
    echo ""
    echo "Coordinator API: http://${COORDINATOR_NODE}:8000"
    echo "Health check: curl http://${COORDINATOR_NODE}:8000/health"
    echo "Workers status: curl http://${COORDINATOR_NODE}:8000/workers/status"
    echo ""
    echo "See curl_examples.sh for usage examples"
}

# Run main function
main
