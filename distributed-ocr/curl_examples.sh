#!/bin/bash

# Example curl commands for distributed OCR system

COORDINATOR_URL="http://158.39.75.48:8000"

echo "=== Distributed OCR System - Curl Examples ==="
echo ""

# Health check
echo "1. Check coordinator health:"
echo "curl ${COORDINATOR_URL}/health"
echo ""

# Check workers status
echo "2. Check all workers status:"
echo "curl ${COORDINATOR_URL}/workers/status"
echo ""

# Process PDF file
echo "3. Process a PDF file (English):"
echo "curl -X POST \"${COORDINATOR_URL}/ocr/pdf\" \\"
echo "  -F \"file=@document.pdf\" \\"
echo "  -F \"language=en\" \\"
echo "  -F \"dpi=200\""
echo ""

# Process PDF with higher DPI
echo "4. Process a PDF file with higher DPI for better quality:"
echo "curl -X POST \"${COORDINATOR_URL}/ocr/pdf\" \\"
echo "  -F \"file=@document.pdf\" \\"
echo "  -F \"language=en\" \\"
echo "  -F \"dpi=300\""
echo ""

# Process PDF with Chinese language
echo "5. Process a Chinese PDF file:"
echo "curl -X POST \"${COORDINATOR_URL}/ocr/pdf\" \\"
echo "  -F \"file=@chinese_document.pdf\" \\"
echo "  -F \"language=ch\" \\"
echo "  -F \"dpi=200\""
echo ""

# Process single image
echo "6. Process a single image:"
echo "curl -X POST \"${COORDINATOR_URL}/ocr/image\" \\"
echo "  -F \"file=@page.png\" \\"
echo "  -F \"language=en\""
echo ""

# Process Norwegian document
echo "7. Process a Norwegian document:"
echo "curl -X POST \"${COORDINATOR_URL}/ocr/pdf\" \\"
echo "  -F \"file=@norwegian_doc.pdf\" \\"
echo "  -F \"language=en\" \\"
echo "  -F \"dpi=200\""
echo ""

# Save output to file
echo "8. Process PDF and save output to file:"
echo "curl -X POST \"${COORDINATOR_URL}/ocr/pdf\" \\"
echo "  -F \"file=@document.pdf\" \\"
echo "  -F \"language=en\" \\"
echo "  -F \"dpi=200\" \\"
echo "  -o output.json"
echo ""

# Pretty print JSON output
echo "9. Process PDF and pretty print output:"
echo "curl -X POST \"${COORDINATOR_URL}/ocr/pdf\" \\"
echo "  -F \"file=@document.pdf\" \\"
echo "  -F \"language=en\" \\"
echo "  -F \"dpi=200\" | jq ."
echo ""

# Extract only combined text
echo "10. Extract only the combined text from PDF:"
echo "curl -X POST \"${COORDINATOR_URL}/ocr/pdf\" \\"
echo "  -F \"file=@document.pdf\" \\"
echo "  -F \"language=en\" \\"
echo "  -F \"dpi=200\" | jq -r '.combined_text'"
echo ""

# Extract metrics only
echo "11. Extract only the metrics:"
echo "curl -X POST \"${COORDINATOR_URL}/ocr/pdf\" \\"
echo "  -F \"file=@document.pdf\" \\"
echo "  -F \"language=en\" \\"
echo "  -F \"dpi=200\" | jq '.metrics'"
echo ""

echo "=== Multi-language Support ==="
echo ""
echo "Supported languages:"
echo "  en  - English"
echo "  ch  - Chinese"
echo "  fr  - French"
echo "  german - German"
echo "  korean - Korean"
echo "  japan - Japanese"
echo ""
echo "Note: For Norwegian and other languages, use 'en' as PaddleOCR will"
echo "      still detect Latin characters effectively."
echo ""
