from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import tempfile
import os
from app.core.ocr_service import ocr_service
from app.dto.schemas import OCRResponse, PDFOCRResponse, ErrorResponse

router = APIRouter(prefix="/ocr", tags=["OCR"])

# Supported file types
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/gif", "image/bmp", "image/tiff"}
SUPPORTED_PDF_TYPE = "application/pdf"


@router.post("/image", response_model=OCRResponse, summary="Extract raw text from legal document image")
async def extract_text_from_image(file: UploadFile = File(...)):
    """
    Extract all raw text content from a Saudi Arabian legal document image using Google Gemini AI.
    
    This endpoint processes images of legal documents and extracts all visible text exactly as it appears, including:
    - Headers and titles
    - Article numbers and legal references  
    - Main content and legal text
    - Tables (with preserved formatting)
    - Lists and numbered items
    - Dates, numbers, and other information
    - Names, places, and legal terms
    
    Returns the complete raw text extraction preserving original formatting.
    """
    try:
        # Validate file type
        if file.content_type not in SUPPORTED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Supported types: {', '.join(SUPPORTED_IMAGE_TYPES)}"
            )
        
        # Read file content
        file_content = await file.read()
        
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Process image with OCR service
        result = ocr_service.extract_text_from_image_bytes(file_content)
        
        return OCRResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@router.post("/pdf", response_model=PDFOCRResponse, summary="Extract raw text from legal document PDF")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    """
    Extract all raw text content from a Saudi Arabian legal document PDF using Google Gemini AI.
    
    This endpoint processes PDF files page by page, converting each page to an image
    and then extracting all visible text exactly as it appears, including:
    - Headers and titles
    - Article numbers and legal references
    - Main content and legal text
    - Tables (with preserved formatting)
    - Lists and numbered items
    - Dates, numbers, and other information
    - Names, places, and legal terms
    
    Returns raw text extraction for each page preserving original formatting.
    """
    try:
        # Validate file type
        if file.content_type != SUPPORTED_PDF_TYPE:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Expected: {SUPPORTED_PDF_TYPE}"
            )
        
        # Read file content
        file_content = await file.read()
        
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Process PDF with OCR service
        pages_results = ocr_service.extract_text_from_pdf_bytes(file_content)
        
        # Convert results to OCRResponse objects and calculate statistics
        ocr_pages = []
        success_count = 0
        
        for page_result in pages_results:
            ocr_page = OCRResponse(**page_result)
            ocr_pages.append(ocr_page)
            
            # Count successful pages (pages without errors)
            if not ocr_page.error:
                success_count += 1
        
        return PDFOCRResponse(
            pages=ocr_pages,
            total_pages=len(ocr_pages),
            success_pages=success_count,
            failed_pages=len(ocr_pages) - success_count
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@router.get("/health", summary="OCR service health check")
async def ocr_health_check():
    """
    Check if the OCR service is properly configured and ready to use.
    """
    try:
        # Check if Google API key is configured
        from app.config.settings import settings
        
        if not settings.GOOGLE_API_KEY:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "Google API key not configured",
                    "details": "Please set GOOGLE_API_KEY in environment variables"
                }
            )
        
        return {
            "status": "healthy",
            "message": "OCR service is ready",
            "supported_image_types": list(SUPPORTED_IMAGE_TYPES),
            "supported_pdf_type": SUPPORTED_PDF_TYPE
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"OCR service error: {str(e)}"
            }
        )
