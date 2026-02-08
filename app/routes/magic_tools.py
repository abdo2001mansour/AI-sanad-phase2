from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import json
import asyncio
from app.dto.schemas import (
    TranslationRequest,
    SummarizationRequest,
    AnalysisRequest,
    RephrasingRequest,
    FileExtractionRequest,
    LanguageDetectionRequest,
    LanguageDetectionResponse,
    MagicToolsTypesResponse,
    MagicToolsErrorResponse,
    MagicToolsSuccessResponse,
    FileResponse
)
from app.core.magic_tools import magic_tools_service
from app.core.magic_tools.translation_service import translation_service
from app.core.magic_tools.language_detection_service import language_detection_service
from app.core.magic_tools.summarization_service import summarization_service
from app.core.magic_tools.analysis_service import analysis_service
from app.core.magic_tools.rephrasing_service import rephrasing_service

router = APIRouter(prefix="/magic-tools", tags=["Magic Tools"])


@router.post("/translation", summary="Translate text to target language")
async def translate_text(request: TranslationRequest, raw_request: Request):
    """
    Translate text to a target language with specified translation type.
    
    Supports three translation types:
    - general: General purpose translation
    - legal: Legal translation (specialized for legal documents)
    - academic: Academic translation (specialized for academic content)
    
    Returns a streamed response (SSE) with the translated text.
    """
    try:
        # Validate input
        if not request.text or not request.text.strip():
            error_response = MagicToolsErrorResponse(
                success=False,
                error="Text cannot be empty",
                error_code="EMPTY_TEXT"
            )
            return JSONResponse(
                status_code=400,
                content=error_response.dict()
            )
        
        if not request.lang or not request.lang.strip():
            error_response = MagicToolsErrorResponse(
                success=False,
                error="Target language cannot be empty",
                error_code="EMPTY_LANGUAGE"
            )
            return JSONResponse(
                status_code=400,
                content=error_response.dict()
            )
        
        # Real translation implementation using Gemini Flash
        async def translation_generator():
            # If file is provided, stream file URL information (TODO: implement file translation)
            if request.file and request.file.list_of_files:
                import uuid
                file_urls = []
                for _ in request.file.list_of_files:
                    file_id = str(uuid.uuid4())
                    file_url = f"https://example.com/files/translation_{file_id}.doc"
                    file_urls.append(file_url)
                
                example_url = "https://example.com/files/translation_12345678-1234-1234-1234-123456789abc.doc"
                
                # Stream file URL information
                file_data = {
                    "success": True,
                    "type": "translation",
                    "file": {
                        "list_of_files": file_urls
                    },
                    "example": example_url,
                    "language": request.lang,
                    "translation_type": request.type
                }
                yield f"data: {json.dumps(file_data, ensure_ascii=False)}\n\n"
                
                # Send completion marker
                final_data = {
                    "success": True,
                    "type": "translation",
                    "completed": True
                }
                yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Use real translation service with Gemini Flash
            async for chunk in translation_service.translate_text_stream(
                text=request.text,
                target_language=request.lang,
                translation_type=request.type
            ):
                if await raw_request.is_disconnected():
                    return
                yield chunk
        
        return StreamingResponse(
            translation_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        error_response = MagicToolsErrorResponse(
            success=False,
            error=str(e),
            error_code="TRANSLATION_ERROR",
            details="An error occurred during translation"
        )
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )


@router.post("/summarize", summary="Summarize text")
async def summarize_text(request: SummarizationRequest, raw_request: Request):
    """
    Summarize text with specified type.
    
    Supports two summarization types:
    - brief: Brief summary (concise overview)
    - detailed: Detailed summary (comprehensive overview)
    
    Returns a streamed response (SSE) with the summarized text.
    """
    try:
        # Validate input
        if not request.text or not request.text.strip():
            error_response = MagicToolsErrorResponse(
                success=False,
                error="Text cannot be empty",
                error_code="EMPTY_TEXT"
            )
            return JSONResponse(
                status_code=400,
                content=error_response.dict()
            )
        
        # Real summarization implementation using Gemini 2.5 Flash
        async def summarization_generator():
            # If file is provided, stream file URL information (TODO: implement file summarization)
            if request.file and request.file.list_of_files:
                import uuid
                file_urls = []
                for _ in request.file.list_of_files:
                    file_id = str(uuid.uuid4())
                    file_url = f"https://example.com/files/summary_{file_id}.doc"
                    file_urls.append(file_url)
                
                example_url = "https://example.com/files/summary_12345678-1234-1234-1234-123456789abc.doc"
                
                # Stream file URL information
                file_data = {
                    "success": True,
                    "type": "summarization",
                    "file": {
                        "list_of_files": file_urls
                    },
                    "example": example_url,
                    "summarization_type": request.type
                }
                yield f"data: {json.dumps(file_data, ensure_ascii=False)}\n\n"
                
                # Send completion marker
                final_data = {
                    "success": True,
                    "type": "summarization",
                    "completed": True
                }
                yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Use real summarization service with Gemini 2.5 Flash
            async for chunk in summarization_service.summarize_text_stream(
                text=request.text,
                summarization_type=request.type
            ):
                if await raw_request.is_disconnected():
                    return
                yield chunk
        
        return StreamingResponse(
            summarization_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        error_response = MagicToolsErrorResponse(
            success=False,
            error=str(e),
            error_code="SUMMARIZATION_ERROR",
            details="An error occurred during summarization"
        )
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )


@router.post("/analyze", summary="Analyze text")
async def analyze_text(request: AnalysisRequest, raw_request: Request):
    """
    Analyze text with specified type.
    
    Supports two analysis types:
    - brief: Brief analysis (quick insights)
    - detailed: Detailed analysis (comprehensive insights)
    
    Returns a streamed response (SSE) with the analysis results.
    """
    try:
        # Validate input
        if not request.text or not request.text.strip():
            error_response = MagicToolsErrorResponse(
                success=False,
                error="Text cannot be empty",
                error_code="EMPTY_TEXT"
            )
            return JSONResponse(
                status_code=400,
                content=error_response.dict()
            )
        
        # Real legal analysis implementation using Gemini 2.5 Flash
        async def analysis_generator():
            # If file is provided, stream file URL information (TODO: implement file analysis)
            if request.file and request.file.list_of_files:
                import uuid
                file_urls = []
                for _ in request.file.list_of_files:
                    file_id = str(uuid.uuid4())
                    file_url = f"https://example.com/files/analysis_{file_id}.doc"
                    file_urls.append(file_url)
                
                example_url = "https://example.com/files/analysis_12345678-1234-1234-1234-123456789abc.doc"
                
                # Stream file URL information
                file_data = {
                    "success": True,
                    "type": "analysis",
                    "file": {
                        "list_of_files": file_urls
                    },
                    "example": example_url,
                    "analysis_type": request.type
                }
                yield f"data: {json.dumps(file_data, ensure_ascii=False)}\n\n"
                
                # Send completion marker
                final_data = {
                    "success": True,
                    "type": "analysis",
                    "completed": True
                }
                yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Use real analysis service with Gemini 2.5 Flash
            async for chunk in analysis_service.analyze_text_stream(
                text=request.text,
                analysis_type=request.type
            ):
                if await raw_request.is_disconnected():
                    return
                yield chunk
        
        return StreamingResponse(
            analysis_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        error_response = MagicToolsErrorResponse(
            success=False,
            error=str(e),
            error_code="ANALYSIS_ERROR",
            details="An error occurred during analysis"
        )
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )


@router.post("/rephrase", summary="Rephrase text")
async def rephrase_text(request: RephrasingRequest, raw_request: Request):
    """
    Rephrase text to improve clarity and style while maintaining the original meaning.
    
    Returns a streamed response (SSE) with the rephrased text.
    """
    try:
        # Validate input
        if not request.text or not request.text.strip():
            error_response = MagicToolsErrorResponse(
                success=False,
                error="Text cannot be empty",
                error_code="EMPTY_TEXT"
            )
            return JSONResponse(
                status_code=400,
                content=error_response.dict()
            )
        
        # Real legal rephrasing implementation using Gemini 2.5 Flash
        async def rephrasing_generator():
            # If file is provided, stream file URL information (TODO: implement file rephrasing)
            if request.file and request.file.list_of_files:
                import uuid
                file_urls = []
                for _ in request.file.list_of_files:
                    file_id = str(uuid.uuid4())
                    file_url = f"https://example.com/files/rephrased_{file_id}.doc"
                    file_urls.append(file_url)
                
                example_url = "https://example.com/files/rephrased_12345678-1234-1234-1234-123456789abc.doc"
                
                # Stream file URL information
                file_data = {
                    "success": True,
                    "type": "rephrasing",
                    "file": {
                        "list_of_files": file_urls
                    },
                    "example": example_url
                }
                yield f"data: {json.dumps(file_data, ensure_ascii=False)}\n\n"
                
                # Send completion marker
                final_data = {
                    "success": True,
                    "type": "rephrasing",
                    "completed": True
                }
                yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Use real rephrasing service with Gemini 2.5 Flash
            async for chunk in rephrasing_service.rephrase_text_stream(
                text=request.text
            ):
                if await raw_request.is_disconnected():
                    return
                yield chunk
        
        return StreamingResponse(
            rephrasing_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        error_response = MagicToolsErrorResponse(
            success=False,
            error=str(e),
            error_code="REPHRASING_ERROR",
            details="An error occurred during rephrasing"
        )
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )


@router.get("/types", response_model=MagicToolsTypesResponse, summary="Get available types for all magic tools")
async def get_magic_tools_types():
    """
    Get all available types and options for magic tools endpoints.
    
    This endpoint returns:
    - Available translation types
    - Available summarization types
    - Available analysis types
    - Supported file types for content extraction
    - Supported languages for translation (48 languages)
    
    This is the only endpoint with real implementation (not dummy).
    """
    # Get supported languages from translation service
    languages_list = [
        {
            "code": lang_info["code"],
            "name_en": lang_info["name_en"],
            "name_ar": lang_info["name_ar"]
        }
        for lang_info in translation_service.languages.values()
    ]
    
    # Sort by English name
    languages_list.sort(key=lambda x: x["name_en"])
    
    return MagicToolsTypesResponse(
        translation_types=["general", "legal", "academic"],
        summarization_types=["brief", "detailed"],
        analysis_types=["brief", "detailed"],
        supported_file_types=[
            ".pdf",
            ".txt",
            ".docx",
            ".md",
            ".pptx",
            ".tex",
            ".ts",
            ".csv",
            ".xlsx",
            ".xls"
        ],
        supported_languages=languages_list
    )


@router.get("/languages", summary="Get list of supported languages for translation")
async def get_supported_languages():
    """
    Get list of all supported languages for translation.
    
    Returns a list of language objects with:
    - code: ISO 639-3 three-letter language code
    - name_en: Language name in English
    - name_ar: Language name in Arabic
    
    Use the language code when making translation requests.
    
    Note: Languages are also available in the GET /api/v1/magic-tools/types endpoint.
    """
    try:
        languages_list = [
            {
                "code": lang_info["code"],
                "name_en": lang_info["name_en"],
                "name_ar": lang_info["name_ar"]
            }
            for lang_info in translation_service.languages.values()
        ]
        
        # Sort by English name
        languages_list.sort(key=lambda x: x["name_en"])
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "languages": languages_list,
                "total": len(languages_list)
            }
        )
    except Exception as e:
        error_response = MagicToolsErrorResponse(
            success=False,
            error=str(e),
            error_code="LANGUAGES_ERROR",
            details="An error occurred while retrieving supported languages"
        )
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )


@router.post("/detect-language", response_model=LanguageDetectionResponse, summary="Detect language of text")
async def detect_language(request: LanguageDetectionRequest):
    """
    Detect the language of the given text using AI.
    
    Uses Google Gemini 2.5 Flash to analyze the text and return the ISO 639-3 
    language code from our supported languages list.
    
    Returns:
    - detected_language: ISO 639-3 code (3 letters) or "none" if language not in supported list
    - language_name_en: Language name in English (null if none)
    - language_name_ar: Language name in Arabic (null if none)
    - confidence: Confidence level (always "high" for Gemini)
    
    Supported languages: 48 languages (see GET /api/v1/magic-tools/languages)
    """
    try:
        # Validate input
        if not request.text or not request.text.strip():
            error_response = MagicToolsErrorResponse(
                success=False,
                error="Text cannot be empty",
                error_code="EMPTY_TEXT"
            )
            return JSONResponse(
                status_code=400,
                content=error_response.dict()
            )
        
        # Detect language using the service
        result = await language_detection_service.detect_language(request.text)
        
        return LanguageDetectionResponse(**result)
    
    except Exception as e:
        error_response = MagicToolsErrorResponse(
            success=False,
            error=str(e),
            error_code="DETECTION_ERROR",
            details="An error occurred during language detection"
        )
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )


@router.post("/extract-content", summary="Extract content from file URL")
async def extract_file_content(request: FileExtractionRequest):
    """
    Extract text content from a file URL.
    
    Supports various file types including PDF, Word documents, images, and text files.
    Use GET /api/v1/magic-tools/types to see all supported file types.
    
    The file will be downloaded, processed, and saved to the sanad-data-source S3 bucket.
    
    Returns a JSON response with the extracted text content as a string.
    """
    try:
        # Validate input
        if not request.file_url or not request.file_url.strip():
            error_response = MagicToolsErrorResponse(
                success=False,
                error="File URL cannot be empty",
                error_code="EMPTY_URL"
            )
            return JSONResponse(
                status_code=400,
                content=error_response.dict()
            )
        
        # Validate URL format (basic check)
        if not (request.file_url.startswith("http://") or request.file_url.startswith("https://")):
            error_response = MagicToolsErrorResponse(
                success=False,
                error="Invalid URL format. URL must start with http:// or https://",
                error_code="INVALID_URL_FORMAT"
            )
            return JSONResponse(
                status_code=400,
                content=error_response.dict()
            )
        
        # Use magic tools service to extract content
        file_url = request.file_url.strip()
        result = await magic_tools_service.extract_content_from_url(file_url)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "type": "file_extraction",
                "content": result["content"],
                "file_url": file_url,
                "filename": result["filename"],
                "content_type": result["content_type"],
                "s3_bucket": result["s3_bucket"],
                "s3_key": result["s3_key"],
                "s3_url": result["s3_url"],
                "content_length": result["content_length"]
            }
        )
    
    except Exception as e:
        # Handle specific error types
        error_message = str(e)
        error_code = "EXTRACTION_ERROR"
        
        if "Failed to download" in error_message:
            error_code = "DOWNLOAD_ERROR"
        elif "Unsupported file type" in error_message:
            error_code = "UNSUPPORTED_FILE_TYPE"
        elif "No text content" in error_message or "empty" in error_message.lower():
            error_code = "NO_CONTENT_EXTRACTED"
        
        error_response = MagicToolsErrorResponse(
            success=False,
            error=error_message,
            error_code=error_code,
            details="An error occurred during file content extraction"
        )
        
        status_code = 400 if error_code in ["DOWNLOAD_ERROR", "UNSUPPORTED_FILE_TYPE", "NO_CONTENT_EXTRACTED"] else 500
        
        return JSONResponse(
            status_code=status_code,
            content=error_response.dict()
        )

