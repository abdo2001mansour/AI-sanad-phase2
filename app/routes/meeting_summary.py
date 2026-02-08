from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
import json
import asyncio
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from app.dto.schemas import MagicToolsErrorResponse, MagicToolsSuccessResponse, FileRequest

router = APIRouter(prefix="/meeting-summary", tags=["Meeting Summary"])


class SummaryType(str, Enum):
    """Summary type options"""
    BRIEF = "brief"
    DETAILED = "detailed"


class VideoToAudioRequest(BaseModel):
    """Request schema for video to audio conversion"""
    video_url: str = Field(..., description="URL of the video file to convert to audio")
    filename: Optional[str] = Field(None, description="Original filename (optional, used for naming the output)")


class VideoToAudioResponse(BaseModel):
    """Response schema for video to audio conversion (SSE stream)"""
    success: bool = Field(True, description="Success status")
    audio_url: str = Field(..., description="URL of the converted audio file (CDN or presigned)")
    cdn_url: Optional[str] = Field(None, description="CDN URL if available")
    presigned_url: Optional[str] = Field(None, description="Presigned URL (24h expiry)")
    s3_bucket: str = Field(..., description="S3 bucket name")
    s3_key: str = Field(..., description="S3 object key")
    duration_seconds: float = Field(..., description="Duration of the audio in seconds")
    duration_minutes: float = Field(..., description="Duration of the audio in minutes")
    audio_size_mb: float = Field(..., description="Size of the audio file in MB")
    processing_time_seconds: float = Field(..., description="Total processing time")


class AudioToTextRequest(BaseModel):
    """Request schema for audio to text transcription"""
    audio_url: str = Field(..., description="URL of the audio file to transcribe")
    file: Optional[FileRequest] = Field(default=None, description="File request object with list_of_files. If provided, returns DOC file URLs instead of text")


class AudioToTextResponse(BaseModel):
    """Response schema for audio to text transcription"""
    success: bool = Field(True, description="Success status")
    transcript: str = Field(..., description="Transcribed text from the audio")
    audio_url: str = Field(..., description="Original audio URL")
    duration_seconds: Optional[float] = Field(None, description="Duration of the audio in seconds")


class TranscriptSummaryRequest(BaseModel):
    """Request schema for meeting transcript summary"""
    transcript: str = Field(..., description="Meeting transcript text to summarize", min_length=1)
    type: SummaryType = Field(..., description="Type of summary: brief or detailed")
    file: Optional[FileRequest] = Field(default=None, description="File request object with list_of_files. If provided, returns DOC file URLs instead of text")


class TranscriptSummaryResponse(BaseModel):
    """Response schema for meeting transcript summary"""
    success: bool = Field(True, description="Success status")
    summary: str = Field(..., description="Summary of the meeting transcript")
    key_points: Optional[List[str]] = Field(None, description="List of key points from the meeting")
    duration_minutes: Optional[float] = Field(None, description="Estimated meeting duration in minutes")


class SupportedFilesResponse(BaseModel):
    """Response schema for supported file types"""
    supported_audio_formats: List[str] = Field(..., description="Supported audio file formats")
    supported_video_formats: List[str] = Field(..., description="Supported video file formats")
    summary_types: List[str] = Field(..., description="Available summary types for meeting transcripts")
    max_video_duration_minutes: int = Field(60, description="Maximum video duration in minutes")


@router.post("/video-to-audio", summary="Convert video to audio (SSE streaming)")
async def convert_video_to_audio(request: VideoToAudioRequest, raw_request: Request):
    """
    Convert a video file to MP3 audio format with real-time progress streaming.
    
    **Features:**
    - Downloads video from URL
    - Validates video duration (max 1 hour)
    - Converts to MP3 audio (192kbps)
    - Uploads to S3 bucket
    - Returns CDN/presigned URL for the audio file
    
    **Supported formats:** .mp4, .avi, .mov, .mkv, .webm, .flv
    
    **Maximum duration:** 60 minutes (1 hour)
    
    **Response:** Server-Sent Events (SSE) stream with:
    - Progress logs during download, conversion, and upload
    - Final event with audio URL and metadata
    
    **SSE Event Types:**
    - `start`: Conversion started
    - `log`: General log message
    - `progress`: Progress update with stage and percentage
    - `complete`: Conversion completed with audio URL
    - `error`: Error occurred
    
    **Example SSE events:**
    ```
    data: {"event": "start", "message": "Starting video to audio conversion", ...}
    data: {"event": "progress", "stage": "download", "progress": 50.0, ...}
    data: {"event": "progress", "stage": "convert", "progress": 100, ...}
    data: {"event": "complete", "audio_url": "https://...", "duration_seconds": 1800, ...}
    data: [DONE]
    ```
    """
    try:
        from app.core.meeting_summary.video_service import video_service
        
        # Validate input
        if not request.video_url or not request.video_url.strip():
            error_response = MagicToolsErrorResponse(
                success=False,
                error="Video URL cannot be empty",
                error_code="EMPTY_URL"
            )
            return JSONResponse(
                status_code=400,
                content=error_response.dict()
            )
        
        # Validate URL format
        if not (request.video_url.startswith("http://") or request.video_url.startswith("https://")):
            error_response = MagicToolsErrorResponse(
                success=False,
                error="Invalid URL format. URL must start with http:// or https://",
                error_code="INVALID_URL_FORMAT"
            )
            return JSONResponse(
                status_code=400,
                content=error_response.dict()
            )
        
        # Stream conversion progress
        async def conversion_stream():
            try:
                async for chunk in video_service.convert_video_to_audio_stream(
                    video_url=request.video_url,
                    original_filename=request.filename
                ):
                    # Check if client disconnected
                    if await raw_request.is_disconnected():
                        return
                    yield chunk
            except asyncio.CancelledError:
                return
            except Exception as e:
                error_data = {
                    "event": "error",
                    "message": str(e),
                    "error_code": "STREAM_ERROR"
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            conversion_stream(),
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
            error_code="VIDEO_TO_AUDIO_ERROR",
            details="An error occurred during video to audio conversion"
        )
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )


@router.post("/audio-to-text", summary="Transcribe audio to text (SSE streaming)")
async def transcribe_audio_to_text(request: AudioToTextRequest, raw_request: Request):
    """
    Transcribe an audio file to text using OpenAI Whisper with real-time progress streaming.
    
    **Features:**
    - Downloads audio from URL
    - Validates audio format and file size (max 25MB)
    - Transcribes using OpenAI Whisper API
    - Returns transcribed text
    
    **Supported formats:** .mp3, .wav, .m4a, .aac, .ogg, .flac
    
    **Maximum file size:** 25 MB (OpenAI Whisper limit)
    
    **Response:** Server-Sent Events (SSE) stream with:
    - Progress logs during download and transcription
    - Final event with transcribed text
    
    **SSE Event Types:**
    - `started`: Transcription started
    - `progress`: Progress update with step and percentage
    - `completed`: Transcription completed with text
    - `error`: Error occurred
    
    **Example SSE events:**
    ```
    data: {"event": "started", "message": "Starting audio to text transcription...", ...}
    data: {"event": "progress", "step": "download", "progress": 30, ...}
    data: {"event": "progress", "step": "transcription", "progress": 60, ...}
    data: {"event": "completed", "text": "...", "word_count": 500, ...}
    data: [DONE]
    ```
    """
    try:
        from app.core.meeting_summary.transcription_service import transcription_service
        import uuid
        
        # Validate input
        if not request.audio_url or not request.audio_url.strip():
            error_response = MagicToolsErrorResponse(
                success=False,
                error="Audio URL cannot be empty",
                error_code="EMPTY_URL"
            )
            return JSONResponse(
                status_code=400,
                content=error_response.dict()
            )
        
        # Validate URL format
        if not (request.audio_url.startswith("http://") or request.audio_url.startswith("https://")):
            error_response = MagicToolsErrorResponse(
                success=False,
                error="Invalid URL format. URL must start with http:// or https://",
                error_code="INVALID_URL_FORMAT"
            )
            return JSONResponse(
                status_code=400,
                content=error_response.dict()
            )
        
        # Stream transcription progress
        async def transcription_stream():
            try:
                transcription_id = str(uuid.uuid4())
                async for event_data in transcription_service.stream_audio_to_text(
                    audio_url=request.audio_url,
                    transcription_id=transcription_id
                ):
                    # Check if client disconnected
                    if await raw_request.is_disconnected():
                        return
                    yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                return
            except Exception as e:
                error_data = {
                    "event": "error",
                    "message": str(e),
                    "error_code": "STREAM_ERROR"
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            transcription_stream(),
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
            error_code="AUDIO_TO_TEXT_ERROR",
            details="An error occurred during audio transcription"
        )
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )


@router.post("/transcript-summary", summary="Summarize meeting transcript (SSE streaming)")
async def summarize_transcript(request: TranscriptSummaryRequest, raw_request: Request):
    """
    Summarize a meeting transcript using Google Gemini with real-time streaming.
    
    **Features:**
    - Summarizes meeting transcripts using AI
    - Extracts key points and action items
    - Supports two summary types
    
    **Summary Types:**
    - `brief`: Concise 3-5 sentence summary with main points
    - `detailed`: Comprehensive summary with sections for topics, decisions, action items
    
    **Response:** Server-Sent Events (SSE) stream with:
    - Summary text streamed word by word
    - Key points extracted from the meeting
    
    **SSE Event Types:**
    - `meeting_summary`: Summary content chunk
    - `completed`: Summary generation completed
    - `error`: Error occurred
    
    **Example SSE events:**
    ```
    data: {"success": true, "type": "meeting_summary", "content": "The ", ...}
    data: {"success": true, "type": "meeting_summary", "content": "meeting ", ...}
    data: {"success": true, "type": "meeting_summary", "completed": true, ...}
    data: [DONE]
    ```
    """
    try:
        from app.core.meeting_summary.summary_service import meeting_summary_service
        
        # Validate input
        if not request.transcript or not request.transcript.strip():
            error_response = MagicToolsErrorResponse(
                success=False,
                error="Transcript cannot be empty",
                error_code="EMPTY_TRANSCRIPT"
            )
            return JSONResponse(
                status_code=400,
                content=error_response.dict()
            )
        
        # Stream summary generation
        async def summary_stream():
            try:
                async for chunk in meeting_summary_service.summarize_transcript_stream(
                    transcript=request.transcript,
                    summary_type=request.type.value
                ):
                    # Check if client disconnected
                    if await raw_request.is_disconnected():
                        return
                    yield chunk
            except asyncio.CancelledError:
                return
            except Exception as e:
                error_data = {
                    "success": False,
                    "error": str(e),
                    "error_code": "STREAM_ERROR"
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            summary_stream(),
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
            error_code="TRANSCRIPT_SUMMARY_ERROR",
            details="An error occurred during transcript summarization"
        )
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )


@router.get("/supported-files", response_model=SupportedFilesResponse, summary="Get supported audio and video file formats")
async def get_supported_files():
    """
    Get all supported audio and video file formats for meeting summary operations.
    
    This endpoint returns:
    - Supported audio file formats (for audio-to-text transcription)
    - Supported video file formats (for video-to-audio conversion)
    - Available summary types (brief, detailed)
    - Maximum video duration (60 minutes)
    """
    return SupportedFilesResponse(
        supported_audio_formats=[
            ".mp3",
            ".wav",
            ".m4a",
            ".aac",
            ".ogg",
            ".flac"
        ],
        supported_video_formats=[
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm",
            ".flv"
        ],
        summary_types=[
            "brief",
            "detailed"
        ],
        max_video_duration_minutes=60
    )


@router.get("/health", summary="Health check for meeting summary service")
async def health_check():
    """
    Check the health status of meeting summary services.
    
    Returns status of:
    - ffmpeg (audio extraction)
    - moviepy (video processing)
    - openai (transcription)
    - gemini (summarization)
    - S3 service (file storage)
    """
    try:
        from app.core.meeting_summary.video_service import video_service, FFMPEG_AVAILABLE, MOVIEPY_AVAILABLE
        from app.core.meeting_summary.transcription_service import transcription_service, OPENAI_AVAILABLE
        from app.core.meeting_summary.summary_service import meeting_summary_service
        
        # Determine overall health
        video_healthy = video_service.moviepy_available and video_service.s3_available
        transcription_healthy = transcription_service.openai_available
        summary_healthy = meeting_summary_service.api_key_configured
        all_healthy = video_healthy and transcription_healthy and summary_healthy
        
        return JSONResponse(
            status_code=200 if all_healthy else 503,
            content={
                "status": "healthy" if all_healthy else "unhealthy",
                "services": {
                    "ffmpeg": {
                        "available": FFMPEG_AVAILABLE,
                        "message": "Ready (bundled via static-ffmpeg)" if FFMPEG_AVAILABLE else "Not installed - run: pip install static-ffmpeg"
                    },
                    "moviepy": {
                        "available": MOVIEPY_AVAILABLE,
                        "message": "Ready" if MOVIEPY_AVAILABLE else "Not installed - run: pip install moviepy"
                    },
                    "video_processing": {
                        "available": video_service.moviepy_available,
                        "message": "Ready" if video_service.moviepy_available else "Requires both moviepy and ffmpeg"
                    },
                    "openai_whisper": {
                        "available": transcription_service.openai_available,
                        "message": "Ready" if transcription_service.openai_available else "Not configured - check OPENAI_API_KEY"
                    },
                    "gemini_summary": {
                        "available": meeting_summary_service.api_key_configured,
                        "message": "Ready" if meeting_summary_service.api_key_configured else "Not configured - check GOOGLE_API_KEY"
                    },
                    "s3": {
                        "available": video_service.s3_available,
                        "message": "Ready" if video_service.s3_available else "Not configured - check AWS credentials"
                    }
                },
                "supported_video_formats": video_service.SUPPORTED_VIDEO_FORMATS,
                "supported_audio_formats": transcription_service.SUPPORTED_AUDIO_FORMATS,
                "summary_types": ["brief", "detailed"],
                "max_video_duration_minutes": video_service.MAX_DURATION_SECONDS // 60,
                "max_audio_size_mb": transcription_service.MAX_FILE_SIZE_BYTES / (1024 * 1024),
                "output_format": video_service.OUTPUT_AUDIO_FORMAT
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@router.get("/debug", summary="Debug information for meeting summary services")
async def debug_info():
    """
    Get detailed debug information about meeting summary services.
    
    Returns:
    - Detailed import status for all dependencies
    - Error messages for failed imports
    - System information
    """
    import sys
    import traceback
    
    debug_data = {
        "python_version": sys.version,
        "imports": {},
        "errors": []
    }
    
    # Check static_ffmpeg
    try:
        import static_ffmpeg
        ffmpeg_path, ffprobe_path = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
        debug_data["imports"]["static_ffmpeg"] = {
            "available": True,
            "ffmpeg_path": ffmpeg_path,
            "ffprobe_path": ffprobe_path
        }
    except Exception as e:
        debug_data["imports"]["static_ffmpeg"] = {
            "available": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        debug_data["errors"].append(f"static_ffmpeg: {str(e)}")
    
    # Check imageio
    try:
        import imageio
        debug_data["imports"]["imageio"] = {
            "available": True,
            "version": getattr(imageio, '__version__', 'unknown')
        }
    except Exception as e:
        debug_data["imports"]["imageio"] = {
            "available": False,
            "error": str(e)
        }
        debug_data["errors"].append(f"imageio: {str(e)}")
    
    # Check imageio_ffmpeg
    try:
        import imageio_ffmpeg
        debug_data["imports"]["imageio_ffmpeg"] = {
            "available": True,
            "version": getattr(imageio_ffmpeg, '__version__', 'unknown')
        }
    except Exception as e:
        debug_data["imports"]["imageio_ffmpeg"] = {
            "available": False,
            "error": str(e)
        }
        debug_data["errors"].append(f"imageio_ffmpeg: {str(e)}")
    
    # Check moviepy (supports both old and new API)
    try:
        # Try moviepy 2.0+ first
        from moviepy import VideoFileClip
        debug_data["imports"]["moviepy"] = {
            "available": True,
            "VideoFileClip": "imported successfully (moviepy 2.x)"
        }
    except ImportError:
        try:
            # Fall back to moviepy 1.x
            from moviepy.editor import VideoFileClip
            debug_data["imports"]["moviepy"] = {
                "available": True,
                "VideoFileClip": "imported successfully (moviepy 1.x)"
            }
        except Exception as e:
            debug_data["imports"]["moviepy"] = {
                "available": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            debug_data["errors"].append(f"moviepy: {str(e)}")
    
    # Check numpy
    try:
        import numpy as np
        debug_data["imports"]["numpy"] = {
            "available": True,
            "version": np.__version__
        }
    except Exception as e:
        debug_data["imports"]["numpy"] = {
            "available": False,
            "error": str(e)
        }
        debug_data["errors"].append(f"numpy: {str(e)}")
    
    # Check decorator
    try:
        import decorator
        debug_data["imports"]["decorator"] = {
            "available": True,
            "version": getattr(decorator, '__version__', 'unknown')
        }
    except Exception as e:
        debug_data["imports"]["decorator"] = {
            "available": False,
            "error": str(e)
        }
        debug_data["errors"].append(f"decorator: {str(e)}")
    
    # Check proglog
    try:
        import proglog
        debug_data["imports"]["proglog"] = {
            "available": True
        }
    except Exception as e:
        debug_data["imports"]["proglog"] = {
            "available": False,
            "error": str(e)
        }
        debug_data["errors"].append(f"proglog: {str(e)}")
    
    # Check openai
    try:
        import openai
        debug_data["imports"]["openai"] = {
            "available": True,
            "version": getattr(openai, '__version__', 'unknown')
        }
    except Exception as e:
        debug_data["imports"]["openai"] = {
            "available": False,
            "error": str(e)
        }
        debug_data["errors"].append(f"openai: {str(e)}")
    
    # Check google genai
    try:
        from google import genai
        debug_data["imports"]["google_genai"] = {
            "available": True
        }
    except Exception as e:
        debug_data["imports"]["google_genai"] = {
            "available": False,
            "error": str(e)
        }
        debug_data["errors"].append(f"google_genai: {str(e)}")
    
    # Check video service
    try:
        from app.core.meeting_summary.video_service import video_service, FFMPEG_AVAILABLE, MOVIEPY_AVAILABLE
        debug_data["video_service"] = {
            "initialized": True,
            "FFMPEG_AVAILABLE": FFMPEG_AVAILABLE,
            "MOVIEPY_AVAILABLE": MOVIEPY_AVAILABLE,
            "moviepy_available": video_service.moviepy_available,
            "ffmpeg_available": video_service.ffmpeg_available,
            "s3_available": video_service.s3_available
        }
    except Exception as e:
        debug_data["video_service"] = {
            "initialized": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        debug_data["errors"].append(f"video_service: {str(e)}")
    
    # Check transcription service
    try:
        from app.core.meeting_summary.transcription_service import transcription_service, OPENAI_AVAILABLE
        debug_data["transcription_service"] = {
            "initialized": True,
            "OPENAI_AVAILABLE": OPENAI_AVAILABLE,
            "openai_available": transcription_service.openai_available,
            "s3_available": transcription_service.s3_available
        }
    except Exception as e:
        debug_data["transcription_service"] = {
            "initialized": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        debug_data["errors"].append(f"transcription_service: {str(e)}")
    
    # Check summary service
    try:
        from app.core.meeting_summary.summary_service import meeting_summary_service
        debug_data["summary_service"] = {
            "initialized": True,
            "api_key_configured": meeting_summary_service.api_key_configured
        }
    except Exception as e:
        debug_data["summary_service"] = {
            "initialized": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        debug_data["errors"].append(f"summary_service: {str(e)}")
    
    return JSONResponse(
        status_code=200,
        content=debug_data
    )


@router.get("/logs", summary="Get recent server logs")
async def get_server_logs(
    count: int = 100,
    level: Optional[str] = None
):
    """
    Get the most recent server logs.
    
    Args:
        count: Number of log entries to return (default: 100, max: 1000)
        level: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        List of recent log entries with timestamp, level, logger, and message
    """
    from app.logging_config import get_recent_logs
    
    # Limit count to reasonable value
    count = min(count, 1000)
    
    logs = get_recent_logs(count)
    
    # Filter by level if specified
    if level:
        level = level.upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level in valid_levels:
            logs = [log for log in logs if log.get("level") == level]
    
    return JSONResponse(
        status_code=200,
        content={
            "total_logs": len(logs),
            "requested_count": count,
            "level_filter": level,
            "logs": logs
        }
    )


@router.get("/logs/errors", summary="Get recent error logs")
async def get_error_logs(count: int = 50):
    """
    Get only ERROR and CRITICAL level logs.
    
    Args:
        count: Number of log entries to return (default: 50, max: 500)
    
    Returns:
        List of recent error log entries
    """
    from app.logging_config import get_recent_logs
    
    count = min(count, 500)
    
    all_logs = get_recent_logs(1000)  # Get more to filter
    
    error_logs = [
        log for log in all_logs 
        if log.get("level") in ["ERROR", "CRITICAL"]
    ]
    
    # Return most recent errors up to count
    error_logs = error_logs[-count:] if len(error_logs) > count else error_logs
    
    return JSONResponse(
        status_code=200,
        content={
            "total_errors": len(error_logs),
            "logs": error_logs
        }
    )


@router.delete("/logs", summary="Clear server logs")
async def clear_server_logs():
    """
    Clear the in-memory log buffer.
    
    Returns:
        Confirmation message
    """
    from app.logging_config import clear_logs
    
    clear_logs()
    
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "Log buffer cleared"
        }
    )
