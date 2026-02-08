from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import Response, StreamingResponse
from typing import Optional
import io
from app.core.speech_service import speech_service
from app.dto.schemas import (
    TextToSpeechRequest,
    SpeechToTextRequest,
    SpeechToTextResponse,
    ErrorResponse
)

router = APIRouter(prefix="/speech", tags=["Speech"])

# Supported audio file types for speech-to-text
SUPPORTED_AUDIO_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/mp4", "audio/wav", 
    "audio/webm", "audio/ogg", "audio/flac", "audio/m4a"
}


@router.post("/text-to-speech", summary="Convert text to speech")
async def text_to_speech(request: TextToSpeechRequest):
    """
    Convert text to speech using OpenAI TTS API.
    
    This endpoint takes text input and generates an audio file using OpenAI's text-to-speech model.
    
    **Parameters:**
    - **text**: The text to convert to speech (required)
    - **model**: TTS model to use - `tts-1` (faster) or `tts-1-hd` (higher quality)
    - **voice**: Voice to use - `alloy`, `echo`, `fable`, `onyx`, `nova`, or `shimmer`
    - **response_format**: Audio format - `mp3`, `opus`, `aac`, or `flac`
    - **speed**: Speed of the generated audio (0.25 to 4.0, default: 1.0)
    
    **Returns:**
    - Audio file in the specified format
    """
    try:
        # Generate speech
        audio_content = await speech_service.text_to_speech_async(
            text=request.text,
            model=request.model,
            voice=request.voice,
            response_format=request.response_format,
            speed=request.speed
        )
        
        # Determine content type based on response format
        content_type_map = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac"
        }
        content_type = content_type_map.get(request.response_format, "audio/mpeg")
        
        # Return audio file
        return Response(
            content=audio_content,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{request.response_format}"'
            }
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")


@router.post("/speech-to-text", response_model=SpeechToTextResponse, summary="Convert speech to text")
async def speech_to_text(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(None, description="Optional language code (e.g., 'en', 'ar', 'es')"),
    prompt: Optional[str] = Form(None, description="Optional text prompt to guide the model"),
    response_format: str = Form("json", description="Response format (json, text, srt, verbose_json, vtt)"),
    temperature: float = Form(0.0, ge=0.0, le=1.0, description="Sampling temperature (0.0 to 1.0)")
):
    """
    Convert speech to text using OpenAI Whisper API.
    
    This endpoint takes an audio file and transcribes it to text using OpenAI's Whisper model.
    
    **Supported Audio Formats:**
    - MP3, MP4, WAV, WebM, OGG, FLAC, M4A
    
    **Parameters:**
    - **file**: Audio file to transcribe (required)
    - **language**: Optional language code to help with transcription accuracy
    - **prompt**: Optional text prompt to guide the model (useful for proper nouns, technical terms)
    - **response_format**: Response format - `json` (default), `text`, `srt`, `verbose_json`, or `vtt`
    - **temperature**: Sampling temperature (0.0 to 1.0, default: 0.0)
    
    **Returns:**
    - Transcribed text with optional metadata (language, duration, timestamps)
    """
    try:
        # Validate file type
        if file.content_type and file.content_type not in SUPPORTED_AUDIO_TYPES:
            # Also check filename extension as fallback
            filename_lower = (file.filename or "").lower()
            if not any(filename_lower.endswith(ext) for ext in ['.mp3', '.mp4', '.wav', '.webm', '.ogg', '.flac', '.m4a', '.mpeg']):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.content_type or 'unknown'}. Supported types: MP3, MP4, WAV, WebM, OGG, FLAC, M4A"
                )
        
        # Read file content
        file_content = await file.read()
        
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Get filename or use default
        filename = file.filename or "audio.mp3"
        
        # Transcribe audio
        result = await speech_service.speech_to_text_async(
            audio_file=file_content,
            filename=filename,
            model="whisper-1",
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature
        )
        
        # Ensure the response matches our schema
        response_data = {
            "text": result.get("text", ""),
            "language": result.get("language"),
            "duration": result.get("duration"),
            "words": result.get("words"),
            "segments": result.get("segments"),
            "format": result.get("format", response_format)
        }
        
        return SpeechToTextResponse(**response_data)
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")


@router.get("/health", summary="Speech service health check")
async def health_check():
    """Health check endpoint for speech service"""
    return {
        "status": "healthy",
        "service": "speech",
        "api_key_configured": speech_service.api_key_configured
    }

