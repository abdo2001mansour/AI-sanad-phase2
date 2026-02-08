"""
Audio to Text transcription service using OpenAI Whisper API.
Supports: .mp3, .wav, .m4a, .aac, .ogg, .flac
"""

import os
import uuid
import tempfile
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from datetime import datetime
import httpx
from app.config.settings import settings

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not available. Install with: pip install openai")

# Import S3 service
try:
    from app.core.database.s3_service import s3_service
    S3_AVAILABLE = s3_service is not None and s3_service.api_configured
except ImportError:
    S3_AVAILABLE = False
    s3_service = None


class TranscriptionService:
    """Service for audio to text transcription using OpenAI Whisper"""
    
    # Supported audio formats
    SUPPORTED_AUDIO_FORMATS = [".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"]
    
    # Maximum file size in bytes (25 MB - OpenAI Whisper limit)
    MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
    
    # S3 bucket for meeting files
    MEETING_BUCKET = "sanad-data-source"
    MEETING_AUDIO_PREFIX = "meeting-audio/"
    TRANSCRIPTION_PREFIX = "meeting-transcriptions/"
    
    # Allowed URL patterns for audio source (sanad-data-source S3 bucket)
    ALLOWED_URL_PATTERNS = [
        "sanad-data-source.s3",
        "s3.amazonaws.com/sanad-data-source",
        "s3.eu-north-1.amazonaws.com/sanad-data-source",
        ".s3.amazonaws.com/",
        ".s3.eu-north-1.amazonaws.com/",
    ]
    
    def __init__(self):
        """Initialize the transcription service"""
        self.openai_available = OPENAI_AVAILABLE and bool(settings.OPENAI_API_KEY)
        self.s3_available = S3_AVAILABLE
        
        # Initialize OpenAI client
        if self.openai_available:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            self.client = None
        
        # Get CDN URL from settings for validation
        self.cdn_url = settings.AWS_CDN_URL
        if self.cdn_url:
            cdn_domain = self.cdn_url.replace("https://", "").replace("http://", "").rstrip("/")
            self.ALLOWED_URL_PATTERNS.append(cdn_domain)
        
        if not OPENAI_AVAILABLE:
            print("Warning: openai not available. Install with: pip install openai")
        elif not settings.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY not configured. Transcription will not work.")
        
        if not self.s3_available:
            print("Warning: S3 service not available.")
    
    def _get_file_extension(self, filename: str) -> str:
        """Get file extension from filename"""
        if '.' in filename:
            return '.' + filename.rsplit('.', 1)[1].lower()
        return ''
    
    def _is_s3_url(self, url: str) -> bool:
        """Check if the URL is from the allowed S3 bucket or CDN"""
        url_lower = url.lower()
        for pattern in self.ALLOWED_URL_PATTERNS:
            if pattern.lower() in url_lower:
                return True
        return False
    
    def _extract_filename_from_url(self, url: str) -> str:
        """Extract filename from URL"""
        # Remove query parameters
        clean_url = url.split('?')[0]
        # Get the last part of the path
        filename = clean_url.rsplit('/', 1)[-1]
        # URL decode if needed
        import urllib.parse
        filename = urllib.parse.unquote(filename)
        return filename if filename else "audio"
    
    def _validate_audio_format(self, filename: str) -> bool:
        """Check if the file has a supported audio format"""
        ext = self._get_file_extension(filename)
        return ext in self.SUPPORTED_AUDIO_FORMATS
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size for display"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    
    async def stream_audio_to_text(
        self,
        audio_url: str,
        filename: Optional[str] = None,
        language: Optional[str] = None,
        transcription_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream audio to text transcription with progress updates.
        
        Args:
            audio_url: URL of the audio file (must be from sanad-data-source bucket)
            filename: Optional custom filename
            language: Optional language code (e.g., 'ar', 'en')
            transcription_id: Optional ID for tracking
            
        Yields:
            Dict with event type and data
        """
        transcription_id = transcription_id or str(uuid.uuid4())
        start_time = datetime.now()
        
        def log_event(event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
            """Create a log event"""
            return {
                "event": event_type,
                "transcription_id": transcription_id,
                "timestamp": datetime.now().isoformat(),
                **data
            }
        
        # Validate dependencies
        if not self.openai_available:
            yield log_event("error", {
                "message": "OpenAI not available. Please check OPENAI_API_KEY configuration.",
                "error_code": "OPENAI_NOT_AVAILABLE"
            })
            return
        
        # Validate URL is from allowed S3 bucket
        if not self._is_s3_url(audio_url):
            yield log_event("error", {
                "message": "Audio URL must be from the sanad-data-source S3 bucket or its configured CDN. External audio sources are not allowed.",
                "error_code": "INVALID_SOURCE"
            })
            return
        
        yield log_event("started", {
            "message": "Starting audio to text transcription...",
            "audio_url": audio_url[:100] + "..." if len(audio_url) > 100 else audio_url,
            "progress": 0
        })
        
        # Extract filename if not provided
        if not filename:
            filename = self._extract_filename_from_url(audio_url)
        
        # Validate audio format
        if not self._validate_audio_format(filename):
            ext = self._get_file_extension(filename)
            yield log_event("error", {
                "message": f"Unsupported audio format: {ext}. Supported formats: {', '.join(self.SUPPORTED_AUDIO_FORMATS)}",
                "error_code": "UNSUPPORTED_FORMAT"
            })
            return
        
        yield log_event("progress", {
            "message": f"Audio file: {filename}",
            "step": "validation",
            "progress": 5
        })
        
        temp_audio_path = None
        
        try:
            # Create temp directory
            temp_dir = tempfile.mkdtemp(prefix="transcription_")
            ext = self._get_file_extension(filename)
            temp_audio_path = os.path.join(temp_dir, f"audio_{transcription_id}{ext}")
            
            yield log_event("progress", {
                "message": "Downloading audio file...",
                "step": "download",
                "progress": 10
            })
            
            # Download audio file with progress tracking
            async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
                async with client.stream("GET", audio_url) as response:
                    if response.status_code != 200:
                        yield log_event("error", {
                            "message": f"Failed to download audio: HTTP {response.status_code}",
                            "error_code": "DOWNLOAD_FAILED"
                        })
                        return
                    
                    # Get content length if available
                    total_size = int(response.headers.get('content-length', 0))
                    
                    if total_size > self.MAX_FILE_SIZE_BYTES:
                        yield log_event("error", {
                            "message": f"Audio file too large. Maximum size: {self._format_file_size(self.MAX_FILE_SIZE_BYTES)}. File size: {self._format_file_size(total_size)}",
                            "error_code": "FILE_TOO_LARGE"
                        })
                        return
                    
                    downloaded_size = 0
                    last_progress_update = 0
                    
                    with open(temp_audio_path, 'wb') as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            # Check file size during download
                            if downloaded_size > self.MAX_FILE_SIZE_BYTES:
                                yield log_event("error", {
                                    "message": f"Audio file too large. Maximum size: {self._format_file_size(self.MAX_FILE_SIZE_BYTES)}",
                                    "error_code": "FILE_TOO_LARGE"
                                })
                                return
                            
                            # Calculate and yield download progress (10-40%)
                            if total_size > 0:
                                progress = int(10 + (downloaded_size / total_size) * 30)
                                if progress >= last_progress_update + 5:
                                    last_progress_update = progress
                                    yield log_event("progress", {
                                        "message": f"Downloading: {self._format_file_size(downloaded_size)} / {self._format_file_size(total_size)}",
                                        "step": "download",
                                        "progress": progress
                                    })
            
            file_size = os.path.getsize(temp_audio_path)
            yield log_event("progress", {
                "message": f"Download complete: {self._format_file_size(file_size)}",
                "step": "download_complete",
                "progress": 40
            })
            
            # Transcribe with OpenAI Whisper
            yield log_event("progress", {
                "message": "Transcribing audio with OpenAI Whisper...",
                "step": "transcription",
                "progress": 45
            })
            
            # Run transcription in a thread pool to not block
            transcription_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._transcribe_audio(temp_audio_path, language)
            )
            
            if "error" in transcription_result:
                yield log_event("error", {
                    "message": transcription_result["error"],
                    "error_code": "TRANSCRIPTION_FAILED"
                })
                return
            
            yield log_event("progress", {
                "message": "Transcription complete!",
                "step": "transcription_complete",
                "progress": 90
            })
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Get the transcribed text
            transcribed_text = transcription_result.get("text", "")
            word_count = len(transcribed_text.split()) if transcribed_text else 0
            
            yield log_event("progress", {
                "message": f"Processed {word_count} words in {processing_time:.1f}s",
                "step": "finalizing",
                "progress": 95
            })
            
            # Optionally save transcription to S3
            transcription_url = None
            if self.s3_available and transcribed_text:
                try:
                    transcription_filename = f"{transcription_id}.txt"
                    object_name = f"{self.TRANSCRIPTION_PREFIX}{transcription_filename}"
                    
                    upload_result = s3_service.upload_file(
                        file_content=transcribed_text.encode('utf-8'),
                        object_name=object_name,
                        bucket_name=self.MEETING_BUCKET,
                        content_type='text/plain; charset=utf-8'
                    )
                    
                    if upload_result.get("success"):
                        transcription_url = upload_result.get("cdn_url") or upload_result.get("presigned_url")
                        yield log_event("progress", {
                            "message": "Transcription saved to S3",
                            "step": "upload",
                            "progress": 98
                        })
                except Exception as e:
                    print(f"Warning: Could not save transcription to S3: {e}")
            
            # Final success event
            yield log_event("completed", {
                "message": "Audio transcription completed successfully!",
                "progress": 100,
                "text": transcribed_text,
                "word_count": word_count,
                "processing_time_seconds": round(processing_time, 2),
                "audio_filename": filename,
                "transcription_url": transcription_url,
                "language": transcription_result.get("language")
            })
            
        except Exception as e:
            yield log_event("error", {
                "message": f"Transcription failed: {str(e)}",
                "error_code": "TRANSCRIPTION_ERROR"
            })
        
        finally:
            # Cleanup temp files
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                    temp_dir = os.path.dirname(temp_audio_path)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
                except Exception as e:
                    print(f"Warning: Could not cleanup temp files: {e}")
    
    def _transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio using OpenAI Whisper API.
        
        Args:
            audio_path: Path to the audio file
            language: Optional language code
            
        Returns:
            Dict with transcription result or error
        """
        try:
            with open(audio_path, "rb") as audio_file:
                # Prepare transcription options
                transcribe_kwargs = {
                    "model": "whisper-1",
                    "file": audio_file,
                    "response_format": "verbose_json"
                }
                
                # Add language if specified
                if language:
                    transcribe_kwargs["language"] = language
                
                # Call OpenAI Whisper API
                response = self.client.audio.transcriptions.create(**transcribe_kwargs)
                
                return {
                    "text": response.text,
                    "language": getattr(response, 'language', language),
                    "duration": getattr(response, 'duration', None)
                }
                
        except Exception as e:
            error_message = str(e)
            if "maximum" in error_message.lower() and "size" in error_message.lower():
                return {"error": f"Audio file exceeds OpenAI's 25MB limit. Please use a smaller file."}
            return {"error": f"OpenAI transcription error: {error_message}"}
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get the current status of the transcription service"""
        return {
            "openai_available": self.openai_available,
            "s3_available": self.s3_available,
            "supported_audio_formats": self.SUPPORTED_AUDIO_FORMATS,
            "max_file_size_mb": self.MAX_FILE_SIZE_BYTES / (1024 * 1024)
        }


# Create singleton instance
transcription_service = TranscriptionService()

