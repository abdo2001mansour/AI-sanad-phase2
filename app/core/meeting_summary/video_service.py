"""
Video to Audio conversion service using moviepy.
Supports: .mp4, .avi, .mov, .mkv, .webm, .flv
Max duration: 1 hour
"""

import os
import uuid
import tempfile
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from datetime import datetime
import httpx
from app.config.settings import settings

# Configure ffmpeg from static-ffmpeg BEFORE importing moviepy
# This ensures moviepy uses the bundled ffmpeg binary
try:
    import static_ffmpeg
    # This downloads the ffmpeg binary if not already present
    ffmpeg_path, ffprobe_path = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
    # Also set for moviepy's config
    os.environ["FFMPEG_BINARY"] = ffmpeg_path
    print(f"Using bundled ffmpeg from static-ffmpeg: {ffmpeg_path}")
    FFMPEG_AVAILABLE = True
except ImportError:
    print("Warning: static-ffmpeg not available. Install with: pip install static-ffmpeg")
    FFMPEG_AVAILABLE = False
except Exception as e:
    print(f"Warning: Could not configure ffmpeg: {e}")
    FFMPEG_AVAILABLE = False

# Try to import moviepy (supports both old and new API)
try:
    # moviepy 2.0+ uses direct import
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    try:
        # moviepy 1.x uses editor submodule
        from moviepy.editor import VideoFileClip
        MOVIEPY_AVAILABLE = True
    except ImportError:
        MOVIEPY_AVAILABLE = False
        VideoFileClip = None
        print("Warning: moviepy not available. Install with: pip install moviepy")

# Import S3 service
try:
    from app.core.database.s3_service import s3_service
    S3_AVAILABLE = s3_service is not None and s3_service.api_configured
except ImportError:
    S3_AVAILABLE = False
    s3_service = None


class VideoService:
    """Service for video to audio conversion with S3 storage"""
    
    # Supported video formats
    SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"]
    
    # Maximum duration in seconds (1 hour)
    MAX_DURATION_SECONDS = 3600
    
    # Output audio format
    OUTPUT_AUDIO_FORMAT = "mp3"
    OUTPUT_AUDIO_BITRATE = "192k"
    
    # S3 bucket for meeting files
    MEETING_BUCKET = "sanad-data-source"
    MEETING_VIDEO_PREFIX = "meeting-videos/"
    MEETING_PREFIX = "meeting-audio/"
    
    # Allowed URL patterns for video source (sanad-data-source S3 bucket)
    ALLOWED_URL_PATTERNS = [
        "sanad-data-source.s3",  # S3 direct URL (e.g., sanad-data-source.s3.amazonaws.com)
        "s3.amazonaws.com/sanad-data-source",  # Path-style S3 URL
        "s3.eu-north-1.amazonaws.com/sanad-data-source",  # Regional path-style URL
        ".s3.amazonaws.com/",  # Virtual-hosted S3 with signed URL (presigned URLs)
        ".s3.eu-north-1.amazonaws.com/",  # Regional virtual-hosted URL
    ]
    
    def __init__(self):
        """Initialize the video service"""
        self.moviepy_available = MOVIEPY_AVAILABLE and FFMPEG_AVAILABLE
        self.ffmpeg_available = FFMPEG_AVAILABLE
        self.s3_available = S3_AVAILABLE
        
        # Get CDN URL from settings for validation
        self.cdn_url = settings.AWS_CDN_URL
        if self.cdn_url:
            # Add CDN URL to allowed patterns
            cdn_domain = self.cdn_url.replace("https://", "").replace("http://", "").rstrip("/")
            self.ALLOWED_URL_PATTERNS.append(cdn_domain)
        
        if not FFMPEG_AVAILABLE:
            print("Warning: ffmpeg not available. Install with: pip install imageio-ffmpeg")
        
        if not MOVIEPY_AVAILABLE:
            print("Warning: moviepy not available. Install with: pip install moviepy")
        
        if not self.s3_available:
            print("Warning: S3 service not available. Audio files will not be uploaded.")
    
    def _get_file_extension(self, filename: str) -> str:
        """Get lowercase file extension from filename or URL"""
        # Handle URLs - get the path component
        if "?" in filename:
            filename = filename.split("?")[0]
        
        _, ext = os.path.splitext(filename.lower())
        return ext
    
    def _is_supported_format(self, filename: str) -> bool:
        """Check if the video format is supported"""
        ext = self._get_file_extension(filename)
        return ext in self.SUPPORTED_VIDEO_FORMATS
    
    def _is_allowed_url(self, url: str) -> bool:
        """
        Check if the URL is from the allowed S3 bucket (sanad-data-source).
        
        Args:
            url: URL to validate
        
        Returns:
            True if URL is from allowed source, False otherwise
        """
        url_lower = url.lower()
        
        for pattern in self.ALLOWED_URL_PATTERNS:
            if pattern.lower() in url_lower:
                return True
        
        return False
    
    def _get_allowed_url_hint(self) -> str:
        """Get a hint message for allowed URLs"""
        if self.cdn_url:
            return f"Video must be uploaded to the S3 bucket first. Use the upload feature or provide a URL from: {self.cdn_url}"
        return "Video must be uploaded to the sanad-data-source S3 bucket first. Use the upload feature."
    
    async def _download_video(
        self, 
        video_url: str, 
        temp_dir: str,
        progress_callback: Optional[callable] = None
    ) -> str:
        """
        Download video from URL to temporary file.
        
        Args:
            video_url: URL of the video
            temp_dir: Temporary directory to save the file
            progress_callback: Optional callback for progress updates
        
        Returns:
            Path to the downloaded video file
        """
        # Get filename from URL
        url_path = video_url.split("?")[0]  # Remove query params
        original_filename = os.path.basename(url_path) or "video.mp4"
        ext = self._get_file_extension(original_filename)
        
        if not ext:
            ext = ".mp4"  # Default extension
        
        # Create temp file path
        temp_video_path = os.path.join(temp_dir, f"input_video{ext}")
        
        if progress_callback:
            await progress_callback("download_start", {"url": video_url, "filename": original_filename})
        
        # Download video
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 min timeout
            async with client.stream("GET", video_url) as response:
                response.raise_for_status()
                
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                
                with open(temp_video_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):  # 1MB chunks
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress = (downloaded / total_size) * 100
                            await progress_callback("download_progress", {
                                "downloaded": downloaded,
                                "total": total_size,
                                "progress": round(progress, 1)
                            })
        
        if progress_callback:
            await progress_callback("download_complete", {
                "path": temp_video_path,
                "size": os.path.getsize(temp_video_path)
            })
        
        return temp_video_path
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds"""
        if not self.moviepy_available:
            raise Exception("moviepy not available")
        
        with VideoFileClip(video_path) as video:
            return video.duration
    
    def _convert_to_audio(
        self, 
        video_path: str, 
        output_path: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Convert video file to audio (MP3).
        
        Args:
            video_path: Path to input video
            output_path: Path for output audio
            progress_callback: Optional sync callback for progress
        
        Returns:
            Dictionary with conversion info
        """
        if not self.moviepy_available:
            raise Exception("moviepy not available. Install with: pip install moviepy")
        
        # Load video
        video = VideoFileClip(video_path)
        
        try:
            duration = video.duration
            
            # Check duration limit
            if duration > self.MAX_DURATION_SECONDS:
                raise Exception(
                    f"Video duration ({duration:.1f}s) exceeds maximum allowed ({self.MAX_DURATION_SECONDS}s = 1 hour)"
                )
            
            # Extract audio
            audio = video.audio
            
            if audio is None:
                raise Exception("Video has no audio track")
            
            # Write audio file (compatible with both moviepy 1.x and 2.x)
            try:
                # moviepy 2.x API
                audio.write_audiofile(
                    output_path,
                    fps=44100,
                    nbytes=2,
                    bitrate=self.OUTPUT_AUDIO_BITRATE,
                    codec="libmp3lame",
                    logger=None
                )
            except TypeError:
                # moviepy 1.x API (with verbose parameter)
                audio.write_audiofile(
                    output_path,
                    fps=44100,
                    nbytes=2,
                    bitrate=self.OUTPUT_AUDIO_BITRATE,
                    codec="libmp3lame",
                    verbose=False,
                    logger=None
                )
            
            audio_size = os.path.getsize(output_path)
            
            return {
                "duration_seconds": duration,
                "audio_size": audio_size,
                "output_path": output_path
            }
        
        finally:
            video.close()
    
    async def _upload_to_s3(
        self, 
        file_path: str, 
        s3_key: str,
        content_type: str = "audio/mpeg",
        progress_callback: Optional[callable] = None
    ) -> Dict[str, str]:
        """
        Upload file to S3 bucket.
        
        Args:
            file_path: Local file path
            s3_key: S3 object key
            content_type: MIME type
            progress_callback: Optional callback for progress
        
        Returns:
            Dictionary with S3 URLs
        """
        if not self.s3_available or not s3_service:
            raise Exception("S3 service not available")
        
        if progress_callback:
            await progress_callback("upload_start", {"key": s3_key})
        
        # Read file
        with open(file_path, "rb") as f:
            file_content = f.read()
        
        # Upload to S3
        try:
            s3_service.s3_client.put_object(
                Bucket=self.MEETING_BUCKET,
                Key=s3_key,
                Body=file_content,
                ContentType=content_type
            )
        except Exception as e:
            raise Exception(f"Failed to upload to S3: {str(e)}")
        
        # Generate URLs
        cdn_url = None
        presigned_url = None
        
        try:
            cdn_url = s3_service.get_cdn_url(s3_key)
        except:
            pass
        
        try:
            presigned_url = s3_service.get_presigned_url(
                bucket_name=self.MEETING_BUCKET,
                key=s3_key,
                expiration=86400  # 24 hours
            )
        except:
            pass
        
        if progress_callback:
            await progress_callback("upload_complete", {
                "key": s3_key,
                "cdn_url": cdn_url,
                "presigned_url": presigned_url
            })
        
        return {
            "bucket": self.MEETING_BUCKET,
            "key": s3_key,
            "cdn_url": cdn_url,
            "presigned_url": presigned_url
        }
    
    async def convert_video_to_audio_stream(
        self, 
        video_url: str,
        original_filename: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Convert video to audio with SSE streaming progress logs.
        
        Args:
            video_url: URL of the video to convert
            original_filename: Optional original filename
        
        Yields:
            SSE-formatted log messages and final result
        """
        import json
        
        # Generate unique ID for this conversion
        conversion_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        def log_event(event_type: str, data: Dict[str, Any]) -> str:
            """Format SSE event"""
            event_data = {
                "event": event_type,
                "timestamp": datetime.now().isoformat(),
                "conversion_id": conversion_id,
                **data
            }
            return f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
        
        # Validate dependencies
        if not self.ffmpeg_available:
            yield log_event("error", {
                "message": "ffmpeg not available. Please install: pip install static-ffmpeg",
                "error_code": "FFMPEG_NOT_AVAILABLE"
            })
            return
        
        if not MOVIEPY_AVAILABLE:
            yield log_event("error", {
                "message": "moviepy not available. Please install: pip install moviepy",
                "error_code": "DEPENDENCY_ERROR"
            })
            return
        
        if not self.s3_available:
            yield log_event("error", {
                "message": "S3 service not available. Check AWS credentials.",
                "error_code": "S3_NOT_CONFIGURED"
            })
            return
        
        # Validate URL
        if not video_url or not video_url.strip():
            yield log_event("error", {
                "message": "Video URL cannot be empty",
                "error_code": "EMPTY_URL"
            })
            return
        
        if not (video_url.startswith("http://") or video_url.startswith("https://")):
            yield log_event("error", {
                "message": "Invalid URL format. Must start with http:// or https://",
                "error_code": "INVALID_URL"
            })
            return
        
        # Validate URL is from allowed S3 bucket
        if not self._is_allowed_url(video_url):
            yield log_event("error", {
                "message": f"Video URL must be from the sanad-data-source S3 bucket. {self._get_allowed_url_hint()}",
                "error_code": "INVALID_SOURCE",
                "allowed_patterns": self.ALLOWED_URL_PATTERNS
            })
            return
        
        # Check file format
        filename = original_filename or os.path.basename(video_url.split("?")[0])
        if not self._is_supported_format(filename):
            ext = self._get_file_extension(filename)
            yield log_event("error", {
                "message": f"Unsupported video format: {ext}. Supported: {', '.join(self.SUPPORTED_VIDEO_FORMATS)}",
                "error_code": "UNSUPPORTED_FORMAT"
            })
            return
        
        # Start conversion
        yield log_event("start", {
            "message": "Starting video to audio conversion",
            "video_url": video_url,
            "filename": filename
        })
        
        temp_dir = None
        
        try:
            # Create temp directory
            temp_dir = tempfile.mkdtemp(prefix="video_convert_")
            yield log_event("log", {"message": f"Created temporary directory"})
            
            # Progress callback for async operations
            async def progress_callback(event: str, data: Dict[str, Any]):
                pass  # We'll yield directly in the main flow
            
            # Download video
            yield log_event("log", {"message": "Downloading video file..."})
            
            video_path = None
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("GET", video_url) as response:
                    if response.status_code != 200:
                        yield log_event("error", {
                            "message": f"Failed to download video. HTTP status: {response.status_code}",
                            "error_code": "DOWNLOAD_FAILED"
                        })
                        return
                    
                    total_size = int(response.headers.get("content-length", 0))
                    downloaded = 0
                    
                    ext = self._get_file_extension(filename) or ".mp4"
                    video_path = os.path.join(temp_dir, f"input{ext}")
                    
                    with open(video_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                if int(progress) % 10 == 0:  # Log every 10%
                                    yield log_event("progress", {
                                        "stage": "download",
                                        "downloaded_mb": round(downloaded / (1024 * 1024), 1),
                                        "total_mb": round(total_size / (1024 * 1024), 1),
                                        "progress": round(progress, 1)
                                    })
            
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            yield log_event("log", {
                "message": f"Video downloaded: {file_size_mb:.1f} MB"
            })
            
            # Check video duration
            yield log_event("log", {"message": "Checking video duration..."})
            
            duration = await asyncio.get_event_loop().run_in_executor(
                None, self._get_video_duration, video_path
            )
            
            duration_minutes = duration / 60
            yield log_event("log", {
                "message": f"Video duration: {duration_minutes:.1f} minutes"
            })
            
            if duration > self.MAX_DURATION_SECONDS:
                yield log_event("error", {
                    "message": f"Video too long ({duration_minutes:.1f} min). Maximum is 60 minutes.",
                    "error_code": "DURATION_EXCEEDED",
                    "duration_seconds": duration,
                    "max_duration_seconds": self.MAX_DURATION_SECONDS
                })
                return
            
            # Convert to audio
            yield log_event("log", {"message": "Converting video to audio (MP3)..."})
            yield log_event("progress", {"stage": "convert", "progress": 0})
            
            audio_path = os.path.join(temp_dir, f"output.{self.OUTPUT_AUDIO_FORMAT}")
            
            # Run conversion in thread pool
            conversion_result = await asyncio.get_event_loop().run_in_executor(
                None, self._convert_to_audio, video_path, audio_path, None
            )
            
            audio_size_mb = conversion_result["audio_size"] / (1024 * 1024)
            yield log_event("log", {
                "message": f"Audio extracted: {audio_size_mb:.1f} MB"
            })
            yield log_event("progress", {"stage": "convert", "progress": 100})
            
            # Upload to S3
            yield log_event("log", {"message": "Uploading audio to S3..."})
            yield log_event("progress", {"stage": "upload", "progress": 0})
            
            # Generate S3 key
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(filename)[0]
            s3_key = f"{self.MEETING_PREFIX}{timestamp}_{base_name}_{conversion_id[:8]}.{self.OUTPUT_AUDIO_FORMAT}"
            
            # Upload
            upload_result = await self._upload_to_s3(
                audio_path, 
                s3_key,
                content_type="audio/mpeg"
            )
            
            yield log_event("progress", {"stage": "upload", "progress": 100})
            yield log_event("log", {"message": "Audio uploaded to S3 successfully"})
            
            # Calculate total time
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Final result
            yield log_event("complete", {
                "message": "Conversion completed successfully",
                "audio_url": upload_result.get("cdn_url") or upload_result.get("presigned_url"),
                "cdn_url": upload_result.get("cdn_url"),
                "presigned_url": upload_result.get("presigned_url"),
                "s3_bucket": upload_result["bucket"],
                "s3_key": upload_result["key"],
                "duration_seconds": duration,
                "duration_minutes": round(duration / 60, 1),
                "audio_size_mb": round(audio_size_mb, 2),
                "processing_time_seconds": round(total_time, 1)
            })
            
            yield "data: [DONE]\n\n"
        
        except httpx.HTTPStatusError as e:
            yield log_event("error", {
                "message": f"HTTP error downloading video: {e.response.status_code}",
                "error_code": "HTTP_ERROR"
            })
        
        except Exception as e:
            yield log_event("error", {
                "message": f"Conversion failed: {str(e)}",
                "error_code": "CONVERSION_ERROR"
            })
        
        finally:
            # Cleanup temp files
            if temp_dir and os.path.exists(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except:
                    pass


# Create singleton instance
video_service = VideoService()

