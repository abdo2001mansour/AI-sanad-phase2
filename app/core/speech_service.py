from openai import OpenAI, AsyncOpenAI
from typing import Optional
import io
from app.config.settings import settings


class SpeechService:
    """Service for handling text-to-speech and speech-to-text operations using OpenAI API"""
    
    def __init__(self):
        """Initialize the speech service with OpenAI API configuration"""
        self.api_key_configured = bool(settings.OPENAI_API_KEY)
        
        if self.api_key_configured:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            self.client = None
            self.async_client = None
    
    def text_to_speech(
        self,
        text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0
    ) -> bytes:
        """
        Convert text to speech using OpenAI TTS API.
        
        Args:
            text: The text to convert to speech
            model: The TTS model to use (tts-1 or tts-1-hd)
            voice: The voice to use (alloy, echo, fable, onyx, nova, shimmer)
            response_format: The audio format (mp3, opus, aac, flac)
            speed: The speed of the generated audio (0.25 to 4.0)
        
        Returns:
            bytes: Audio file content
        
        Raises:
            Exception: If API key is not configured or if an error occurs
        """
        if not self.api_key_configured:
            raise Exception("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            response = self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=response_format,
                speed=speed
            )
            
            # Read the audio content
            audio_content = response.content
            
            return audio_content
        
        except Exception as e:
            raise Exception(f"Error generating speech: {str(e)}")
    
    async def text_to_speech_async(
        self,
        text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        response_format: str = "mp3",
        speed: float = 1.0
    ) -> bytes:
        """
        Async version of text_to_speech.
        
        Args:
            text: The text to convert to speech
            model: The TTS model to use (tts-1 or tts-1-hd)
            voice: The voice to use (alloy, echo, fable, onyx, nova, shimmer)
            response_format: The audio format (mp3, opus, aac, flac)
            speed: The speed of the generated audio (0.25 to 4.0)
        
        Returns:
            bytes: Audio file content
        
        Raises:
            Exception: If API key is not configured or if an error occurs
        """
        if not self.api_key_configured:
            raise Exception("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            response = await self.async_client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=response_format,
                speed=speed
            )
            
            # Read the audio content
            audio_content = response.content
            
            return audio_content
        
        except Exception as e:
            raise Exception(f"Error generating speech: {str(e)}")
    
    def speech_to_text(
        self,
        audio_file: bytes,
        filename: str = "audio.mp3",
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0
    ) -> dict:
        """
        Convert speech to text using OpenAI Whisper API.
        
        Args:
            audio_file: The audio file content as bytes
            filename: The filename of the audio file (used to determine format)
            model: The model to use (whisper-1)
            language: Optional language code (e.g., 'en', 'ar', 'es')
            prompt: Optional text prompt to guide the model
            response_format: The format of the response (json, text, srt, verbose_json, vtt)
            temperature: Sampling temperature (0.0 to 1.0)
        
        Returns:
            dict: Transcription result with text and metadata
        
        Raises:
            Exception: If API key is not configured or if an error occurs
        """
        if not self.api_key_configured:
            raise Exception("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
        
        if not audio_file:
            raise ValueError("Audio file cannot be empty")
        
        try:
            # Create a file-like object from bytes
            audio_file_obj = io.BytesIO(audio_file)
            audio_file_obj.name = filename
            
            # Prepare the request parameters
            params = {
                "model": model,
                "file": audio_file_obj,
                "response_format": response_format,
                "temperature": temperature
            }
            
            if language:
                params["language"] = language
            if prompt:
                params["prompt"] = prompt
            
            # Call the transcription API
            transcript = self.client.audio.transcriptions.create(**params)
            
            # Format the response based on response_format
            if response_format == "json":
                # JSON format returns a Transcription object with text attribute
                return {
                    "text": transcript.text,
                    "language": getattr(transcript, 'language', None),
                    "duration": getattr(transcript, 'duration', None),
                    "words": None,
                    "segments": None,
                    "format": "json"
                }
            elif response_format == "verbose_json":
                # Verbose JSON includes more details like words and segments
                return {
                    "text": transcript.text,
                    "language": getattr(transcript, 'language', None),
                    "duration": getattr(transcript, 'duration', None),
                    "words": getattr(transcript, 'words', None),
                    "segments": getattr(transcript, 'segments', None),
                    "format": "verbose_json"
                }
            else:
                # For text, srt, vtt formats, transcript is a string
                return {
                    "text": str(transcript),
                    "language": None,
                    "duration": None,
                    "words": None,
                    "segments": None,
                    "format": response_format
                }
        
        except Exception as e:
            raise Exception(f"Error transcribing audio: {str(e)}")
    
    async def speech_to_text_async(
        self,
        audio_file: bytes,
        filename: str = "audio.mp3",
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0
    ) -> dict:
        """
        Async version of speech_to_text.
        
        Args:
            audio_file: The audio file content as bytes
            filename: The filename of the audio file (used to determine format)
            model: The model to use (whisper-1)
            language: Optional language code (e.g., 'en', 'ar', 'es')
            prompt: Optional text prompt to guide the model
            response_format: The format of the response (json, text, srt, verbose_json, vtt)
            temperature: Sampling temperature (0.0 to 1.0)
        
        Returns:
            dict: Transcription result with text and metadata
        
        Raises:
            Exception: If API key is not configured or if an error occurs
        """
        if not self.api_key_configured:
            raise Exception("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
        
        if not audio_file:
            raise ValueError("Audio file cannot be empty")
        
        try:
            # Create a file-like object from bytes
            audio_file_obj = io.BytesIO(audio_file)
            audio_file_obj.name = filename
            
            # Prepare the request parameters
            params = {
                "model": model,
                "file": audio_file_obj,
                "response_format": response_format,
                "temperature": temperature
            }
            
            if language:
                params["language"] = language
            if prompt:
                params["prompt"] = prompt
            
            # Call the transcription API
            transcript = await self.async_client.audio.transcriptions.create(**params)
            
            # Format the response based on response_format
            if response_format == "json":
                # JSON format returns a Transcription object with text attribute
                return {
                    "text": transcript.text,
                    "language": getattr(transcript, 'language', None),
                    "duration": getattr(transcript, 'duration', None),
                    "words": None,
                    "segments": None,
                    "format": "json"
                }
            elif response_format == "verbose_json":
                # Verbose JSON includes more details like words and segments
                return {
                    "text": transcript.text,
                    "language": getattr(transcript, 'language', None),
                    "duration": getattr(transcript, 'duration', None),
                    "words": getattr(transcript, 'words', None),
                    "segments": getattr(transcript, 'segments', None),
                    "format": "verbose_json"
                }
            else:
                # For text, srt, vtt formats, transcript is a string
                return {
                    "text": str(transcript),
                    "language": None,
                    "duration": None,
                    "words": None,
                    "segments": None,
                    "format": response_format
                }
        
        except Exception as e:
            raise Exception(f"Error transcribing audio: {str(e)}")


# Create a singleton instance
speech_service = SpeechService()

