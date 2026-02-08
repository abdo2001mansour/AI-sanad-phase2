"""
Language Detection Service

Detects text language using OpenAI GPT-5.2
"""
import json
import os
from pathlib import Path
from openai import AsyncOpenAI
from app.config.settings import settings
from app.prompts.magic_tools.language_detection import get_language_detection_system_prompt


class LanguageDetectionService:
    """Service for detecting text language using OpenAI GPT-5.2"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = os.getenv("MAGIC_TOOLS_MODEL", "gpt-5.2-2025-12-11")
        self.languages = self._load_languages_config()
    
    def _load_languages_config(self) -> dict:
        """Load languages configuration from JSON file"""
        try:
            config_path = Path(__file__).parent.parent.parent / "config" / "languages_config.json"
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load languages config: {e}")
            return {}
    
    async def detect_language(self, text: str, model: str = None) -> dict:
        """Detect the language of the given text"""
        model = model or self.model
        
        try:
            system_prompt = get_language_detection_system_prompt()
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
                max_completion_tokens=10
            )
            
            detected_code = response.choices[0].message.content.strip().lower()
            
            # Look up language info
            if detected_code in self.languages:
                lang_info = self.languages[detected_code]
                return {
                    "success": True,
                    "detected_language": detected_code,
                    "language_name_en": lang_info.get("name_en", detected_code),
                    "language_name_ar": lang_info.get("name_ar", detected_code)
                }
            else:
                return {
                    "success": True,
                    "detected_language": detected_code,
                    "language_name_en": detected_code,
                    "language_name_ar": detected_code
                }
                
        except Exception as e:
            print(f"Language detection error: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_code": "DETECTION_ERROR"
            }


language_detection_service = LanguageDetectionService()
