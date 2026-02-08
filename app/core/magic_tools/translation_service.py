"""
Translation Service

Translates text using OpenAI GPT-5.2
"""
import json
import os
from typing import AsyncGenerator
from pathlib import Path
from openai import AsyncOpenAI
from app.config.settings import settings
from app.prompts.magic_tools.translation import get_translation_system_prompt


class TranslationService:
    """Service for translating text using OpenAI GPT-5.2"""
    
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
    
    def _get_language_name(self, lang_code: str) -> str:
        """Get language name from code"""
        lang_code_lower = lang_code.lower()
        if lang_code_lower in self.languages:
            return self.languages[lang_code_lower]["name_en"]
        for code, lang_info in self.languages.items():
            if lang_info["name_en"].lower() == lang_code_lower or lang_info["name_ar"] == lang_code:
                return lang_info["name_en"]
        return lang_code
    
    async def translate_text_stream(
        self,
        text: str,
        target_language: str,
        translation_type: str = "general",
        model: str = None
    ) -> AsyncGenerator[str, None]:
        """Translate text with streaming"""
        model = model or self.model
        language_name = self._get_language_name(target_language)
        
        try:
            system_prompt = get_translation_system_prompt(language_name, translation_type)
            
            stream = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_completion_tokens=8192,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    chunk_data = {
                        "success": True,
                        "type": "translation",
                        "content": content,
                        "language": target_language,
                        "translation_type": translation_type
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            
            yield f"data: {json.dumps({'success': True, 'type': 'translation', 'completed': True}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"Translation error: {e}")
            yield f"data: {json.dumps({'success': False, 'error': str(e), 'error_code': 'TRANSLATION_ERROR'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
    
    async def translate_text(
        self,
        text: str,
        target_language: str,
        translation_type: str = "general",
        model: str = None
    ) -> dict:
        """Translate text (non-streaming)"""
        model = model or self.model
        language_name = self._get_language_name(target_language)
        
        try:
            system_prompt = get_translation_system_prompt(language_name, translation_type)
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_completion_tokens=8192
            )
            
            translated_text = response.choices[0].message.content
            
            return {
                "success": True,
                "type": "translation",
                "content": translated_text,
                "language": target_language,
                "translation_type": translation_type
            }
            
        except Exception as e:
            print(f"Translation error: {e}")
            raise Exception(f"Translation failed: {str(e)}")


translation_service = TranslationService()
