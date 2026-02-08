"""
Rephrasing Service

Rephrases text using OpenAI GPT-5.2
"""
import json
import os
from typing import AsyncGenerator
from openai import AsyncOpenAI
from app.config.settings import settings
from app.prompts.magic_tools.rephrasing import get_rephrasing_system_prompt


class RephrasingService:
    """Service for rephrasing text using OpenAI GPT-5.2"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = os.getenv("MAGIC_TOOLS_MODEL", "gpt-5.2-2025-12-11")
    
    async def rephrase_text_stream(
        self,
        text: str,
        model: str = None
    ) -> AsyncGenerator[str, None]:
        """Rephrase text with streaming"""
        model = model or self.model
        
        try:
            system_prompt = get_rephrasing_system_prompt()
            
            stream = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Rephrase this text:\n\n{text}"}
                ],
                temperature=0.4,
                max_completion_tokens=4096,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    chunk_data = {
                        "success": True,
                        "type": "rephrasing",
                        "content": content
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            
            yield f"data: {json.dumps({'success': True, 'type': 'rephrasing', 'completed': True}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"Rephrasing error: {e}")
            yield f"data: {json.dumps({'success': False, 'error': str(e), 'error_code': 'REPHRASING_ERROR'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
    
    async def rephrase_text(
        self,
        text: str,
        model: str = None
    ) -> dict:
        """Rephrase text (non-streaming)"""
        model = model or self.model
        
        try:
            system_prompt = get_rephrasing_system_prompt()
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Rephrase this text:\n\n{text}"}
                ],
                temperature=0.4,
                max_completion_tokens=4096
            )
            
            rephrased = response.choices[0].message.content
            
            return {
                "success": True,
                "type": "rephrasing",
                "content": rephrased
            }
            
        except Exception as e:
            print(f"Rephrasing error: {e}")
            raise Exception(f"Rephrasing failed: {str(e)}")


rephrasing_service = RephrasingService()
