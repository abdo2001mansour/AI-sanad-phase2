"""
Summarization Service

Summarizes text using OpenAI GPT-5.2
"""
import json
import os
from typing import AsyncGenerator
from openai import AsyncOpenAI
from app.config.settings import settings
from app.prompts.magic_tools.summarization import get_summarization_system_prompt


class SummarizationService:
    """Service for summarizing text using OpenAI GPT-5.2"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = os.getenv("MAGIC_TOOLS_MODEL", "gpt-5.2-2025-12-11")
    
    async def summarize_text_stream(
        self,
        text: str,
        summary_type: str = "brief",
        model: str = None
    ) -> AsyncGenerator[str, None]:
        """Summarize text with streaming"""
        model = model or self.model
        
        try:
            system_prompt = get_summarization_system_prompt(summary_type)
            
            stream = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarize this text:\n\n{text}"}
                ],
                temperature=0.3,
                max_completion_tokens=4096,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    chunk_data = {
                        "success": True,
                        "type": "summarization",
                        "content": content,
                        "summary_type": summary_type
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            
            yield f"data: {json.dumps({'success': True, 'type': 'summarization', 'completed': True}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"Summarization error: {e}")
            yield f"data: {json.dumps({'success': False, 'error': str(e), 'error_code': 'SUMMARIZATION_ERROR'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
    
    async def summarize_text(
        self,
        text: str,
        summary_type: str = "brief",
        model: str = None
    ) -> dict:
        """Summarize text (non-streaming)"""
        model = model or self.model
        
        try:
            system_prompt = get_summarization_system_prompt(summary_type)
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarize this text:\n\n{text}"}
                ],
                temperature=0.3,
                max_completion_tokens=4096
            )
            
            summary = response.choices[0].message.content
            
            return {
                "success": True,
                "type": "summarization",
                "content": summary,
                "summary_type": summary_type
            }
            
        except Exception as e:
            print(f"Summarization error: {e}")
            raise Exception(f"Summarization failed: {str(e)}")


summarization_service = SummarizationService()
