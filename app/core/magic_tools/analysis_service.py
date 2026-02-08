"""
Analysis Service

Analyzes text using OpenAI GPT-5.2
"""
import json
import os
from typing import AsyncGenerator
from openai import AsyncOpenAI
from app.config.settings import settings
from app.prompts.magic_tools.analysis import get_analysis_system_prompt


class AnalysisService:
    """Service for analyzing text using OpenAI GPT-5.2"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = os.getenv("MAGIC_TOOLS_MODEL", "gpt-5.2-2025-12-11")
    
    async def analyze_text_stream(
        self,
        text: str,
        analysis_type: str = "brief",
        model: str = None
    ) -> AsyncGenerator[str, None]:
        """Analyze text with streaming"""
        model = model or self.model
        
        try:
            system_prompt = get_analysis_system_prompt(analysis_type)
            
            stream = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this text:\n\n{text}"}
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
                        "type": "analysis",
                        "content": content,
                        "analysis_type": analysis_type
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            
            yield f"data: {json.dumps({'success': True, 'type': 'analysis', 'completed': True}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"Analysis error: {e}")
            yield f"data: {json.dumps({'success': False, 'error': str(e), 'error_code': 'ANALYSIS_ERROR'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
    
    async def analyze_text(
        self,
        text: str,
        analysis_type: str = "brief",
        model: str = None
    ) -> dict:
        """Analyze text (non-streaming)"""
        model = model or self.model
        
        try:
            system_prompt = get_analysis_system_prompt(analysis_type)
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this text:\n\n{text}"}
                ],
                temperature=0.3,
                max_completion_tokens=4096
            )
            
            analysis = response.choices[0].message.content
            
            return {
                "success": True,
                "type": "analysis",
                "content": analysis,
                "analysis_type": analysis_type
            }
            
        except Exception as e:
            print(f"Analysis error: {e}")
            raise Exception(f"Analysis failed: {str(e)}")


analysis_service = AnalysisService()
