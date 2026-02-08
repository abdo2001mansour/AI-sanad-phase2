"""
Judgment Analysis Service

This service analyzes court judgments using OpenAI directly.
"""
import os
from typing import Optional, AsyncGenerator
from openai import AsyncOpenAI
from app.config.settings import settings
from app.prompts.legal_tools.judgment_analysis import (
    get_judgment_system_prompt,
    get_judgment_user_prompt,
    ANALYSIS_TYPES
)


class JudgmentAnalysisService:
    """Service for analyzing court judgments using OpenAI directly"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = os.getenv("LEGAL_TOOLS_MODEL", "gpt-5.2-2025-12-11")
    
    async def analyze_judgment_stream(
        self,
        text: str,
        analysis_type: Optional[str] = None,
        file_content: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Analyze judgment with streaming response using OpenAI directly
        """
        if not analysis_type:
            analysis_type = "detailed"
        
        system_prompt = get_judgment_system_prompt(analysis_type)
        
        full_text = text
        if file_content:
            full_text = f"{text}\n\n{file_content}"
        
        user_prompt = get_judgment_user_prompt(full_text, analysis_type)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_completion_tokens=4000,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
                        
        except Exception as e:
            yield f"\n\n[خطأ: {str(e)}]"
    
    def get_analysis_types(self) -> list:
        """Get available analysis types"""
        return [
            {"id": k, "name_ar": v["name_ar"], "name_en": v["name_en"]}
            for k, v in ANALYSIS_TYPES.items()
        ]


# Singleton instance
judgment_analysis_service = JudgmentAnalysisService()
