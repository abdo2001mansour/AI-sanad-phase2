"""
Legal Article Explanation Service

This service explains legal articles and regulations using OpenAI directly.
"""
import os
from typing import Optional, AsyncGenerator
from openai import AsyncOpenAI
from app.config.settings import settings
from app.prompts.legal_tools.article_explanation import (
    get_article_system_prompt,
    get_article_user_prompt,
    get_explanation_levels_data
)


class ArticleExplanationService:
    """Service for explaining legal articles using OpenAI directly"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = os.getenv("LEGAL_TOOLS_MODEL", "gpt-5.2-2025-12-11")
    
    async def explain_article_stream(
        self,
        text: str,
        explanation_level: Optional[str] = None,
        file_content: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Explain legal article with streaming response using OpenAI directly
        """
        if not explanation_level:
            explanation_level = "simple"
        
        system_prompt = get_article_system_prompt(explanation_level)
        
        full_text = text
        if file_content:
            full_text = f"{text}\n\n{file_content}"
        
        user_prompt = get_article_user_prompt(full_text, explanation_level)
        
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
            yield f"\n\n[Error: {str(e)}]"
    
    def get_explanation_levels(self) -> list:
        """Get available explanation levels"""
        return get_explanation_levels_data()


# Singleton instance
article_explanation_service = ArticleExplanationService()

