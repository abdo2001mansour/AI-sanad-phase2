"""
Legal Memo Generator Service

This service generates professional legal memos using OpenAI directly.
Supports all Saudi court memo types with proper legal structure.
"""
import os
from typing import Optional, AsyncGenerator
from openai import AsyncOpenAI
from app.config.settings import settings
from app.prompts.legal_tools.memo_generator import (
    get_memo_system_prompt,
    get_memo_user_prompt,
    get_memo_types_data,
    MEMO_TYPES
)


class LegalMemoGeneratorService:
    """Service for generating legal memos using OpenAI directly"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = os.getenv("LEGAL_TOOLS_MODEL", "gpt-5.2-2025-12-11")
    
    async def generate_memo_stream(
        self,
        text: str,
        memo_type: Optional[str] = None,
        file_content: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate legal memo with streaming response using OpenAI directly
        """
        if not memo_type:
            memo_type = "claim"  # Default to statement of claim
        
        system_prompt = get_memo_system_prompt(memo_type)
        
        full_text = text
        if file_content:
            full_text = f"{text}\n\n{file_content}"
        
        user_prompt = get_memo_user_prompt(full_text, memo_type)
        
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
    
    def get_memo_types(self) -> list:
        """Get available memo types with full metadata"""
        return get_memo_types_data()
    
    def get_memo_type_info(self, memo_type: str) -> dict:
        """Get info for a specific memo type"""
        return MEMO_TYPES.get(memo_type.lower(), MEMO_TYPES["claim"])


# Singleton instance
legal_memo_generator_service = LegalMemoGeneratorService()
