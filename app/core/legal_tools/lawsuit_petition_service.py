"""
Lawsuit Petition Draft Service

This service generates professional lawsuit petitions using OpenAI directly.
"""
import os
from typing import Optional, AsyncGenerator
from openai import AsyncOpenAI
from app.config.settings import settings
from app.prompts.legal_tools.lawsuit_petition import (
    get_petition_system_prompt,
    get_petition_user_prompt,
    get_case_types_data,
    get_courts_data
)


class LawsuitPetitionService:
    """Service for generating lawsuit petitions using OpenAI directly"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = os.getenv("LEGAL_TOOLS_MODEL", "gpt-5.2-2025-12-11")
    
    async def generate_petition_stream(
        self,
        text: str,
        case_type: Optional[str] = None,
        court: Optional[str] = None,
        parties: Optional[str] = None,
        facts: Optional[str] = None,
        requests: Optional[str] = None,
        file_content: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate lawsuit petition with streaming response using OpenAI directly
        """
        if not case_type:
            case_type = "commercial"
        
        system_prompt = get_petition_system_prompt(case_type)
        
        full_text = text
        if file_content:
            full_text = f"{text}\n\n{file_content}"
        
        user_prompt = get_petition_user_prompt(
            full_text, 
            case_type=case_type,
            court=court,
            parties=parties,
            facts=facts,
            requests=requests
        )
        
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
    
    def get_case_types(self) -> list:
        """Get available case types"""
        return get_case_types_data()
    
    def get_courts(self) -> list:
        """Get available courts"""
        return get_courts_data()


# Singleton instance
lawsuit_petition_service = LawsuitPetitionService()

