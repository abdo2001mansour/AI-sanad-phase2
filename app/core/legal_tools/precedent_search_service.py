"""
Precedent Search Service

This service searches for judicial precedents using OpenAI.
Returns structured JSON.
"""
import os
import json
from typing import Optional, List
from openai import AsyncOpenAI
from app.config.settings import settings
from app.prompts.legal_tools.precedent_search import (
    get_precedent_system_prompt,
    get_precedent_user_prompt,
    get_court_types_data
)


class PrecedentSearchService:
    """Service for searching judicial precedents using OpenAI"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = os.getenv("LEGAL_TOOLS_MODEL", "gpt-4.1-2025-04-14")
    
    async def search_precedents(
        self,
        text: str,
        court_type: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        file_content: Optional[str] = None
    ) -> dict:
        """
        Search for judicial precedents and return structured JSON response
        """
        if not court_type:
            court_type = "all"
        
        system_prompt = get_precedent_system_prompt(court_type)
        
        full_text = text
        if file_content:
            full_text = f"{text}\n\n{file_content}"
        
        user_prompt = get_precedent_user_prompt(full_text, court_type, keywords)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.4,
                max_completion_tokens=6000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = {
                    "query_summary": text[:200],
                    "court_filter": court_type,
                    "total_results": 0,
                    "precedents": [],
                    "error": "فشل في تحليل الاستجابة"
                }
            
            return result
                        
        except Exception as e:
            return {
                "query_summary": text[:200],
                "court_filter": court_type,
                "total_results": 0,
                "precedents": [],
                "error": str(e)
            }
    
    def get_court_types(self) -> list:
        """Get available court types"""
        return get_court_types_data()


# Singleton instance
precedent_search_service = PrecedentSearchService()
