"""
Regulatory Updates Service

This service tracks regulatory and legislative updates using OpenAI.
Returns structured JSON.
"""
import os
import json
from typing import Optional
from openai import AsyncOpenAI
from app.config.settings import settings
from app.prompts.legal_tools.regulatory_updates import (
    get_regulatory_system_prompt,
    get_regulatory_user_prompt,
    get_legal_domains_data
)


class RegulatoryUpdatesService:
    """Service for tracking regulatory updates using OpenAI"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = os.getenv("LEGAL_TOOLS_MODEL", "gpt-4.1-2025-04-14")
    
    async def get_updates(
        self,
        text: str,
        legal_domain: Optional[str] = None,
        file_content: Optional[str] = None
    ) -> dict:
        """
        Get regulatory updates and return structured JSON response
        """
        if not legal_domain:
            legal_domain = "commercial"
        
        system_prompt = get_regulatory_system_prompt(legal_domain)
        
        full_text = text
        if file_content:
            full_text = f"{text}\n\n{file_content}"
        
        user_prompt = get_regulatory_user_prompt(full_text, legal_domain)
        
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
                    "legal_domain": legal_domain,
                    "query_topic": text[:200],
                    "total_updates": 0,
                    "updates": [],
                    "error": "فشل في تحليل الاستجابة"
                }
            
            return result
                        
        except Exception as e:
            return {
                "legal_domain": legal_domain,
                "query_topic": text[:200],
                "total_updates": 0,
                "updates": [],
                "error": str(e)
            }
    
    def get_legal_domains(self) -> list:
        """Get available legal domains"""
        return get_legal_domains_data()


# Singleton instance
regulatory_updates_service = RegulatoryUpdatesService()
