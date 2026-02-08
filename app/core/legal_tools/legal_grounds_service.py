"""
Legal Grounds Service

This service links facts to relevant legal provisions using OpenAI.
Returns structured JSON with citations organized by source type.
"""
import os
import json
from typing import Optional
from openai import AsyncOpenAI
from app.config.settings import settings
from app.prompts.legal_tools.legal_grounds import (
    get_grounds_system_prompt,
    get_grounds_user_prompt
)


class LegalGroundsService:
    """Service for finding legal grounds using OpenAI"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = os.getenv("LEGAL_TOOLS_MODEL", "gpt-4.1-2025-04-14")
    
    async def find_legal_grounds(
        self,
        text: str,
        file_content: Optional[str] = None
    ) -> dict:
        """
        Find legal grounds and return structured JSON response
        """
        system_prompt = get_grounds_system_prompt()
        
        full_text = text
        if file_content:
            full_text = f"{text}\n\n{file_content}"
        
        user_prompt = get_grounds_user_prompt(full_text)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_completion_tokens=8000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = {
                    "facts_summary": text[:200],
                    "legal_grounds": {
                        "sharia_principles": [],
                        "regulations": [],
                        "executive_bylaws": [],
                        "regulatory_decisions": [],
                        "judicial_precedents": [],
                        "jurisprudential_principles": [],
                        "international_agreements": []
                    },
                    "citation_summary": "",
                    "strongest_grounds": [],
                    "error": "فشل في تحليل الاستجابة"
                }
            
            return result
                        
        except Exception as e:
            return {
                "facts_summary": text[:200],
                "legal_grounds": {
                    "sharia_principles": [],
                    "regulations": [],
                    "executive_bylaws": [],
                    "regulatory_decisions": [],
                    "judicial_precedents": [],
                    "jurisprudential_principles": [],
                    "international_agreements": []
                },
                "citation_summary": "",
                "strongest_grounds": [],
                "error": str(e)
            }


# Singleton instance
legal_grounds_service = LegalGroundsService()
