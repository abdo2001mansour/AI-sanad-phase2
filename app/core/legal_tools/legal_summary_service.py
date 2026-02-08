"""
Legal Summary Service

This service summarizes legal documents using OpenAI.
Returns structured JSON.
"""
import os
import json
from typing import Optional
from openai import AsyncOpenAI
from app.config.settings import settings
from app.prompts.legal_tools.legal_summary import (
    get_summary_system_prompt,
    get_summary_user_prompt,
    get_summary_types_data
)


class LegalSummaryService:
    """Service for summarizing legal documents using OpenAI"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = os.getenv("LEGAL_TOOLS_MODEL", "gpt-4.1-2025-04-14")
    
    async def summarize_document(
        self,
        text: str,
        summary_type: Optional[str] = None,
        file_content: Optional[str] = None
    ) -> dict:
        """
        Summarize legal document and return structured JSON response
        """
        if not summary_type:
            summary_type = "executive"
        
        system_prompt = get_summary_system_prompt(summary_type)
        
        full_text = text
        if file_content:
            full_text = f"{text}\n\n{file_content}"
        
        user_prompt = get_summary_user_prompt(full_text, summary_type)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_completion_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = {
                    "summary_type": summary_type,
                    "main_summary": "حدث خطأ في معالجة النتائج",
                    "key_points": [],
                    "error": "فشل في تحليل الاستجابة"
                }
            
            return result
                        
        except Exception as e:
            return {
                "summary_type": summary_type,
                "main_summary": "",
                "key_points": [],
                "error": str(e)
            }
    
    def get_summary_types(self) -> list:
        """Get available summary types"""
        return get_summary_types_data()


# Singleton instance
legal_summary_service = LegalSummaryService()
