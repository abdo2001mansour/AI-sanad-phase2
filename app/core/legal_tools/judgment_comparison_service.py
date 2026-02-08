"""
Judgment Comparison Service

This service compares court judgments using OpenAI directly.
Returns structured JSON for table display.
"""
import os
import json
from typing import Optional
from openai import AsyncOpenAI
from app.config.settings import settings
from app.prompts.legal_tools.judgment_comparison import (
    get_comparison_system_prompt,
    get_comparison_user_prompt,
    COMPARISON_CRITERIA
)


class JudgmentComparisonService:
    """Service for comparing court judgments using OpenAI directly"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = os.getenv("LEGAL_TOOLS_MODEL", "gpt-4.1-2025-04-14")
    
    async def compare_judgments(
        self,
        text: str,
        comparison_criteria: Optional[str] = None,
        file_content: Optional[str] = None
    ) -> dict:
        """
        Compare judgments and return structured JSON response
        """
        if not comparison_criteria:
            comparison_criteria = "all"
        
        system_prompt = get_comparison_system_prompt(comparison_criteria)
        
        full_text = text
        if file_content:
            full_text = f"{text}\n\n{file_content}"
        
        user_prompt = get_comparison_user_prompt(full_text, comparison_criteria)
        
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
            
            # Parse JSON response
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # If JSON parsing fails, return error structure
                result = {
                    "comparison_table": [],
                    "legal_conclusion": "حدث خطأ في معالجة النتائج",
                    "usage_recommendation": "",
                    "error": "فشل في تحليل الاستجابة"
                }
            
            return result
                        
        except Exception as e:
            return {
                "comparison_table": [],
                "legal_conclusion": "",
                "usage_recommendation": "",
                "error": str(e)
            }
    
    def get_comparison_criteria(self) -> list:
        """Get available comparison criteria"""
        return [
            {"id": k, "name_ar": v["name_ar"], "name_en": v["name_en"]}
            for k, v in COMPARISON_CRITERIA.items()
        ]


# Singleton instance
judgment_comparison_service = JudgmentComparisonService()
