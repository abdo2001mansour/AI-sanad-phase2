"""
Case Evaluation Service

This service evaluates legal cases using OpenAI.
Returns structured JSON with assessment, strengths, weaknesses, and recommendations.
"""
import os
import json
from typing import Optional
from openai import AsyncOpenAI
from app.config.settings import settings
from app.prompts.legal_tools.case_evaluation import (
    get_evaluation_system_prompt,
    get_evaluation_user_prompt,
    get_case_types_data,
    get_evaluation_purposes_data
)


class CaseEvaluationService:
    """Service for evaluating legal cases using OpenAI"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = os.getenv("LEGAL_TOOLS_MODEL", "gpt-4.1-2025-04-14")
    
    async def evaluate_case(
        self,
        text: str,
        case_type: Optional[str] = None,
        evaluation_purpose: Optional[str] = None,
        file_content: Optional[str] = None
    ) -> dict:
        """
        Evaluate legal case and return structured JSON response
        """
        if not case_type:
            case_type = "general"
        if not evaluation_purpose:
            evaluation_purpose = "pre_filing"
        
        system_prompt = get_evaluation_system_prompt(case_type, evaluation_purpose)
        
        full_text = text
        if file_content:
            full_text = f"{text}\n\n{file_content}"
        
        user_prompt = get_evaluation_user_prompt(full_text, case_type, evaluation_purpose)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_completion_tokens=6000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = {
                    "case_type": case_type,
                    "evaluation_purpose": evaluation_purpose,
                    "overall_assessment": {"status": "غير محدد", "status_code": "unknown"},
                    "success_probability": {"percentage": 0},
                    "strengths": [],
                    "weaknesses": [],
                    "potential_risks": [],
                    "strategic_recommendations": [],
                    "error": "فشل في تحليل الاستجابة"
                }
            
            return result
                        
        except Exception as e:
            return {
                "case_type": case_type,
                "evaluation_purpose": evaluation_purpose,
                "overall_assessment": {"status": "خطأ", "status_code": "error"},
                "success_probability": {"percentage": 0},
                "strengths": [],
                "weaknesses": [],
                "potential_risks": [],
                "strategic_recommendations": [],
                "error": str(e)
            }
    
    def get_case_types(self) -> list:
        """Get available case types"""
        return get_case_types_data()
    
    def get_evaluation_purposes(self) -> list:
        """Get available evaluation purposes"""
        return get_evaluation_purposes_data()


# Singleton instance
case_evaluation_service = CaseEvaluationService()
