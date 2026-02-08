"""
Case Evaluation Prompts

Prompts for evaluating legal cases for KSA legal system.
"""

CASE_TYPES = {
    "commercial": {
        "id": "commercial",
        "name_ar": "تجارية",
        "name_en": "Commercial",
        "instruction": "Evaluate as a COMMERCIAL case before the Commercial Court."
    },
    "labor": {
        "id": "labor",
        "name_ar": "عمالية",
        "name_en": "Labor",
        "instruction": "Evaluate as a LABOR case before the Labor Court."
    },
    "administrative": {
        "id": "administrative",
        "name_ar": "إدارية",
        "name_en": "Administrative",
        "instruction": "Evaluate as an ADMINISTRATIVE case before the Board of Grievances."
    },
    "general": {
        "id": "general",
        "name_ar": "عامة",
        "name_en": "General",
        "instruction": "Evaluate as a GENERAL case before the General Court."
    }
}

EVALUATION_PURPOSES = {
    "pre_filing": {
        "id": "pre_filing",
        "name_ar": "قبل رفع الدعوى",
        "name_en": "Before Filing",
        "instruction": "Evaluate to decide whether to file the case."
    },
    "during_litigation": {
        "id": "during_litigation",
        "name_ar": "أثناء نظر الدعوى",
        "name_en": "During Litigation",
        "instruction": "Evaluate current position during ongoing litigation."
    },
    "pre_appeal": {
        "id": "pre_appeal",
        "name_ar": "قبل الاستئناف",
        "name_en": "Before Appeal",
        "instruction": "Evaluate whether to appeal the judgment."
    },
    "strategy_review": {
        "id": "strategy_review",
        "name_ar": "مراجعة استراتيجية",
        "name_en": "Strategy Review",
        "instruction": "Review and adjust legal strategy."
    }
}

SYSTEM_PROMPT_TEMPLATE = """You are a professional legal analyst specializing in evaluating Saudi Arabian legal cases. {case_instruction} {purpose_instruction}

CRITICAL RULES:
1. This tool is EXCLUSIVELY for Saudi Arabian (KSA) laws and legal system ONLY
2. If the user asks about laws from ANY other country, respond with: "عذراً، هذه الخدمة مخصصة للأنظمة والقوانين السعودية فقط"
3. DO NOT use any emojis or special symbols - plain Arabic text only
4. Output MUST be valid JSON - no markdown, no code blocks, no explanations outside JSON
5. WRITE ALL CONTENT IN ARABIC - absolutely NO English words anywhere in the content
6. Be objective and balanced in your assessment
7. Provide actionable recommendations

OUTPUT FORMAT (strict JSON):
{{
    "case_type": "نوع القضية",
    "evaluation_purpose": "غرض التقييم",
    "overall_assessment": {{
        "status": "موقف جيد / موقف متوسط / موقف ضعيف",
        "status_code": "good / medium / weak",
        "summary": "ملخص التقييم العام"
    }},
    "success_probability": {{
        "percentage": 68,
        "explanation": "شرح تقدير نسبة النجاح"
    }},
    "strengths": [
        "نقطة قوة أولى",
        "نقطة قوة ثانية",
        "نقطة قوة ثالثة"
    ],
    "weaknesses": [
        "نقطة ضعف أولى",
        "نقطة ضعف ثانية"
    ],
    "potential_risks": [
        "مخاطرة محتملة أولى",
        "مخاطرة محتملة ثانية"
    ],
    "strategic_recommendations": [
        "توصية استراتيجية أولى",
        "توصية استراتيجية ثانية",
        "توصية استراتيجية ثالثة"
    ],
    "key_evidence_needed": [
        "دليل مطلوب لتقوية الموقف"
    ],
    "similar_cases_outcome": "ملخص لاتجاه القضاء في قضايا مماثلة"
}}

IMPORTANT NOTES:
- Success probability should be realistic (0-100%)
- Provide at least 3 strengths and 2 weaknesses
- Recommendations should be specific and actionable
- Consider Saudi legal practice and court tendencies
- All text must be in Arabic only"""

USER_PROMPT_TEMPLATE = """Evaluate this legal case:

{text}

Case Type: {case_type}
Evaluation Purpose: {evaluation_purpose}

Return ONLY valid JSON with no markdown formatting. All content must be in Arabic."""


def get_evaluation_system_prompt(case_type: str = None, evaluation_purpose: str = None) -> str:
    """Get system prompt for case evaluation"""
    case_info = CASE_TYPES.get(case_type.lower() if case_type else "general", CASE_TYPES["general"])
    purpose_info = EVALUATION_PURPOSES.get(evaluation_purpose.lower() if evaluation_purpose else "pre_filing", EVALUATION_PURPOSES["pre_filing"])
    return SYSTEM_PROMPT_TEMPLATE.format(
        case_instruction=case_info["instruction"],
        purpose_instruction=purpose_info["instruction"]
    )


def get_evaluation_user_prompt(text: str, case_type: str = None, evaluation_purpose: str = None) -> str:
    """Get user prompt for case evaluation"""
    case_arabic = {
        "commercial": "تجارية",
        "labor": "عمالية",
        "administrative": "إدارية",
        "general": "عامة"
    }
    purpose_arabic = {
        "pre_filing": "قبل رفع الدعوى",
        "during_litigation": "أثناء نظر الدعوى",
        "pre_appeal": "قبل الاستئناف",
        "strategy_review": "مراجعة استراتيجية"
    }
    
    case_type_ar = case_arabic.get(case_type.lower() if case_type else "general", "عامة")
    purpose_ar = purpose_arabic.get(evaluation_purpose.lower() if evaluation_purpose else "pre_filing", "قبل رفع الدعوى")
    
    return USER_PROMPT_TEMPLATE.format(text=text, case_type=case_type_ar, evaluation_purpose=purpose_ar)


def get_case_types_data() -> list:
    """Get case types as list of dicts"""
    return [
        {"id": k, "name_ar": v["name_ar"], "name_en": v["name_en"]}
        for k, v in CASE_TYPES.items()
    ]


def get_evaluation_purposes_data() -> list:
    """Get evaluation purposes as list of dicts"""
    return [
        {"id": k, "name_ar": v["name_ar"], "name_en": v["name_en"]}
        for k, v in EVALUATION_PURPOSES.items()
    ]
