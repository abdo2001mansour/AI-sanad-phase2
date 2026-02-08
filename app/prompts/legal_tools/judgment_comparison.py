"""
Judgment Comparison Prompts

Prompts for comparing court judgments for KSA legal system.
"""

COMPARISON_CRITERIA = {
    "all": {
        "id": "all",
        "name_ar": "الكل",
        "name_en": "All",
        "instruction": "Compare ALL aspects: facts, reasoning, ruling, and legal principles."
    },
    "facts": {
        "id": "facts",
        "name_ar": "الوقائع",
        "name_en": "Facts",
        "instruction": "Focus comparison on the FACTS of each case only."
    },
    "reasoning": {
        "id": "reasoning",
        "name_ar": "التسبيب",
        "name_en": "Reasoning",
        "instruction": "Focus comparison on the LEGAL REASONING and justifications only."
    },
    "ruling": {
        "id": "ruling",
        "name_ar": "المنطوق",
        "name_en": "Ruling",
        "instruction": "Focus comparison on the FINAL RULING and decisions only."
    },
    "legal_principle": {
        "id": "legal_principle",
        "name_ar": "القاعدة القانونية",
        "name_en": "Legal Principle",
        "instruction": "Focus comparison on the LEGAL PRINCIPLES applied in each case."
    }
}

SYSTEM_PROMPT_TEMPLATE = """You are a professional legal analyst specializing in comparing Saudi Arabian court judgments. {comparison_instruction}

CRITICAL RULES:
1. This tool is EXCLUSIVELY for Saudi Arabian (KSA) laws and legal system ONLY
2. If the user asks about laws from ANY other country, respond with: "عذراً، هذه الخدمة مخصصة للأنظمة والقوانين السعودية فقط"
3. Output MUST be valid JSON - no markdown, no code blocks, no explanations outside JSON
4. WRITE ALL CONTENT IN ARABIC - absolutely NO English words anywhere in the content
5. DO NOT use any emojis or special symbols - plain Arabic text only
6. Follow Saudi Arabian legal system conventions
7. Cite Saudi laws and regulations in Arabic
8. Be precise and analytical in your comparison

OUTPUT FORMAT (strict JSON):
{{
    "comparison_table": [
        {{
            "aspect": "الوقائع",
            "judgment_1": "وصف تفصيلي للوقائع في الحكم الأول",
            "judgment_2": "وصف تفصيلي للوقائع في الحكم الثاني",
            "judgment_3": "وصف الوقائع في الحكم الثالث (إن وجد)",
            "notes": "أبرز الفروق والملاحظات بين الأحكام"
        }},
        {{
            "aspect": "التسبيب",
            "judgment_1": "شرح الأسباب القانونية في الحكم الأول",
            "judgment_2": "شرح التسبيب في الحكم الثاني",
            "judgment_3": "التسبيب في الحكم الثالث (إن وجد)",
            "notes": "نقاط القوة أو الضعف في التسبيب، التباين أو التوافق"
        }},
        {{
            "aspect": "المنطوق",
            "judgment_1": "نص الحكم أو القرار النهائي",
            "judgment_2": "نص الحكم الثاني",
            "judgment_3": "المنطوق الثالث (إن وجد)",
            "notes": "أي اختلاف في القرار، نسبة تطبيق القانون، العقوبة أو التعويض"
        }},
        {{
            "aspect": "القاعدة القانونية",
            "judgment_1": "المبادئ القانونية المستند إليها",
            "judgment_2": "المبادئ في الحكم الثاني",
            "judgment_3": "المبادئ في الحكم الثالث (إن وجد)",
            "notes": "تباين أو اتفاق المبادئ، مدى اتساقها مع الاجتهادات السابقة"
        }}
    ],
    "legal_conclusion": "خلاصة تحليلية شاملة تتضمن: اتجاهات قضائية، نقاط اتفاق واختلاف، تأثير الوقائع والتسبيب على النتائج",
    "usage_recommendation": "توصيات عملية لاستخدام نتائج المقارنة في إعداد المذكرات، تقديم الاستشارات، أو بناء حجة قانونية"
}}

IMPORTANT NOTES:
- If only 2 judgments provided, set judgment_3 to null or empty string
- Each aspect row must have meaningful analysis
- legal_conclusion should identify patterns and trends
- usage_recommendation should provide actionable advice for lawyers
- All text must be in Arabic only"""

USER_PROMPT_TEMPLATE = """Compare the following court judgments ({arabic_criteria}):

{text}

Analyze and compare these judgments. Return ONLY valid JSON with no markdown formatting. All content must be in Arabic."""


def get_comparison_system_prompt(comparison_criteria: str) -> str:
    """Get system prompt for judgment comparison"""
    criteria_info = COMPARISON_CRITERIA.get(comparison_criteria.lower(), COMPARISON_CRITERIA["all"])
    return SYSTEM_PROMPT_TEMPLATE.format(comparison_instruction=criteria_info["instruction"])


def get_comparison_user_prompt(text: str, comparison_criteria: str) -> str:
    """Get user prompt for judgment comparison"""
    criteria_arabic = {
        "all": "مقارنة شاملة",
        "facts": "مقارنة الوقائع",
        "reasoning": "مقارنة التسبيب",
        "ruling": "مقارنة المنطوق",
        "legal_principle": "مقارنة القاعدة القانونية"
    }
    arabic_criteria = criteria_arabic.get(comparison_criteria.lower(), "مقارنة شاملة")
    return USER_PROMPT_TEMPLATE.format(arabic_criteria=arabic_criteria, text=text)
