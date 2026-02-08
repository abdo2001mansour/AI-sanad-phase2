"""
Legal Summary Prompts

Prompts for summarizing legal documents for KSA legal system.
"""

SUMMARY_TYPES = {
    "executive": {
        "id": "executive",
        "name_ar": "ملخص تنفيذي",
        "name_en": "Executive Summary",
        "instruction": "Provide an EXECUTIVE SUMMARY - a quick, organized summary of the main topic and key legal points. Suitable for quick decision-making without going into details."
    },
    "key_points": {
        "id": "key_points",
        "name_ar": "نقاط رئيسية",
        "name_en": "Key Points",
        "instruction": "Extract KEY POINTS - present the most important points, facts, and legal articles in clear bullet points for quick review and preparation."
    },
    "for_memo": {
        "id": "for_memo",
        "name_ar": "لاستخدامه في مذكرة",
        "name_en": "For Memo Use",
        "instruction": "Generate a summary with FORMAL LEGAL WORDING suitable for direct insertion into legal memos and pleadings."
    }
}

SYSTEM_PROMPT_TEMPLATE = """You are a professional legal document summarizer for Saudi Arabia (KSA). {summary_instruction}

CRITICAL RULES:
1. This tool is EXCLUSIVELY for Saudi Arabian (KSA) laws and legal system ONLY
2. If the user asks about laws from ANY other country, respond with: "عذراً، هذه الخدمة مخصصة للأنظمة والقوانين السعودية فقط"
3. DO NOT use any emojis or special symbols - plain Arabic text only
4. Output MUST be valid JSON - no markdown, no code blocks, no explanations outside JSON
5. WRITE ALL CONTENT IN ARABIC - absolutely NO English words anywhere in the content
6. Follow Saudi Arabian legal system conventions
7. Cite Saudi laws and regulations in Arabic when relevant
8. Be concise but comprehensive

OUTPUT FORMAT (strict JSON):
{{
    "summary_type": "نوع الملخص",
    "document_title": "عنوان أو وصف المستند (إن وجد)",
    "main_summary": "الملخص الرئيسي للمستند",
    "key_points": [
        "النقطة الأولى",
        "النقطة الثانية",
        "النقطة الثالثة"
    ],
    "legal_references": [
        "المادة أو النظام المذكور"
    ],
    "parties_involved": "الأطراف المعنية (إن وجدت)",
    "dates_mentioned": ["التواريخ المهمة"],
    "amounts_mentioned": ["المبالغ المالية إن وجدت"],
    "recommendations": "توصيات أو ملاحظات للقارئ"
}}

IMPORTANT NOTES:
- Preserve all important legal information
- Maintain accuracy of dates, numbers, and party names
- Highlight legal implications and consequences
- All text must be in Arabic only"""

USER_PROMPT_TEMPLATE = """Summarize this legal document ({arabic_type}):

{text}

Return ONLY valid JSON with no markdown formatting. All content must be in Arabic."""


def get_summary_system_prompt(summary_type: str) -> str:
    """Get system prompt for legal summary"""
    type_info = SUMMARY_TYPES.get(summary_type.lower(), SUMMARY_TYPES["executive"])
    return SYSTEM_PROMPT_TEMPLATE.format(summary_instruction=type_info["instruction"])


def get_summary_user_prompt(text: str, summary_type: str) -> str:
    """Get user prompt for legal summary"""
    type_arabic = {
        "executive": "ملخص تنفيذي",
        "key_points": "نقاط رئيسية",
        "for_memo": "لاستخدامه في مذكرة"
    }
    arabic_type = type_arabic.get(summary_type.lower(), "ملخص تنفيذي")
    return USER_PROMPT_TEMPLATE.format(arabic_type=arabic_type, text=text)


def get_summary_types_data() -> list:
    """Get summary types as list of dicts"""
    return [
        {"id": k, "name_ar": v["name_ar"], "name_en": v["name_en"]}
        for k, v in SUMMARY_TYPES.items()
    ]
