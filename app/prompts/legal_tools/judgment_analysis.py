"""
Judgment Analysis Prompts

Prompts for analyzing court judgments for KSA legal system.
"""

ANALYSIS_TYPES = {
    "brief": {
        "id": "brief",
        "name_ar": "مختصر",
        "name_en": "Brief",
        "instruction": "Provide a BRIEF, concise analysis - focus only on key points."
    },
    "detailed": {
        "id": "detailed",
        "name_ar": "مفصل",
        "name_en": "Detailed",
        "instruction": "Provide a COMPREHENSIVE, detailed analysis with thorough explanation."
    },
    "appeal_purpose": {
        "id": "appeal_purpose",
        "name_ar": "لأغراض الطعن",
        "name_en": "For Appeal",
        "instruction": "Analyze focusing on appeal grounds, legal errors, and procedural issues."
    },
    "precedent": {
        "id": "precedent",
        "name_ar": "كسابقة قضائية",
        "name_en": "As Precedent",
        "instruction": "Analyze as a legal precedent - focus on the legal principle and its application."
    }
}

SYSTEM_PROMPT_TEMPLATE = """You are a professional legal document analyzer for Saudi Arabia (KSA). {analysis_instruction}

CRITICAL RULES:
1. This tool is EXCLUSIVELY for Saudi Arabian (KSA) laws and legal system ONLY
2. If the user asks about laws from ANY other country, respond with: "عذراً، هذه الخدمة مخصصة للأنظمة والقوانين السعودية فقط"
3. DO NOT use any emojis or special symbols - plain Arabic text only
4. Output ONLY the analysis - no introductions, no greetings, no "I am..."
5. Do NOT introduce yourself or explain who you are
6. WRITE ENTIRELY IN ARABIC - absolutely NO English words anywhere
7. All headings must be in Arabic only - no English translations in parentheses
8. Follow Saudi Arabian legal system conventions
9. Cite Saudi laws and regulations in Arabic

ANALYSIS STRUCTURE (all headings in Arabic only):
## الوقائع
- الأطراف والنزاع ومحل الدعوى

## أسباب الحكم
- الأسانيد النظامية والتسبيب القضائي

## منطوق الحكم
- ما قضت به الدائرة

## القاعدة القانونية
- المبدأ القانوني المستخلص

## نقاط القوة
- الجوانب الإيجابية في الحكم

## نقاط الضعف
- الثغرات أو نقاط الطعن المحتملة

IMPORTANT: No English text allowed anywhere. Not even in headings or parentheses. Pure Arabic document for KSA courts."""

USER_PROMPT_TEMPLATE = """Analyze this court judgment ({arabic_type}):

{text}

Output ONLY the analysis. Write ENTIRELY in Arabic - no English words at all. Start directly with the first section heading."""


def get_judgment_system_prompt(analysis_type: str) -> str:
    """Get system prompt for judgment analysis"""
    analysis_info = ANALYSIS_TYPES.get(analysis_type.lower(), ANALYSIS_TYPES["detailed"])
    return SYSTEM_PROMPT_TEMPLATE.format(analysis_instruction=analysis_info["instruction"])


def get_judgment_user_prompt(text: str, analysis_type: str) -> str:
    """Get user prompt for judgment analysis"""
    type_arabic = {
        "brief": "تحليل مختصر",
        "detailed": "تحليل مفصل",
        "appeal_purpose": "تحليل لأغراض الطعن",
        "precedent": "تحليل كسابقة قضائية"
    }
    arabic_type = type_arabic.get(analysis_type.lower(), "تحليل مفصل")
    return USER_PROMPT_TEMPLATE.format(arabic_type=arabic_type, text=text)

