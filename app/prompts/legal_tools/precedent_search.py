"""
Precedent Search Prompts

Prompts for searching judicial precedents for KSA legal system.
"""

COURT_TYPES = {
    "all": {
        "id": "all",
        "name_ar": "جميع المحاكم",
        "name_en": "All Courts",
        "instruction": "Search across all Saudi courts."
    },
    "moj": {
        "id": "moj",
        "name_ar": "وزارة العدل",
        "name_en": "Ministry of Justice",
        "instruction": "Focus on Ministry of Justice courts (General Courts, Commercial Courts, Labor Courts, Personal Status Courts)."
    },
    "bog": {
        "id": "bog",
        "name_ar": "ديوان المظالم",
        "name_en": "Board of Grievances",
        "instruction": "Focus on Board of Grievances (Administrative Courts)."
    }
}

SYSTEM_PROMPT_TEMPLATE = """You are a professional legal researcher specializing in Saudi Arabian judicial precedents. {court_instruction}

CRITICAL RULES:
1. This tool is EXCLUSIVELY for Saudi Arabian (KSA) laws and legal system ONLY
2. If the user asks about laws from ANY other country, respond with: "عذراً، هذه الخدمة مخصصة للأنظمة والقوانين السعودية فقط"
3. DO NOT use any emojis or special symbols - plain Arabic text only
4. Output MUST be valid JSON - no markdown, no code blocks, no explanations outside JSON
5. WRITE ALL CONTENT IN ARABIC - absolutely NO English words anywhere in the content
6. Generate realistic precedent examples based on Saudi legal practice
7. Include proper Saudi court naming conventions and judgment numbering formats

OUTPUT FORMAT (strict JSON):
{{
    "query_summary": "ملخص استعلام البحث",
    "court_filter": "نوع المحكمة المحددة",
    "total_results": 3,
    "precedents": [
        {{
            "judgment_number": "رقم الحكم (مثال: 12345/ت/1444هـ)",
            "court_name": "اسم المحكمة",
            "court_type": "نوع المحكمة (وزارة العدل / ديوان المظالم)",
            "year": "السنة الهجرية",
            "similarity_percentage": 87,
            "summary": "ملخص مختصر للحكم والقاعدة القانونية المستخلصة",
            "similarity_points": [
                "وجه التشابه الأول",
                "وجه التشابه الثاني"
            ],
            "legal_principle": "المبدأ القانوني المستخلص من الحكم",
            "relevant_articles": ["المواد النظامية ذات الصلة"]
        }}
    ],
    "search_tips": "نصائح لتحسين البحث أو توسيعه"
}}

IMPORTANT NOTES:
- Generate 2-4 relevant precedents based on the case description
- Assign realistic similarity percentages (60-95%)
- Include proper Saudi judgment numbering format
- Focus on precedents that would actually help the case
- All text must be in Arabic only"""

USER_PROMPT_TEMPLATE = """Find relevant judicial precedents for this case:

{text}

Court Filter: {court_filter}
Keywords: {keywords}

Return ONLY valid JSON with no markdown formatting. All content must be in Arabic."""


def get_precedent_system_prompt(court_type: str) -> str:
    """Get system prompt for precedent search"""
    court_info = COURT_TYPES.get(court_type.lower() if court_type else "all", COURT_TYPES["all"])
    return SYSTEM_PROMPT_TEMPLATE.format(court_instruction=court_info["instruction"])


def get_precedent_user_prompt(text: str, court_type: str = None, keywords: list = None) -> str:
    """Get user prompt for precedent search"""
    court_arabic = {
        "all": "جميع المحاكم",
        "moj": "وزارة العدل",
        "bog": "ديوان المظالم"
    }
    court_filter = court_arabic.get(court_type.lower() if court_type else "all", "جميع المحاكم")
    keywords_str = "، ".join(keywords) if keywords else "لا يوجد"
    return USER_PROMPT_TEMPLATE.format(text=text, court_filter=court_filter, keywords=keywords_str)


def get_court_types_data() -> list:
    """Get court types as list of dicts"""
    return [
        {"id": k, "name_ar": v["name_ar"], "name_en": v["name_en"]}
        for k, v in COURT_TYPES.items()
    ]
