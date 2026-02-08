"""
Legal Article Explanation Prompts

Prompts for explaining and simplifying legal articles and regulations.
"""

EXPLANATION_LEVELS = {
    "simple": {
        "id": "simple",
        "name_ar": "مبسط",
        "name_en": "Simple",
        "instruction": "Explain in SIMPLE, easy-to-understand language suitable for non-lawyers. Avoid legal jargon. Use everyday examples."
    },
    "legal": {
        "id": "legal",
        "name_ar": "قانوني",
        "name_en": "Legal/Technical",
        "instruction": "Explain in TECHNICAL legal language suitable for legal professionals. Include precise legal terminology and comprehensive legal analysis."
    },
    "with_examples": {
        "id": "with_examples",
        "name_ar": "مع أمثلة قضائية",
        "name_en": "With Case Examples",
        "instruction": "Explain with PRACTICAL CASE EXAMPLES. Include hypothetical or actual court cases showing how this article applies in practice."
    }
}

SYSTEM_PROMPT_TEMPLATE = """You are a Saudi legal expert specializing in explaining and simplifying legal texts and regulations. {level_instruction}

CRITICAL RULES:
1. This tool is EXCLUSIVELY for Saudi Arabian (KSA) laws and legal system ONLY
2. If the user asks about laws from ANY other country, respond with: "عذراً، هذه الخدمة مخصصة للأنظمة والقوانين السعودية فقط"
3. DO NOT use any emojis or special symbols - plain Arabic text only
4. Output ONLY the legal explanation - no introductions, no greetings
5. Do NOT introduce yourself or say who you are
6. WRITE ENTIRELY IN ARABIC - absolutely NO English words anywhere
7. All headings, content, terms, and explanations must be in Arabic only
8. Focus on Saudi Arabian legal context and system
9. Reference related Saudi laws and regulations when relevant

EXPLANATION STRUCTURE (all in Arabic):

## النص الأصلي
(Quote the original legal text if provided)

## الشرح
(Clear explanation of the article's meaning and purpose)

## نطاق التطبيق
(When and where this article applies)

## الآثار النظامية
(Legal consequences and implications)

## أمثلة تطبيقية
(Practical examples of application)

## تحذيرات شائعة
(Common mistakes or warnings in applying this article)

## المواد ذات الصلة
(Related articles and regulations)

IMPORTANT:
- No English text allowed anywhere
- Pure Arabic document
- Clear structured explanation
- Practical focus on application"""

USER_PROMPT_TEMPLATE = """Explain this legal article or text:

{text}

Explanation Level: {level_arabic}

Output ONLY the legal explanation in Arabic. Write ENTIRELY in Arabic - no English words at all. Start with the content directly."""


def get_article_system_prompt(explanation_level: str = None) -> str:
    """Get system prompt for article explanation"""
    level_info = EXPLANATION_LEVELS.get(
        explanation_level.lower() if explanation_level else "simple", 
        EXPLANATION_LEVELS["simple"]
    )
    return SYSTEM_PROMPT_TEMPLATE.format(level_instruction=level_info["instruction"])


def get_article_user_prompt(text: str, explanation_level: str = None) -> str:
    """Get user prompt for article explanation"""
    level_arabic = {
        "simple": "مبسط - شرح بسيط لغير المتخصصين",
        "legal": "قانوني - شرح تقني للمتخصصين",
        "with_examples": "مع أمثلة قضائية - شرح مع حالات تطبيقية"
    }
    level = explanation_level.lower() if explanation_level else "simple"
    arabic_level = level_arabic.get(level, level_arabic["simple"])
    
    return USER_PROMPT_TEMPLATE.format(text=text, level_arabic=arabic_level)


def get_explanation_levels_data() -> list:
    """Get explanation levels as list of dicts"""
    return [
        {"id": k, "name_ar": v["name_ar"], "name_en": v["name_en"]}
        for k, v in EXPLANATION_LEVELS.items()
    ]

