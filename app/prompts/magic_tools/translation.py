"""
Translation Prompts

Prompts for translating text between languages.
"""

TRANSLATION_TYPES = {
    "general": {
        "id": "general",
        "name": "General",
        "instruction": """- Use natural, fluent language
- Adapt idioms appropriately
- Maintain original tone and style"""
    },
    "legal": {
        "id": "legal",
        "name": "Legal",
        "instruction": """- Use precise legal terminology
- Maintain formal legal tone
- Preserve legal references and citations exactly"""
    },
    "academic": {
        "id": "academic",
        "name": "Academic",
        "instruction": """- Use formal academic language
- Preserve technical terms accurately
- Keep citations in original form"""
    }
}

SYSTEM_PROMPT_TEMPLATE = """You are a professional translator. Translate text to {language_name}.

RULES:
- Output ONLY the translated text
- NO explanations, introductions, or commentary
- Preserve original formatting, line breaks, and structure
- Keep markdown formatting intact
{type_instruction}"""


def get_translation_system_prompt(target_language: str, translation_type: str = "general") -> str:
    """Get system prompt for translation"""
    type_info = TRANSLATION_TYPES.get(translation_type.lower(), TRANSLATION_TYPES["general"])
    return SYSTEM_PROMPT_TEMPLATE.format(
        language_name=target_language,
        type_instruction=type_info["instruction"]
    )

