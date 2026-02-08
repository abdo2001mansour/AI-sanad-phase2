"""
Analysis Prompts

Prompts for analyzing text.
"""

ANALYSIS_TYPES = {
    "brief": {
        "id": "brief",
        "name": "Brief",
        "instruction": "Provide a brief analysis highlighting the key points and main arguments."
    },
    "detailed": {
        "id": "detailed",
        "name": "Detailed",
        "instruction": "Provide a comprehensive analysis examining all aspects, arguments, strengths, and weaknesses."
    }
}

SYSTEM_PROMPT_TEMPLATE = """You are a professional text analyst.

TASK: {type_instruction}

RULES:
- Output ONLY the analysis
- NO introductions like "Here is my analysis" or "I will analyze"
- Do NOT introduce yourself
- Preserve the language of the original text
- Use clear structure with headings if needed
- Be objective and thorough"""


def get_analysis_system_prompt(analysis_type: str = "brief") -> str:
    """Get system prompt for analysis"""
    type_info = ANALYSIS_TYPES.get(analysis_type.lower(), ANALYSIS_TYPES["brief"])
    return SYSTEM_PROMPT_TEMPLATE.format(type_instruction=type_info["instruction"])

