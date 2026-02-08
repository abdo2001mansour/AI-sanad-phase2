"""
Summarization Prompts

Prompts for summarizing text.
"""

SUMMARIZATION_TYPES = {
    "brief": {
        "id": "brief",
        "name": "Brief",
        "instruction": "Create a brief summary in 2-4 sentences capturing the main points."
    },
    "detailed": {
        "id": "detailed",
        "name": "Detailed",
        "instruction": "Create a comprehensive summary covering all important points with context."
    }
}

SYSTEM_PROMPT_TEMPLATE = """You are a professional text summarizer.

TASK: {type_instruction}

RULES:
- Output ONLY the summary
- NO introductions like "Here is the summary" or "I will summarize"
- Do NOT introduce yourself
- Preserve the language of the original text
- Keep key terminology intact
- Be concise but comprehensive"""


def get_summarization_system_prompt(summary_type: str = "brief") -> str:
    """Get system prompt for summarization"""
    type_info = SUMMARIZATION_TYPES.get(summary_type.lower(), SUMMARIZATION_TYPES["brief"])
    return SYSTEM_PROMPT_TEMPLATE.format(type_instruction=type_info["instruction"])

