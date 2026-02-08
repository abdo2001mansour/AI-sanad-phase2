"""
Chat Prompts

System prompts for different AI personas in chat.
"""

from app.prompts.chat.personas import (
    get_system_prompt_for_persona,
    SANAD_SYSTEM_PROMPT,
    RASHED_SYSTEM_PROMPT,
    NORA_SYSTEM_PROMPT,
    PERSONAS
)

__all__ = [
    'get_system_prompt_for_persona',
    'SANAD_SYSTEM_PROMPT',
    'RASHED_SYSTEM_PROMPT',
    'NORA_SYSTEM_PROMPT',
    'PERSONAS'
]

