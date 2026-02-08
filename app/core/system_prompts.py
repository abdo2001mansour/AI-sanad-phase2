"""
System prompts for different AI personas

This file now imports from the centralized prompts folder.
Kept for backward compatibility with existing imports.
"""

from app.prompts.chat.personas import (
    get_system_prompt_for_persona as get_system_prompt_for_model,
    SANAD_SYSTEM_PROMPT,
    RASHED_SYSTEM_PROMPT,
    NORA_SYSTEM_PROMPT,
    PERSONAS
)

__all__ = [
    'get_system_prompt_for_model',
    'SANAD_SYSTEM_PROMPT',
    'RASHED_SYSTEM_PROMPT',
    'NORA_SYSTEM_PROMPT',
    'PERSONAS'
]
