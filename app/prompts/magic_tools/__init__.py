"""
Magic Tools Prompts
"""

from app.prompts.magic_tools.translation import (
    get_translation_system_prompt,
    TRANSLATION_TYPES
)

from app.prompts.magic_tools.summarization import (
    get_summarization_system_prompt,
    SUMMARIZATION_TYPES
)

from app.prompts.magic_tools.analysis import (
    get_analysis_system_prompt,
    ANALYSIS_TYPES
)

from app.prompts.magic_tools.rephrasing import (
    get_rephrasing_system_prompt
)

from app.prompts.magic_tools.language_detection import (
    get_language_detection_system_prompt
)

__all__ = [
    'get_translation_system_prompt',
    'TRANSLATION_TYPES',
    'get_summarization_system_prompt',
    'SUMMARIZATION_TYPES',
    'get_analysis_system_prompt',
    'ANALYSIS_TYPES',
    'get_rephrasing_system_prompt',
    'get_language_detection_system_prompt'
]

