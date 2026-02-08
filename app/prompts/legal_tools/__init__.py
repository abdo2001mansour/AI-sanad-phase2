"""
Legal Tools Prompts
"""

from app.prompts.legal_tools.memo_generator import (
    get_memo_system_prompt,
    get_memo_user_prompt,
    MEMO_TYPES
)

from app.prompts.legal_tools.judgment_analysis import (
    get_judgment_system_prompt,
    get_judgment_user_prompt,
    ANALYSIS_TYPES
)

from app.prompts.legal_tools.lawsuit_petition import (
    get_petition_system_prompt,
    get_petition_user_prompt,
    get_case_types_data,
    get_courts_data,
    CASE_TYPES,
    COURTS
)

from app.prompts.legal_tools.article_explanation import (
    get_article_system_prompt,
    get_article_user_prompt,
    get_explanation_levels_data,
    EXPLANATION_LEVELS
)

__all__ = [
    'get_memo_system_prompt',
    'get_memo_user_prompt',
    'MEMO_TYPES',
    'get_judgment_system_prompt',
    'get_judgment_user_prompt',
    'ANALYSIS_TYPES',
    'get_petition_system_prompt',
    'get_petition_user_prompt',
    'get_case_types_data',
    'get_courts_data',
    'CASE_TYPES',
    'COURTS',
    'get_article_system_prompt',
    'get_article_user_prompt',
    'get_explanation_levels_data',
    'EXPLANATION_LEVELS'
]
