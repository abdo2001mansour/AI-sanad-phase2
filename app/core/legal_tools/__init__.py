"""
Legal Tools Services

This module contains AI-powered legal tools services.
"""

from app.core.legal_tools.legal_memo_generator_service import legal_memo_generator_service
from app.core.legal_tools.judgment_analysis_service import judgment_analysis_service
from app.core.legal_tools.lawsuit_petition_service import lawsuit_petition_service
from app.core.legal_tools.article_explanation_service import article_explanation_service

__all__ = [
    "legal_memo_generator_service",
    "judgment_analysis_service",
    "lawsuit_petition_service",
    "article_explanation_service"
]
