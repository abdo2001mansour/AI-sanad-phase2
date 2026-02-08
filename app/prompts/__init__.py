"""
Prompts Module

Universal prompts folder containing all AI prompts organized by service.

Structure:
- chat/        - Chat personas (Sanad, Rashed, Nora)
- legal_tools/ - Legal document generation prompts
- magic_tools/ - Magic tools prompts (translation, summarization, etc.)
"""

from app.prompts import chat
from app.prompts import legal_tools
from app.prompts import magic_tools

__all__ = ['chat', 'legal_tools', 'magic_tools']
