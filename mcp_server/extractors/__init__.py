"""Auto-registers built-in extractors."""

from mcp_server.extractors.base import (
    ExtractionResult,
    ExtractorBase,
    get_extractor,
    register_extractor,
)
from mcp_server.extractors.text import TextExtractor
from mcp_server.extractors.code import CodeExtractor

# Register built-in extractors (order matters: last registered wins for overlapping extensions)
_text = TextExtractor()
_code = CodeExtractor()
register_extractor(_text, is_fallback=True)
register_extractor(_code)

__all__ = [
    "ExtractionResult", "ExtractorBase", "get_extractor", "register_extractor",
    "TextExtractor", "CodeExtractor",
]
