"""Auto-registers built-in extractors."""

from mcp_server.extractors.base import (
    ExtractionResult,
    ExtractorBase,
    get_extractor,
    register_extractor,
)
from mcp_server.extractors.code import CodeExtractor
from mcp_server.extractors.docx import DocxExtractor
from mcp_server.extractors.image import ImageExtractor
from mcp_server.extractors.markdown import MarkdownExtractor
from mcp_server.extractors.pdf import PdfExtractor
from mcp_server.extractors.structured_data import StructuredDataExtractor
from mcp_server.extractors.text import TextExtractor

# Register built-in extractors (order matters: last registered wins for overlapping extensions)
_text = TextExtractor()
_code = CodeExtractor()
_markdown = MarkdownExtractor()
_structured = StructuredDataExtractor()
_pdf = PdfExtractor()
_docx = DocxExtractor()
_image = ImageExtractor()

register_extractor(_text, is_fallback=True)
register_extractor(_code)
register_extractor(_markdown)
register_extractor(_structured)
register_extractor(_pdf)
register_extractor(_docx)
register_extractor(_image)

__all__ = [
    "CodeExtractor",
    "DocxExtractor",
    "ExtractionResult",
    "ExtractorBase",
    "ImageExtractor",
    "MarkdownExtractor",
    "PdfExtractor",
    "StructuredDataExtractor",
    "TextExtractor",
    "get_extractor",
    "register_extractor",
]
