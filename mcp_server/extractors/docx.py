import logging
from pathlib import Path
from mcp_server.extractors.base import ExtractorBase, ExtractionResult

log = logging.getLogger("rag-mcp")

try:
    import docx
    HAS_PYTHON_DOCX = True
except ImportError:
    HAS_PYTHON_DOCX = False


class DocxExtractor(ExtractorBase):
    """Word document extractor. Requires: pip install python-docx"""

    def supported_extensions(self) -> set[str]:
        return {".docx"}

    def supported_filenames(self) -> set[str]:
        return set()

    def extract(self, path: Path) -> ExtractionResult:
        if not HAS_PYTHON_DOCX:
            msg = f"Skipping {path}: DOCX support not installed. Run 'pip install python-docx'"
            log.warning(msg)
            return ExtractionResult(text=msg, content_type="document", metadata={"error": "missing_dependency"})

        try:
            doc = docx.Document(str(path))
            text = "\n".join([p.text for p in doc.paragraphs])
            return ExtractionResult(text=text, content_type="document")
        except Exception as e:
            log.warning("Failed to extract DOCX %s: %s", path, e)
            return ExtractionResult(text=str(e), content_type="document", metadata={"error": str(e)})
