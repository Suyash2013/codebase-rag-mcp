import logging
from pathlib import Path

from mcp_server.extractors.base import ExtractionResult, ExtractorBase

log = logging.getLogger("omni-rag")

try:
    import fitz  # PyMuPDF

    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


class PdfExtractor(ExtractorBase):
    """PDF content extractor. Requires: pip install omni-rag-mcp[pdf]"""

    def supported_extensions(self) -> set[str]:
        return {".pdf"}

    def supported_filenames(self) -> set[str]:
        return set()

    def extract(self, path: Path) -> ExtractionResult:
        if not HAS_PYMUPDF:
            msg = f"Skipping {path}: PDF support not installed. Run 'pip install pymupdf'"
            log.warning(msg)
            return ExtractionResult(
                text=msg, content_type="document", metadata={"error": "missing_dependency"}
            )

        try:
            doc = fitz.open(str(path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return ExtractionResult(
                text=text, content_type="document", metadata={"pages": len(doc)}
            )
        except Exception as e:
            log.warning("Failed to extract PDF %s: %s", path, e)
            return ExtractionResult(
                text=str(e), content_type="document", metadata={"error": str(e)}
            )
