import logging
from pathlib import Path

from mcp_server.extractors.base import ExtractionResult, ExtractorBase

log = logging.getLogger("omni-rag")

try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


class ImageExtractor(ExtractorBase):
    """Image content extractor using OCR (Tesseract). Requires: pip install pytesseract Pillow"""

    def supported_extensions(self) -> set[str]:
        return {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def supported_filenames(self) -> set[str]:
        return set()

    def extract(self, path: Path) -> ExtractionResult:
        if not HAS_OCR:
            msg = f"Skipping {path}: OCR support not installed. Run 'pip install pytesseract Pillow'"
            log.warning(msg)
            return ExtractionResult(text=msg, content_type="document", metadata={"error": "missing_dependency"})

        try:
            img = Image.open(str(path))
            text = pytesseract.image_to_string(img)
            return ExtractionResult(text=text, content_type="document", metadata={"size": img.size, "mode": img.mode})
        except Exception as e:
            log.warning("Failed to extract image OCR %s: %s", path, e)
            return ExtractionResult(text=str(e), content_type="document", metadata={"error": str(e)})
