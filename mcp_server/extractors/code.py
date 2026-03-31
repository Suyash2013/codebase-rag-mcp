"""Extractor for source code files."""

from pathlib import Path

from mcp_server.extractors.base import ExtractionResult, ExtractorBase

# Language detection map
_LANG_MAP = {
    ".py": "python", ".pyw": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin", ".kts": "kotlin",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".fish": "shell",
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "scss", ".less": "less",
    ".vue": "vue", ".svelte": "svelte",
    ".sql": "sql",
    ".graphql": "graphql", ".gql": "graphql",
    ".proto": "protobuf",
    ".dockerfile": "dockerfile",
    ".gradle": "gradle",
    ".cmake": "cmake",
    ".makefile": "makefile",
    ".r": "r", ".R": "r",
}

_CODE_EXTENSIONS = set(_LANG_MAP.keys())

_CODE_FILENAMES = {
    "Dockerfile", "Makefile", "CMakeLists.txt", "Jenkinsfile",
    "Procfile", "Vagrantfile", "Gemfile", "Rakefile",
    ".gitignore", ".dockerignore", ".eslintrc", ".prettierrc",
}

_FILENAME_LANG = {
    "Dockerfile": "dockerfile",
    "Makefile": "makefile",
    "CMakeLists.txt": "cmake",
    "Jenkinsfile": "groovy",
    "Gemfile": "ruby",
    "Rakefile": "ruby",
}


class CodeExtractor(ExtractorBase):
    """Extractor for source code files."""

    def supported_extensions(self) -> set[str]:
        return _CODE_EXTENSIONS

    def supported_filenames(self) -> set[str]:
        return _CODE_FILENAMES

    def extract(self, path: Path) -> ExtractionResult:
        text = path.read_text(encoding="utf-8", errors="replace")
        language = _FILENAME_LANG.get(path.name) or _LANG_MAP.get(path.suffix.lower(), "unknown")
        return ExtractionResult(
            text=text,
            content_type="code",
            metadata={"language": language},
        )
