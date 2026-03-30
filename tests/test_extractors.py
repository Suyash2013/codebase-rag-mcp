"""Tests for extractor base, TextExtractor, and CodeExtractor."""

import os
from pathlib import Path
from mcp_server.extractors.base import ExtractionResult
from mcp_server.extractors.text import TextExtractor
from mcp_server.extractors.code import CodeExtractor


class TestTextExtractor:
    def setup_method(self):
        self.extractor = TextExtractor()

    def test_supported_extensions(self):
        exts = self.extractor.supported_extensions()
        assert ".txt" in exts
        assert ".log" in exts

    def test_extract_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        result = self.extractor.extract(f)
        assert isinstance(result, ExtractionResult)
        assert result.text == "hello world"
        assert result.content_type == "plain_text"

    def test_extract_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = self.extractor.extract(f)
        assert result.text == ""
        assert result.content_type == "plain_text"


class TestCodeExtractor:
    def setup_method(self):
        self.extractor = CodeExtractor()

    def test_supported_extensions(self):
        exts = self.extractor.supported_extensions()
        assert ".py" in exts
        assert ".js" in exts
        assert ".ts" in exts
        assert ".go" in exts

    def test_supported_filenames(self):
        names = self.extractor.supported_filenames()
        assert "Dockerfile" in names
        assert "Makefile" in names

    def test_extract_python_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def hello():\n    pass\n", encoding="utf-8")
        result = self.extractor.extract(f)
        assert result.content_type == "code"
        assert "def hello" in result.text
        assert result.metadata.get("language") == "python"

    def test_extract_js_file(self, tmp_path):
        f = tmp_path / "test.js"
        f.write_text("function hello() {}", encoding="utf-8")
        result = self.extractor.extract(f)
        assert result.content_type == "code"
        assert result.metadata.get("language") == "javascript"
