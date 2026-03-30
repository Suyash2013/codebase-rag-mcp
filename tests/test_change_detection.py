"""Tests for change detection abstraction."""

import json
import os
from pathlib import Path
from mcp_server.change_detection.base import ChangeReport
from mcp_server.change_detection.file_hash_detector import FileHashDetector


class TestFileHashDetector:
    def test_no_checkpoint_means_no_changes_detected(self, tmp_path):
        detector = FileHashDetector()
        assert not detector.has_checkpoint(str(tmp_path))

    def test_save_and_detect_no_changes(self, tmp_path):
        (tmp_path / "file.txt").write_text("hello")
        detector = FileHashDetector()
        detector.save_checkpoint(str(tmp_path))
        assert detector.has_checkpoint(str(tmp_path))
        report = detector.detect_changes(str(tmp_path))
        assert not report.has_changes
        assert report.changed_files == []
        assert report.deleted_files == []

    def test_detect_modified_file(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        detector = FileHashDetector()
        detector.save_checkpoint(str(tmp_path))
        f.write_text("world")
        report = detector.detect_changes(str(tmp_path))
        assert report.has_changes
        assert "file.txt" in report.changed_files

    def test_detect_new_file(self, tmp_path):
        (tmp_path / "old.txt").write_text("old")
        detector = FileHashDetector()
        detector.save_checkpoint(str(tmp_path))
        (tmp_path / "new.txt").write_text("new")
        report = detector.detect_changes(str(tmp_path))
        assert report.has_changes
        assert "new.txt" in report.changed_files

    def test_detect_deleted_file(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        detector = FileHashDetector()
        detector.save_checkpoint(str(tmp_path))
        f.unlink()
        report = detector.detect_changes(str(tmp_path))
        assert report.has_changes
        assert "file.txt" in report.deleted_files
