"""Tests for change detection abstraction."""

import subprocess

from mcp_server.change_detection.file_hash_detector import FileHashDetector
from mcp_server.change_detection.git_detector import GitDetector


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


class TestGitDetector:
    def _init_repo(self, path):
        subprocess.run(["git", "init"], cwd=path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=path, capture_output=True)
        (path / "file.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True)

    def test_no_checkpoint_means_changes_detected(self, tmp_path):
        self._init_repo(tmp_path)
        detector = GitDetector()
        assert not detector.has_checkpoint(str(tmp_path))
        report = detector.detect_changes(str(tmp_path))
        assert report.has_changes
        assert "No prior index" in report.details

    def test_save_and_detect_no_changes(self, tmp_path):
        self._init_repo(tmp_path)
        detector = GitDetector()
        detector.save_checkpoint(str(tmp_path))
        assert detector.has_checkpoint(str(tmp_path))
        report = detector.detect_changes(str(tmp_path))
        assert not report.has_changes

    def test_detect_uncommitted_changes(self, tmp_path):
        self._init_repo(tmp_path)
        detector = GitDetector()
        detector.save_checkpoint(str(tmp_path))
        (tmp_path / "new.py").write_text("print(1)")
        report = detector.detect_changes(str(tmp_path))
        assert report.has_changes
        assert "new.py" in report.changed_files

    def test_detect_committed_changes(self, tmp_path):
        self._init_repo(tmp_path)
        detector = GitDetector()
        detector.save_checkpoint(str(tmp_path))

        (tmp_path / "other.txt").write_text("other")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "next"], cwd=tmp_path, capture_output=True)

        report = detector.detect_changes(str(tmp_path))
        assert report.has_changes
        assert "other.txt" in report.changed_files

    def test_detect_deleted_file(self, tmp_path):
        self._init_repo(tmp_path)
        detector = GitDetector()
        detector.save_checkpoint(str(tmp_path))
        (tmp_path / "file.txt").unlink()
        report = detector.detect_changes(str(tmp_path))
        assert report.has_changes
        assert "file.txt" in report.deleted_files
