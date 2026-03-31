"""Tests for CLI entry points."""

import json
from pathlib import Path
from unittest.mock import patch

from mcp_server.cli import _find_claude_config, setup


def test_find_claude_config_returns_path():
    """Should return a Path object."""
    result = _find_claude_config()
    assert isinstance(result, Path)


def test_setup_already_registered(tmp_path, capsys):
    """Should skip if already registered."""
    config_file = tmp_path / ".claude.json"
    config_file.write_text(json.dumps({"mcpServers": {"omni-rag": {"command": "omni-rag"}}}))

    with patch("mcp_server.cli._find_claude_config", return_value=config_file):
        setup()

    output = capsys.readouterr().out
    assert "already registered" in output


def test_setup_registers_new(tmp_path, capsys):
    """Should register when not present."""
    config_file = tmp_path / ".claude.json"
    config_file.write_text(json.dumps({"mcpServers": {}}))

    with (
        patch("mcp_server.cli._find_claude_config", return_value=config_file),
        patch("shutil.which", return_value="/usr/bin/omni-rag"),
    ):
        setup()

    output = capsys.readouterr().out
    assert "Registered" in output

    # Verify config was written
    config = json.loads(config_file.read_text())
    assert "omni-rag" in config["mcpServers"]


def test_setup_creates_config_if_missing(tmp_path, capsys):
    """Should create config file if it doesn't exist."""
    config_file = tmp_path / "subdir" / ".claude.json"

    with (
        patch("mcp_server.cli._find_claude_config", return_value=config_file),
        patch("shutil.which", return_value="/usr/bin/omni-rag"),
    ):
        setup()

    assert config_file.exists()
    config = json.loads(config_file.read_text())
    assert "omni-rag" in config["mcpServers"]


def test_setup_no_config_prints_manual(capsys):
    """Should print manual instructions when config not found."""
    with patch("mcp_server.cli._find_claude_config", return_value=None):
        setup()

    output = capsys.readouterr().out
    assert "Manual setup" in output
