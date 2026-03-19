"""CLI entry points for codebase-rag-mcp."""

import json
import os
import shutil
import sys
from pathlib import Path


def main():
    """Run the MCP server."""
    from mcp_server.server import mcp
    mcp.run()


def setup():
    """Auto-register codebase-rag-mcp with Claude Code."""
    print("=== codebase-rag-mcp Setup ===\n")

    # Find the codebase-rag command
    cmd = shutil.which("codebase-rag")
    if cmd:
        command = "codebase-rag"
    else:
        # Fall back to python -m
        command = sys.executable
        args = ["-m", "mcp_server.server"]
        print(f"Note: 'codebase-rag' not found in PATH, using: {command} {' '.join(args)}")

    # Determine Claude Code config path
    config_path = _find_claude_config()

    if not config_path:
        print("Could not find Claude Code configuration.")
        print("\nManual setup — add this to your Claude Code MCP settings:\n")
        _print_manual_config(command, cmd is not None)
        return

    # Read existing config
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}
    except (json.JSONDecodeError, OSError):
        config = {}

    # Add MCP server entry
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    if "codebase-rag" in config["mcpServers"]:
        print("codebase-rag is already registered in Claude Code.")
        print(f"Config file: {config_path}")
        return

    if cmd:
        config["mcpServers"]["codebase-rag"] = {
            "command": "codebase-rag",
        }
    else:
        config["mcpServers"]["codebase-rag"] = {
            "command": command,
            "args": ["-m", "mcp_server.server"],
        }

    # Write config
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"Registered codebase-rag-mcp in {config_path}")
        print("\nRestart Claude Code to activate the plugin.")
    except OSError as e:
        print(f"Error writing config: {e}")
        print("\nManual setup:\n")
        _print_manual_config(command, cmd is not None)


def _find_claude_config() -> Path | None:
    """Find the Claude Code configuration file."""
    # Check common locations
    candidates = []

    # ~/.claude.json (global)
    home = Path.home()
    candidates.append(home / ".claude.json")

    # ~/.claude/settings.json
    candidates.append(home / ".claude" / "settings.json")

    # Check CLAUDE_CONFIG env var
    env_config = os.environ.get("CLAUDE_CONFIG")
    if env_config:
        candidates.insert(0, Path(env_config))

    # Return first existing, or first writable candidate
    for p in candidates:
        if p.exists():
            return p

    # Default to ~/.claude.json
    return candidates[0]


def _print_manual_config(command: str, is_direct: bool):
    """Print manual configuration instructions."""
    if is_direct:
        config = {
            "mcpServers": {
                "codebase-rag": {
                    "command": "codebase-rag",
                }
            }
        }
    else:
        config = {
            "mcpServers": {
                "codebase-rag": {
                    "command": command,
                    "args": ["-m", "mcp_server.server"],
                }
            }
        }

    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup()
    else:
        main()
