"""CLI entry points for rag-mcp."""

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
    """Auto-register rag-mcp with Claude Code."""
    print("=== rag-mcp Setup ===\n")

    # Find the rag-mcp command
    cmd = shutil.which("rag-mcp")
    if cmd:
        command = "rag-mcp"
    else:
        # Fall back to python -m
        command = sys.executable
        args = ["-m", "mcp_server.server"]
        print(f"Note: 'rag-mcp' not found in PATH, using: {command} {' '.join(args)}")

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
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}
    except (json.JSONDecodeError, OSError):
        config = {}

    # Add MCP server entry
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    if "rag-mcp" in config["mcpServers"]:
        print("rag-mcp is already registered in Claude Code.")
        print(f"Config file: {config_path}")
        return

    if cmd:
        config["mcpServers"]["rag-mcp"] = {
            "command": "rag-mcp",
        }
    else:
        config["mcpServers"]["rag-mcp"] = {
            "command": command,
            "args": ["-m", "mcp_server.server"],
        }

    # Write config
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"Registered rag-mcp in {config_path}")
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
    server_config: dict[str, object] = {"command": command}
    if not is_direct:
        server_config["args"] = ["-m", "mcp_server.server"]
    else:
        server_config["command"] = "rag-mcp"
    config = {"mcpServers": {"rag-mcp": server_config}}

    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup()
    else:
        main()
