#!/usr/bin/env python3
"""Release script — bump version in pyproject.toml, commit, tag, and push."""

import re
import subprocess
import sys
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
# PEP 440: X.Y.Z or X.Y.ZrcN
VALID_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(rc\d+)?$")


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    if dry_run:
        args.remove("--dry-run")

    if not args:
        print("Usage: python scripts/release.py [--dry-run] VERSION")
        print("  VERSION: PEP 440 version (e.g., 3.1.0 or 3.1.0rc1)")
        sys.exit(1)

    version = args[0]

    # Validate version format
    if not VALID_VERSION_RE.match(version):
        die(f"Invalid version '{version}'. Expected format: X.Y.Z or X.Y.ZrcN")

    # Read current version
    text = PYPROJECT.read_text(encoding="utf-8")
    match = VERSION_RE.search(text)
    if not match:
        die("Could not find version in pyproject.toml")

    current = match.group(1)
    if current == version:
        die(f"Version is already {version}")

    tag = f"v{version}"

    # Check tag doesn't exist
    result = run(["git", "tag", "-l", tag], check=False)
    if tag in result.stdout.strip().splitlines():
        die(f"Tag {tag} already exists")

    # Check working tree is clean (tracked files only)
    result = run(["git", "diff", "--quiet", "HEAD"], check=False)
    if result.returncode != 0:
        status = run(["git", "status", "--short"], check=False)
        die(f"Working tree is dirty. Commit or stash changes first.\n{status.stdout}")

    print(f"Releasing: {current} -> {version}")
    print(f"  Tag: {tag}")
    print()

    if dry_run:
        print("[dry-run] Would update pyproject.toml version")
        print(f'[dry-run] Would commit: "release: v{version}"')
        print(f"[dry-run] Would create tag: {tag}")
        print("[dry-run] Would push commit and tag to origin")
        print()
        print("No changes made (dry-run mode).")
        return

    # Update version in pyproject.toml
    new_text = text.replace(f'version = "{current}"', f'version = "{version}"')
    PYPROJECT.write_text(new_text, encoding="utf-8")
    print(f"Updated pyproject.toml: {current} -> {version}")

    # Commit
    run(["git", "add", str(PYPROJECT)])
    run(["git", "commit", "-m", f"release: v{version}"])
    print(f"Committed: release: v{version}")

    # Tag
    run(["git", "tag", "-a", tag, "-m", f"Release {tag}"])
    print(f"Tagged: {tag}")

    # Push
    print("Pushing to origin...")
    result = run(["git", "push", "origin", "HEAD"], check=False)
    if result.returncode != 0:
        print(f"Warning: push failed: {result.stderr}", file=sys.stderr)
        print("Tag is local. You can retry with: git push origin HEAD && git push origin", tag)
        sys.exit(1)

    result = run(["git", "push", "origin", tag], check=False)
    if result.returncode != 0:
        print(f"Warning: tag push failed: {result.stderr}", file=sys.stderr)
        print(f"Retry with: git push origin {tag}")
        sys.exit(1)

    print()
    print(f"Released v{version}!")
    print("CI will now build, test, and publish to PyPI.")


if __name__ == "__main__":
    main()
