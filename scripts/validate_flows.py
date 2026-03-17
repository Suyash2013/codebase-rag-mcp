#!/usr/bin/env python3
"""Validate all Langflow JSON flow files in the flows/ directory."""

import json
import sys
from pathlib import Path


def validate_flow(path: Path) -> list[str]:
    """Validate a single flow file. Returns list of errors (empty if valid)."""
    errors = []

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return [f"Cannot read file: {e}"]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    if not isinstance(data, dict):
        return ["Root is not a JSON object"]

    if "data" in data:
        flow_data = data["data"]
        if "nodes" not in flow_data:
            errors.append("Missing 'data.nodes'")
        if "edges" not in flow_data:
            errors.append("Missing 'data.edges'")

        for i, edge in enumerate(flow_data.get("edges", [])):
            if "source" not in edge:
                errors.append(f"Edge {i}: missing 'source'")
            if "target" not in edge:
                errors.append(f"Edge {i}: missing 'target'")

    return errors


def main():
    flows_dir = Path(__file__).parent.parent / "flows"

    if not flows_dir.exists():
        print(f"Flows directory not found: {flows_dir}")
        sys.exit(1)

    flow_files = list(flows_dir.glob("*.json"))
    if not flow_files:
        print("No JSON files found in flows/")
        sys.exit(0)

    all_valid = True
    for path in sorted(flow_files):
        errors = validate_flow(path)
        if errors:
            print(f"FAIL: {path.name}")
            for e in errors:
                print(f"  - {e}")
            all_valid = False
        else:
            print(f"OK:   {path.name}")

    if all_valid:
        print(f"\nAll {len(flow_files)} flow(s) valid.")
        sys.exit(0)
    else:
        print(f"\nSome flows have errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
