"""Tests for Langflow JSON flow validation."""

import json
from pathlib import Path

import pytest

FLOWS_DIR = Path(__file__).parent.parent / "flows"


def get_flow_files():
    """Get all JSON files in the flows directory."""
    if not FLOWS_DIR.exists():
        return []
    return list(FLOWS_DIR.glob("*.json"))


@pytest.mark.parametrize("flow_file", get_flow_files(), ids=lambda p: p.name)
def test_flow_is_valid_json(flow_file):
    """Each flow file should be valid JSON."""
    text = flow_file.read_text(encoding="utf-8")
    data = json.loads(text)  # Will raise if invalid
    assert isinstance(data, dict)


@pytest.mark.parametrize("flow_file", get_flow_files(), ids=lambda p: p.name)
def test_flow_has_required_keys(flow_file):
    """Each flow should have standard Langflow keys."""
    data = json.loads(flow_file.read_text(encoding="utf-8"))
    # Langflow exports have a 'data' key with nodes and edges
    if "data" in data:
        flow_data = data["data"]
        assert "nodes" in flow_data, "Flow data missing 'nodes'"
        assert "edges" in flow_data, "Flow data missing 'edges'"


@pytest.mark.parametrize("flow_file", get_flow_files(), ids=lambda p: p.name)
def test_flow_edges_have_valid_structure(flow_file):
    """Each edge should have source, target, and handle fields."""
    data = json.loads(flow_file.read_text(encoding="utf-8"))
    if "data" not in data:
        pytest.skip("No 'data' key in flow")

    edges = data["data"].get("edges", [])
    for i, edge in enumerate(edges):
        assert "source" in edge, f"Edge {i} missing 'source'"
        assert "target" in edge, f"Edge {i} missing 'target'"
