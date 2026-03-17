"""MCP tools for ingestion control."""

import json

import requests

from config.settings import settings
from mcp_server.ingestion import check_local_changes, ingest_directory, needs_ingestion


def ingest_current_directory(force: bool = False) -> str:
    """Manually trigger ingestion of the current working directory.

    Use force=True to re-ingest even if already indexed (e.g., after code changes).
    Timeout: 24 hours (configurable via RAG_INGESTION_TIMEOUT_HOURS).

    Args:
        force: If True, re-ingest even if the directory is already indexed.
    """
    directory = settings.get_working_directory()

    try:
        if not force and not needs_ingestion(directory):
            return f"Directory already indexed: {directory}. Use force=True to re-ingest."

        return ingest_directory(directory)

    except Exception as exc:
        return f"Error during ingestion: {exc}"


def check_index_status() -> str:
    """Check if the current working directory is indexed and whether local changes exist.

    Use this to decide whether to use semantic search or fall back to file search.
    If has_changes is True, file search may be more accurate for changed files.
    """
    directory = settings.get_working_directory()

    try:
        indexed = not needs_ingestion(directory)
        changes = check_local_changes(directory)

        result = {
            "directory": directory,
            "is_indexed": indexed,
            "recommendation": (
                "Use semantic search"
                if indexed and not changes.get("has_changes")
                else "Consider file search for changed files"
                if indexed and changes.get("has_changes")
                else "Ingestion needed — will auto-ingest on first search"
            ),
            **changes,
        }

        return json.dumps(result, indent=2)

    except Exception as exc:
        return f"Error checking index status: {exc}"


def trigger_langflow_ingestion(directory: str = "") -> str:
    """Trigger Langflow ingestion pipeline via REST API.

    This calls the Langflow flow configured in RAG_LANGFLOW_FLOW_ID to run
    the full ingestion pipeline (Directory -> Splitter -> Embeddings -> Qdrant).

    Args:
        directory: Directory to ingest. Defaults to current working directory.
    """
    if not settings.langflow_flow_id:
        return "Langflow flow ID not configured. Set RAG_LANGFLOW_FLOW_ID."

    directory = directory or settings.get_working_directory()

    try:
        url = f"{settings.langflow_base_url}/api/v1/run/{settings.langflow_flow_id}"
        resp = requests.post(
            url,
            json={"input_value": directory, "output_type": "text"},
            timeout=settings.ingestion_timeout_hours * 3600,
        )
        resp.raise_for_status()
        return f"Langflow ingestion triggered for {directory}: {resp.json()}"
    except Exception as exc:
        return f"Error triggering Langflow ingestion: {exc}"
