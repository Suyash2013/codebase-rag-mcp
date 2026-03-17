#!/usr/bin/env python3
"""Health check — verifies Ollama and Qdrant connectivity before running the MCP server."""

import sys

import requests


def check_ollama(base_url: str = "http://localhost:11434", model: str = "snowflake-arctic-embed:latest") -> bool:
    """Check if Ollama is running and the embedding model is available."""
    print(f"Checking Ollama at {base_url}...")

    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        print(f"  Ollama is running. Models: {len(models)}")

        model_base = model.split(":")[0]
        found = any(model_base in m for m in models)
        if found:
            print(f"  Model '{model}' is available.")
        else:
            print(f"  WARNING: Model '{model}' not found. Run: ollama pull {model}")
            return False

        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def check_qdrant(host: str = "localhost", port: int = 6333) -> bool:
    """Check if Qdrant is running."""
    print(f"Checking Qdrant at {host}:{port}...")

    try:
        resp = requests.get(f"http://{host}:{port}/collections", timeout=5)
        resp.raise_for_status()
        collections = resp.json().get("result", {}).get("collections", [])
        print(f"  Qdrant is running. Collections: {len(collections)}")
        for c in collections:
            print(f"    - {c.get('name', 'unknown')}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        return False


def main():
    print("=== Codebase RAG MCP — Health Check ===\n")

    ollama_ok = check_ollama()
    print()
    qdrant_ok = check_qdrant()

    print("\n=== Summary ===")
    print(f"  Ollama: {'OK' if ollama_ok else 'FAILED'}")
    print(f"  Qdrant: {'OK' if qdrant_ok else 'FAILED'}")

    if ollama_ok and qdrant_ok:
        print("\nAll checks passed. Ready to run MCP server.")
        sys.exit(0)
    else:
        print("\nSome checks failed. Fix the issues above before starting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
