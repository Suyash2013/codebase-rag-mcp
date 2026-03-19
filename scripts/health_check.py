#!/usr/bin/env python3
"""Health check — verifies configured backends are available."""

import os
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


def check_qdrant_remote(host: str = "localhost", port: int = 6333) -> bool:
    """Check if remote Qdrant is running."""
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


def check_qdrant_local(path: str) -> bool:
    """Check if local Qdrant path is writable."""
    print(f"Checking local Qdrant storage at {path}...")
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, ".health_check")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        print("  Local storage path is writable.")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def check_onnx_model(model_name: str, model_dir: str) -> bool:
    """Check if ONNX model is downloaded."""
    model_path = os.path.join(model_dir, model_name, "model.onnx")
    print(f"Checking ONNX model '{model_name}'...")
    if os.path.exists(model_path):
        print(f"  Model found at {model_path}")
        return True
    else:
        print(f"  Model not found (will auto-download on first use)")
        return True  # Not a failure — auto-downloads


def check_api_key(provider: str, key: str) -> bool:
    """Check if API key is configured for cloud providers."""
    print(f"Checking {provider} API key...")
    if key:
        print(f"  API key configured (starts with {key[:8]}...)")
        return True
    else:
        print(f"  FAILED: No API key set. Set RAG_{provider.upper()}_API_KEY.")
        return False


def main():
    from config.settings import settings

    print("=== Codebase RAG MCP — Health Check ===\n")
    print(f"Configuration:")
    print(f"  Embedding provider: {settings.embedding_provider}")
    print(f"  Qdrant mode: {settings.qdrant_mode}")
    print(f"  Working directory: {settings.get_working_directory()}")
    print()

    all_ok = True

    # Check embedding provider
    match settings.embedding_provider:
        case "onnx":
            all_ok &= check_onnx_model(
                settings.onnx_model_name,
                settings.get_onnx_model_path(),
            )
        case "ollama":
            all_ok &= check_ollama(settings.ollama_base_url, settings.ollama_embed_model)
        case "openai":
            all_ok &= check_api_key("openai", settings.openai_api_key)
        case "voyage":
            all_ok &= check_api_key("voyage", settings.voyage_api_key)

    print()

    # Check Qdrant
    if settings.qdrant_mode == "local":
        all_ok &= check_qdrant_local(settings.get_qdrant_local_path())
    else:
        all_ok &= check_qdrant_remote(settings.qdrant_host, settings.qdrant_port)

    print(f"\n=== {'All checks passed' if all_ok else 'Some checks failed'} ===")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
