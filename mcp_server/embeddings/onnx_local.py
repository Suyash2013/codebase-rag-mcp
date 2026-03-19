"""ONNX local embedding provider — zero-config, auto-downloads model."""

import logging
import os
from pathlib import Path

from config.settings import settings
from mcp_server.embeddings.base import EmbeddingProvider

log = logging.getLogger("codebase-rag-mcp")

# Model registry: name -> (repo_id, file, dimension)
_MODEL_REGISTRY = {
    "all-MiniLM-L6-v2": {
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
    },
}


def _download_model(model_name: str, model_dir: str) -> str:
    """Download ONNX model from HuggingFace Hub if not cached. Returns model directory path."""
    model_path = Path(model_dir) / model_name
    onnx_path = model_path / "model.onnx"

    if onnx_path.exists():
        return str(model_path)

    info = _MODEL_REGISTRY.get(model_name)
    if not info:
        raise RuntimeError(
            f"Unknown ONNX model '{model_name}'. "
            f"Available: {', '.join(_MODEL_REGISTRY.keys())}"
        )

    log.info("Downloading ONNX model '%s' (first-time setup)...", model_name)
    os.makedirs(model_path, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=info["repo_id"],
            local_dir=str(model_path),
            allow_patterns=["*.onnx", "*.json", "*.txt", "tokenizer*"],
        )
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required for ONNX model auto-download. "
            "Install it with: pip install huggingface-hub"
        )

    if not onnx_path.exists():
        raise RuntimeError(
            f"ONNX model file not found after download at {onnx_path}. "
            f"The model repo may not contain an ONNX export."
        )

    log.info("Model downloaded to %s", model_path)
    return str(model_path)


class OnnxLocalProvider(EmbeddingProvider):
    """Zero-config local embedding provider using ONNX Runtime."""

    def __init__(self) -> None:
        self._session = None
        self._tokenizer = None
        self._dimension = None

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return

        model_dir = settings.get_onnx_model_path()
        model_name = settings.onnx_model_name

        model_path = _download_model(model_name, model_dir)

        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise RuntimeError(
                f"ONNX provider requires onnxruntime and tokenizers. "
                f"Install with: pip install onnxruntime tokenizers. Error: {e}"
            )

        onnx_file = Path(model_path) / "model.onnx"
        tokenizer_file = Path(model_path) / "tokenizer.json"

        if not tokenizer_file.exists():
            raise RuntimeError(f"Tokenizer not found at {tokenizer_file}")

        self._session = ort.InferenceSession(str(onnx_file))
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self._tokenizer.enable_truncation(max_length=512)
        self._tokenizer.enable_padding(length=512)

        info = _MODEL_REGISTRY.get(model_name, {})
        self._dimension = info.get("dimension", 384)

        log.info("Loaded ONNX model '%s' (%d-dim)", model_name, self._dimension)

    def embed(self, text: str) -> list[float]:
        self._ensure_loaded()

        encoded = self._tokenizer.encode(text)
        input_ids = [encoded.ids]
        attention_mask = [encoded.attention_mask]

        import numpy as np
        outputs = self._session.run(
            None,
            {
                "input_ids": np.array(input_ids, dtype=np.int64),
                "attention_mask": np.array(attention_mask, dtype=np.int64),
                "token_type_ids": np.zeros_like(input_ids, dtype=np.int64),
            },
        )

        # Mean pooling over token embeddings
        token_embeddings = outputs[0]  # (1, seq_len, hidden_dim)
        mask = np.array(attention_mask, dtype=np.float32).reshape(1, -1, 1)
        pooled = (token_embeddings * mask).sum(axis=1) / mask.sum(axis=1)

        # Normalize
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        normalized = (pooled / norm).flatten().tolist()

        return normalized

    def dimension(self) -> int:
        if self._dimension is not None:
            info = _MODEL_REGISTRY.get(settings.onnx_model_name, {})
            return info.get("dimension", 384)
        self._ensure_loaded()
        return self._dimension
