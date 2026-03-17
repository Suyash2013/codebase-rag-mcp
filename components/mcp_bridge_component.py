"""
MCP Bridge Component for Langflow
===================================
Queries the same Qdrant index used by the MCP server, enabling Langflow
flows to perform semantic search over the indexed codebase.
"""

from lfx.custom.custom_component.component import Component
from lfx.io import IntInput, MessageTextInput, StrInput
from lfx.schema.message import Message
from lfx.template.field.base import Output


class MCPBridgeComponent(Component):
    display_name = "Codebase Search (Qdrant)"
    description = (
        "Semantic search over a Qdrant-indexed codebase. "
        "Uses the same index as the codebase-rag MCP server."
    )
    icon = "search"
    name = "MCPBridge"

    inputs = [
        MessageTextInput(
            name="query",
            display_name="Search Query",
            info="Natural language description of what you're looking for.",
            tool_mode=True,
        ),
        IntInput(
            name="n_results",
            display_name="Number of Results",
            info="How many results to return (1-20).",
            value=10,
        ),
        StrInput(
            name="file_pattern",
            display_name="File Pattern (optional)",
            info="Substring to filter file paths, e.g. 'viewmodel', '.py', 'src/main'.",
            value="",
            required=False,
        ),
        StrInput(
            name="qdrant_host",
            display_name="Qdrant Host",
            info="Qdrant server host.",
            value="localhost",
            advanced=True,
        ),
        IntInput(
            name="qdrant_port",
            display_name="Qdrant Port",
            info="Qdrant server port.",
            value=6333,
            advanced=True,
        ),
        StrInput(
            name="collection_name",
            display_name="Collection Name",
            info="Qdrant collection to search.",
            value="codebase",
            advanced=True,
        ),
        StrInput(
            name="ollama_base_url",
            display_name="Ollama Base URL",
            info="Ollama API base URL for embeddings.",
            value="http://localhost:11434",
            advanced=True,
        ),
        StrInput(
            name="embed_model",
            display_name="Embedding Model",
            info="Ollama embedding model name.",
            value="snowflake-arctic-embed:latest",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Search Results",
            name="results",
            method="run_search",
        ),
    ]

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding from Ollama."""
        import requests

        # Try new API first
        try:
            resp = requests.post(
                f"{self.ollama_base_url}/api/embed",
                json={"model": self.embed_model, "input": text},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["embeddings"][0]
        except Exception:
            pass

        # Legacy fallback
        resp = requests.post(
            f"{self.ollama_base_url}/api/embeddings",
            json={"model": self.embed_model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def run_search(self) -> Message:
        """Execute semantic search against Qdrant."""
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        n = max(1, min(self.n_results, 20))

        try:
            client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            embedding = self._get_embedding(self.query)

            response = client.query_points(
                collection_name=self.collection_name,
                query=embedding,
                limit=n * 5 if self.file_pattern else n,
            )

            hits = []
            for point in response.points:
                payload = point.payload or {}
                file_path = payload.get("file_path", "unknown")

                if self.file_pattern and self.file_pattern.lower() not in file_path.lower():
                    continue

                hits.append(
                    f"**{file_path}** (score: {point.score:.4f})\n"
                    f"```\n{payload.get('text', '')}\n```"
                )

                if len(hits) >= n:
                    break

            if not hits:
                return Message(text=f"No results found for: {self.query}")

            header = f"Found {len(hits)} results for: \"{self.query}\"\n\n"
            return Message(text=header + "\n\n---\n\n".join(hits))

        except Exception as e:
            self.log(f"Search failed: {e}")
            return Message(text=f"Error searching codebase: {e}")
