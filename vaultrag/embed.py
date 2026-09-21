"""Embeddings via a local Ollama model (default: bge-m3, multilingual).

All network errors are translated into a single actionable OllamaError so the
CLI can tell the user exactly what to fix (start Ollama, pull the model).
"""

from __future__ import annotations

import ollama


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or a model is missing."""


class Embedder:
    def __init__(self, host: str, model: str) -> None:
        self.model = model
        self._client = ollama.Client(host=host)
        self._host = host

    def _wrap(self, exc: Exception) -> OllamaError:
        msg = str(exc)
        if "connection" in msg.lower() or "refused" in msg.lower():
            return OllamaError(
                f"Cannot reach Ollama at {self._host}. Start it with `ollama serve` "
                f"(or open the Ollama app)."
            )
        if "not found" in msg.lower() or "model" in msg.lower():
            return OllamaError(
                f"Model '{self.model}' is not available. Pull it with "
                f"`ollama pull {self.model}`."
            )
        return OllamaError(f"Ollama embedding call failed: {msg}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input."""
        if not texts:
            return []
        try:
            resp = self._client.embed(model=self.model, input=texts)
        except Exception as exc:  # noqa: BLE001 - normalise all client errors
            raise self._wrap(exc) from exc
        return list(resp["embeddings"])

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
