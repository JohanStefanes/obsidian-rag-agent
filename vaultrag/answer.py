"""Answer generation: assemble retrieved context and ask the local chat model.

The model is instructed to answer in the language of the question (the vault is
mostly German), to ground answers in the provided notes, and to cite sources as
[[Note Title]] so the user can jump straight to them in Obsidian.
"""

from __future__ import annotations

import ollama

from .embed import OllamaError
from .models import Hit

_SYSTEM = (
    "Du bist ein Assistent, der Fragen ausschliesslich anhand der bereitgestellten "
    "Notizen aus einem persoenlichen Obsidian-Vault beantwortet. Antworte in der "
    "Sprache der Frage. Stuetze dich nur auf den Kontext; wenn er die Antwort nicht "
    "hergibt, sage das offen. Zitiere die verwendeten Notizen als [[Notiztitel]]. "
    "Verwende keine Gedankenstriche (— oder –) im Fliesstext."
)


def _format_context(hits: list[Hit]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(f"[[{hit.chunk.title}]]\n{hit.chunk.text}")
    return "\n\n---\n\n".join(blocks)


def build_messages(query: str, hits: list[Hit]) -> list[dict]:
    context = _format_context(hits)
    user = (
        f"Kontext aus dem Vault:\n\n{context}\n\n"
        f"Frage: {query}\n\n"
        f"Beantworte die Frage anhand des Kontexts und nenne die Quellen als [[Notiz]]."
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]


def stream_answer(query: str, hits: list[Hit], host: str, model: str):
    """Yield answer text chunks from the local chat model."""
    client = ollama.Client(host=host)
    messages = build_messages(query, hits)
    try:
        for part in client.chat(model=model, messages=messages, stream=True):
            yield part["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "not found" in msg.lower():
            raise OllamaError(f"Chat model '{model}' missing. Pull it: `ollama pull {model}`.") from exc
        if "connection" in msg.lower() or "refused" in msg.lower():
            raise OllamaError(f"Cannot reach Ollama at {host}. Start it with `ollama serve`.") from exc
        raise OllamaError(f"Ollama chat call failed: {msg}") from exc


def source_list(hits: list[Hit]) -> list[str]:
    """Distinct source note paths, in first-seen order, for a citations footer."""
    seen: dict[str, None] = {}
    for hit in hits:
        seen.setdefault(hit.chunk.rel_path, None)
    return list(seen)
