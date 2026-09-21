"""Answer generation: assemble retrieved context and ask the local chat model.

The model is instructed to answer in the language of the question (the vault is
mostly German), to ground answers in the provided notes, and to cite sources as
[[Note Title]] so the user can jump straight to them in Obsidian.
"""

from __future__ import annotations

import ollama

from .config import Config
from .embed import OllamaError
from .models import Hit

_SYSTEM = (
    "Du bist ein Assistent, der Fragen anhand der bereitgestellten Notizen aus einem "
    "persoenlichen Obsidian-Vault beantwortet. Antworte in der Sprache der Frage. "
    "Stuetze dich auf den Kontext; wenn er die Antwort nicht hergibt, sage das offen. "
    "Zitiere die verwendeten Notizen als [[Notiztitel]]. Der Abschnitt 'Setup' "
    "beschreibt die Konfiguration dieses Tools (z.B. den Dateipfad des Vaults auf der "
    "Festplatte); nutze ihn nur fuer Fragen ueber das Tool selbst und zitiere ihn nicht "
    "als Notiz. Verwende Schweizer Rechtschreibung (ss statt ß) und keine "
    "Gedankenstriche (— oder –) im Fliesstext."
)


def _format_context(hits: list[Hit]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(f"[[{hit.chunk.title}]]\n{hit.chunk.text}")
    return "\n\n---\n\n".join(blocks)


def _clean_history(history: list[dict] | None, keep: int = 8) -> list[dict]:
    """Keep the last `keep` valid user/assistant turns for conversational context."""
    out = []
    for m in history or []:
        role, content = m.get("role"), m.get("content")
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out[-keep:]


def build_messages(
    query: str, hits: list[Hit], cfg: Config, history: list[dict] | None = None
) -> list[dict]:
    context = _format_context(hits)
    setup = (
        "Setup (Konfiguration dieses Tools, keine Vault-Notiz):\n"
        f"- Vault-Pfad auf der Festplatte: {cfg.vault_path}\n"
        f"- Embedding-Modell: {cfg.embed_model}\n"
        f"- Chat-Modell: {cfg.chat_model}"
    )
    user = (
        f"{setup}\n\n"
        f"Kontext aus dem Vault:\n\n{context}\n\n"
        f"Frage: {query}\n\n"
        f"Beantworte die Frage und beziehe den bisherigen Gespraechsverlauf mit ein "
        f"(z.B. Korrekturen des Nutzers). Geht es um das Tool selbst (z.B. den Vault-Pfad "
        f"oder die Modelle), nutze den Setup-Abschnitt und zitiere ihn nicht als [[Notiz]]. "
        f"Sonst stuetze dich auf den Kontext und nenne die Quellen als [[Notiz]]."
    )
    return [
        {"role": "system", "content": _SYSTEM},
        *_clean_history(history),
        {"role": "user", "content": user},
    ]


def stream_answer(query: str, hits: list[Hit], cfg: Config, history: list[dict] | None = None):
    """Yield answer text chunks from the local chat model."""
    client = ollama.Client(host=cfg.ollama_host)
    model = cfg.chat_model
    messages = build_messages(query, hits, cfg, history)
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
