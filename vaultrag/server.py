"""Local web UI for the RAG pipeline.

A tiny FastAPI app that serves a single-page chat UI and streams answers over
Server-Sent Events, reusing the exact same retrieve + answer path as the CLI.
It is meant to be run on localhost only; nothing here is exposed publicly.

Kept in an optional `web` extra so the core stays dependency-light:
    uv sync --extra web
    uv run vaultrag serve
"""

from __future__ import annotations

import json
from pathlib import Path

# Imported at module level (not inside create_app) so that FastAPI can resolve
# the `request: Request` annotation, which `from __future__ import annotations`
# turns into a string it looks up in this module's globals. server.py itself is
# only imported lazily by the CLI, so the web extra stays optional.
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from .answer import source_list, stream_answer
from .config import Config
from .embed import Embedder, OllamaError
from .retrieve import retrieve
from .store import Store

WEB_DIR = Path(__file__).parent / "web"


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(cfg: Config):
    app = FastAPI(title="vaultrag", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/status")
    def status() -> dict:
        store = Store(cfg.db_path)
        stats = store.stats()
        store.close()
        return {
            **stats,
            "vault_name": cfg.vault_path.name,
            "embed_model": cfg.embed_model,
            "chat_model": cfg.chat_model,
        }

    @app.post("/api/ask")
    async def ask(request: Request):
        body = await request.json()
        q = (body.get("question") or "").strip()
        history = body.get("history") or []
        k = int(body.get("k") or 0)

        # For short follow-ups ("stimmt nicht ...", "und warum?"), fold in the
        # previous user turn so retrieval has something concrete to match.
        retrieval_query = q
        prev_user = next(
            (m["content"] for m in reversed(history) if m.get("role") == "user"), None
        )
        if prev_user and len(q.split()) < 6:
            retrieval_query = f"{prev_user}\n{q}"

        def gen():
            store = Store(cfg.db_path)
            try:
                if store.stats()["notes"] == 0:
                    yield _sse("error", {"message": "Index is empty. Run `vaultrag index`."})
                    return
                embedder = Embedder(cfg.ollama_host, cfg.embed_model)
                try:
                    hits = retrieve(retrieval_query, store, embedder, k=k or cfg.top_k)
                except OllamaError as exc:
                    yield _sse("error", {"message": str(exc)})
                    return
                yield _sse("sources", {"sources": source_list(hits)})
                if not hits:
                    yield _sse("done", {})
                    return
                try:
                    for piece in stream_answer(q, hits, cfg, history=history):
                        yield _sse("token", {"text": piece})
                except OllamaError as exc:
                    yield _sse("error", {"message": str(exc)})
                    return
                yield _sse("done", {})
            finally:
                store.close()

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def serve(cfg: Config, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")
