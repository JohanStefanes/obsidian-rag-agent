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

from .answer import source_list, stream_answer
from .config import Config
from .embed import Embedder, OllamaError
from .retrieve import retrieve
from .store import Store

WEB_DIR = Path(__file__).parent / "web"


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(cfg: Config):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse

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

    @app.get("/api/ask")
    def ask(q: str, k: int = 0):
        def gen():
            store = Store(cfg.db_path)
            try:
                if store.stats()["notes"] == 0:
                    yield _sse("error", {"message": "Index is empty. Run `vaultrag index`."})
                    return
                embedder = Embedder(cfg.ollama_host, cfg.embed_model)
                try:
                    hits = retrieve(q, store, embedder, k=k or cfg.top_k)
                except OllamaError as exc:
                    yield _sse("error", {"message": str(exc)})
                    return
                yield _sse("sources", {"sources": source_list(hits)})
                if not hits:
                    yield _sse("done", {})
                    return
                try:
                    for piece in stream_answer(q, hits, cfg):
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
