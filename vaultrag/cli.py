"""Command-line entry point: `vaultrag index | ask | status`."""

from __future__ import annotations

import argparse
import sys
import time

from .answer import source_list, stream_answer
from .config import Config
from .embed import Embedder, OllamaError
from .retrieve import retrieve
from .store import Store
from .vault import load_vault


def _cmd_index(args: argparse.Namespace, cfg: Config) -> int:
    print(f"Vault:  {cfg.vault_path}")
    print(f"Model:  {cfg.embed_model} (embeddings)")
    print("Loading notes ...", flush=True)
    notes = load_vault(cfg.vault_path)
    print(f"  {len(notes)} indexable notes (Graphify and stubs excluded)")

    store = Store(cfg.db_path)
    embedder = Embedder(cfg.ollama_host, cfg.embed_model)
    start = time.time()
    done = 0

    def progress(rel_path: str) -> None:
        nonlocal done
        done += 1
        if done % 25 == 0:
            print(f"  embedded {done} notes ...", flush=True)

    try:
        stats = store.index_notes(notes, embedder, rebuild=args.rebuild, progress=progress)
    except OllamaError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    elapsed = time.time() - start
    print(
        f"Done in {elapsed:.1f}s: "
        f"+{stats['added']} added, ~{stats['updated']} updated, "
        f"={stats['unchanged']} unchanged, -{stats['removed']} removed; "
        f"{stats['chunks']} chunks (re)embedded."
    )
    return 0


def _cmd_ask(args: argparse.Namespace, cfg: Config) -> int:
    store = Store(cfg.db_path)
    if store.stats()["notes"] == 0:
        print("Index is empty. Run `vaultrag index` first.", file=sys.stderr)
        store.close()
        return 1
    embedder = Embedder(cfg.ollama_host, cfg.embed_model)
    try:
        hits = retrieve(
            args.question, store, embedder, k=args.k or cfg.top_k, expand_links=not args.no_expand
        )
    except OllamaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        store.close()
        return 1

    if not hits:
        print("No relevant notes found.")
        store.close()
        return 0

    try:
        for piece in stream_answer(args.question, hits, cfg.ollama_host, cfg.chat_model):
            print(piece, end="", flush=True)
        print()
    except OllamaError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        store.close()
        return 1

    print("\nQuellen:")
    for path in source_list(hits):
        print(f"  - {path}")
    store.close()
    return 0


def _cmd_status(args: argparse.Namespace, cfg: Config) -> int:
    store = Store(cfg.db_path)
    stats = store.stats()
    store.close()
    size_mb = cfg.db_path.stat().st_size / 1e6 if cfg.db_path.exists() else 0.0
    print(f"Vault:       {cfg.vault_path}")
    print(f"Index DB:    {cfg.db_path} ({size_mb:.1f} MB)")
    print(f"Embed model: {cfg.embed_model}   Chat model: {cfg.chat_model}")
    print(f"Notes:       {stats['notes']}")
    print(f"Chunks:      {stats['chunks']}")
    print(f"Dim:         {stats['dim']}")
    return 0


def _cmd_serve(args: argparse.Namespace, cfg: Config) -> int:
    try:
        from .server import serve
    except ModuleNotFoundError:
        print(
            "Web dependencies are missing. Install them with `uv sync --extra web`.",
            file=sys.stderr,
        )
        return 1
    print(f"vaultrag UI:  http://{args.host}:{args.port}")
    print(f"Vault:        {cfg.vault_path}")
    print("Ctrl-C to stop.")
    try:
        serve(cfg, args.host, args.port)
    except ModuleNotFoundError:
        print("Web dependencies are missing. Install with `uv sync --extra web`.", file=sys.stderr)
        return 1
    return 0


def _cmd_app(args: argparse.Namespace, cfg: Config) -> int:
    try:
        from .desktop import run_desktop
        import webview  # noqa: F401 - presence check for a clear error
    except ModuleNotFoundError:
        print(
            "Desktop dependencies are missing. Install them with "
            "`uv sync --extra web --extra app`.",
            file=sys.stderr,
        )
        return 1
    run_desktop(cfg, args.host, args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vaultrag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build or refresh the vault index")
    p_index.add_argument("--rebuild", action="store_true", help="drop and rebuild from scratch")
    p_index.set_defaults(func=_cmd_index)

    p_ask = sub.add_parser("ask", help="ask a question against the vault")
    p_ask.add_argument("question", help="the question (quote it)")
    p_ask.add_argument("-k", type=int, default=0, help="number of chunks to retrieve")
    p_ask.add_argument("--no-expand", action="store_true", help="disable wikilink expansion")
    p_ask.set_defaults(func=_cmd_ask)

    p_status = sub.add_parser("status", help="show index stats")
    p_status.set_defaults(func=_cmd_status)

    p_serve = sub.add_parser("serve", help="run the local web UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    p_app = sub.add_parser("app", help="run the UI in a native desktop window")
    p_app.add_argument("--host", default="127.0.0.1")
    p_app.add_argument("--port", type=int, default=8000)
    p_app.set_defaults(func=_cmd_app)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = Config.load()
    try:
        return args.func(args, cfg)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
