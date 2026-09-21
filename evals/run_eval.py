"""Retrieval eval: hit@k over a hand-written question set.

Each question lists `expect` substrings; a question counts as a hit if any
retrieved source path contains any expected substring. Run against the real
index (needs the embed model available in Ollama):

    uv run python evals/run_eval.py           # uses default k
    uv run python evals/run_eval.py --k 8

This is intentionally a plain script (not a `vaultrag` subcommand) so the eval
set stays a versioned, editable artifact next to the numbers it produces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vaultrag.answer import source_list
from vaultrag.config import Config
from vaultrag.embed import Embedder
from vaultrag.retrieve import retrieve
from vaultrag.store import Store

QUESTIONS = Path(__file__).with_name("questions.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=0, help="chunks to retrieve (0 = config default)")
    args = parser.parse_args()

    cfg = Config.load()
    store = Store(cfg.db_path)
    if store.stats()["notes"] == 0:
        print("Index empty. Run `vaultrag index` first.")
        return 1
    embedder = Embedder(cfg.ollama_host, cfg.embed_model)
    cases = json.loads(QUESTIONS.read_text(encoding="utf-8"))

    hits = 0
    for case in cases:
        results = retrieve(case["q"], store, embedder, k=args.k or cfg.top_k)
        sources = source_list(results)
        matched = any(
            any(exp.lower() in src.lower() for src in sources) for exp in case["expect"]
        )
        hits += matched
        mark = "PASS" if matched else "MISS"
        top = sources[0] if sources else "(none)"
        print(f"[{mark}] {case['q']}\n        top: {top}")

    n = len(cases)
    print(f"\nhit@{args.k or cfg.top_k}: {hits}/{n} = {hits / n:.0%}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
