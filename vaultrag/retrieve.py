"""Retrieval: embed the query, KNN over chunks, optional 1-hop link expansion.

The vault is a dense wikilink graph with MOC hubs, so pulling chunks from notes
that the top hits link to often surfaces the connective context a pure vector
search misses.
"""

from __future__ import annotations

from .embed import Embedder
from .models import Hit
from .store import Store


def retrieve(
    query: str,
    store: Store,
    embedder: Embedder,
    *,
    k: int = 6,
    expand_links: bool = True,
) -> list[Hit]:
    """Return up to ~k primary hits, plus a few link-expanded neighbours."""
    query_vec = embedder.embed_one(query)
    hits = store.search(query_vec, k)
    if not hits or not expand_links:
        return hits

    top_paths = [h.chunk.rel_path for h in hits[: max(1, k // 3)]]
    seen_paths = {h.chunk.rel_path for h in hits}

    linked_titles = store.links_of(top_paths)
    if not linked_titles:
        return hits

    extra = [
        c
        for c in store.chunks_for_notes(linked_titles)
        if c.rel_path not in seen_paths and c.ordinal == 0  # lead chunk only
    ]
    # A small, bounded number of expansion chunks; distance unknown, so tag as inf.
    for chunk in extra[:3]:
        hits.append(Hit(chunk=chunk, distance=float("inf")))
    return hits
