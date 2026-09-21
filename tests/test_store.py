from __future__ import annotations

from pathlib import Path

from vaultrag.retrieve import retrieve
from vaultrag.store import Store
from vaultrag.vault import load_vault


def test_index_and_incremental(tmp_path: Path, vault: Path, fake_embedder) -> None:
    store = Store(tmp_path / "idx.db")
    notes = load_vault(vault)
    stats = store.index_notes(notes, fake_embedder)
    assert stats["added"] == 3
    assert stats["chunks"] > 0
    assert store.stats()["notes"] == 3

    # Re-index unchanged: everything unchanged, nothing re-embedded.
    stats2 = store.index_notes(load_vault(vault), fake_embedder)
    assert stats2["unchanged"] == 3
    assert stats2["chunks"] == 0
    store.close()


def test_deletion_is_synced(tmp_path: Path, vault: Path, fake_embedder) -> None:
    store = Store(tmp_path / "idx.db")
    store.index_notes(load_vault(vault), fake_embedder)
    (vault / "Lernjournale/LJ-KW16-2026.md").unlink()
    stats = store.index_notes(load_vault(vault), fake_embedder)
    assert stats["removed"] == 1
    assert store.stats()["notes"] == 2
    store.close()


def test_search_returns_hits(tmp_path: Path, vault: Path, fake_embedder) -> None:
    store = Store(tmp_path / "idx.db")
    store.index_notes(load_vault(vault), fake_embedder)
    hits = retrieve("Applikationssicherheit", store, fake_embedder, k=3, expand_links=False)
    assert hits
    assert all(h.chunk.text for h in hits)
    store.close()
