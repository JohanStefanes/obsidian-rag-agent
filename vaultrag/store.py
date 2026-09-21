"""SQLite + sqlite-vec index.

One file holds everything: note metadata, chunk text, and a vec0 virtual table
for cosine KNN. Chosen over a heavyweight vector DB to match the project's
stdlib-sqlite house style and because the corpus is small (~600 notes).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Callable

import sqlite_vec

from .chunk import chunk_note
from .embed import Embedder
from .models import Chunk, Hit, Note


class Store:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self._load_extension()
        self._create_base_schema()

    def _load_extension(self) -> None:
        try:
            self.db.enable_load_extension(True)
            sqlite_vec.load(self.db)
            self.db.enable_load_extension(False)
        except (AttributeError, sqlite3.OperationalError) as exc:
            raise RuntimeError(
                "This Python build cannot load SQLite extensions, which sqlite-vec "
                "needs. Use the uv-managed Python (`uv run ...`) rather than a system "
                "Python compiled without extension support."
            ) from exc

    def _create_base_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS notes (
                rel_path TEXT PRIMARY KEY,
                title TEXT,
                mtime REAL,
                content_hash TEXT,
                tags_json TEXT,
                links_json TEXT
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                rel_path TEXT,
                title TEXT,
                heading TEXT,
                text TEXT,
                ordinal INTEGER,
                tags_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_rel_path ON chunks(rel_path);
            """
        )
        self.db.commit()

    # -- vec table (created once the embedding dimension is known) -----------

    def _dim(self) -> int | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = 'dim'").fetchone()
        return int(row["value"]) if row else None

    def _ensure_vec_table(self, dim: int) -> None:
        existing = self._dim()
        if existing is None:
            self.db.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
                f"chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{dim}] distance_metric=cosine)"
            )
            self.db.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('dim', ?)", (str(dim),)
            )
            self.db.commit()
        elif existing != dim:
            raise RuntimeError(
                f"Embedding dimension changed ({existing} -> {dim}). The embed model "
                f"differs from the one this index was built with. Re-run `vaultrag index "
                f"--rebuild`."
            )

    # -- indexing ------------------------------------------------------------

    def _delete_note(self, rel_path: str) -> None:
        ids = [
            r["id"]
            for r in self.db.execute(
                "SELECT id FROM chunks WHERE rel_path = ?", (rel_path,)
            )
        ]
        if ids:
            self.db.executemany("DELETE FROM vec_chunks WHERE chunk_id = ?", [(i,) for i in ids])
        self.db.execute("DELETE FROM chunks WHERE rel_path = ?", (rel_path,))
        self.db.execute("DELETE FROM notes WHERE rel_path = ?", (rel_path,))

    def index_notes(
        self,
        notes: list[Note],
        embedder: Embedder,
        *,
        rebuild: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, int]:
        """Incrementally sync the index to `notes`. Returns a stats dict."""
        if rebuild:
            self.db.executescript(
                "DROP TABLE IF EXISTS vec_chunks; DELETE FROM chunks; "
                "DELETE FROM notes; DELETE FROM meta;"
            )
            self.db.commit()

        existing = {
            r["rel_path"]: r["content_hash"]
            for r in self.db.execute("SELECT rel_path, content_hash FROM notes")
        }
        current = {n.rel_path for n in notes}
        stats = {"added": 0, "updated": 0, "unchanged": 0, "removed": 0, "chunks": 0}

        # Remove notes that vanished from the vault.
        for gone in existing.keys() - current:
            self._delete_note(gone)
            stats["removed"] += 1

        for note in notes:
            prior = existing.get(note.rel_path)
            if prior == note.content_hash:
                stats["unchanged"] += 1
                continue

            chunks = chunk_note(note)
            if not chunks:
                # Still record the note so we don't re-process it every run.
                self._delete_note(note.rel_path)
                self._insert_note_row(note)
                stats["updated" if prior else "added"] += 1
                continue

            vectors = embedder.embed([c.text for c in chunks])
            self._ensure_vec_table(len(vectors[0]))

            self._delete_note(note.rel_path)
            self._insert_note_row(note)
            for chunk, vec in zip(chunks, vectors):
                cur = self.db.execute(
                    "INSERT INTO chunks(rel_path, title, heading, text, ordinal, tags_json) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        chunk.rel_path,
                        chunk.title,
                        chunk.heading,
                        chunk.text,
                        chunk.ordinal,
                        json.dumps(chunk.tags),
                    ),
                )
                self.db.execute(
                    "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                    (cur.lastrowid, sqlite_vec.serialize_float32(vec)),
                )
                stats["chunks"] += 1

            stats["updated" if prior else "added"] += 1
            if progress:
                progress(note.rel_path)

        self.db.commit()
        return stats

    def _insert_note_row(self, note: Note) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO notes(rel_path, title, mtime, content_hash, "
            "tags_json, links_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                note.rel_path,
                note.title,
                note.mtime,
                note.content_hash,
                json.dumps(note.tags),
                json.dumps(note.links),
            ),
        )

    # -- search --------------------------------------------------------------

    def search(self, query_vec: list[float], k: int) -> list[Hit]:
        if self._dim() is None:
            return []
        rows = self.db.execute(
            """
            SELECT c.rel_path, c.title, c.heading, c.text, c.ordinal, c.tags_json, v.distance
            FROM vec_chunks v
            JOIN chunks c ON c.id = v.chunk_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (sqlite_vec.serialize_float32(query_vec), k),
        ).fetchall()
        hits: list[Hit] = []
        for r in rows:
            chunk = Chunk(
                rel_path=r["rel_path"],
                title=r["title"],
                heading=r["heading"],
                text=r["text"],
                ordinal=r["ordinal"],
                tags=json.loads(r["tags_json"] or "[]"),
            )
            hits.append(Hit(chunk=chunk, distance=r["distance"]))
        return hits

    def chunks_for_notes(self, titles: list[str]) -> list[Chunk]:
        """Fetch chunks whose source note title is in `titles` (for link expansion)."""
        if not titles:
            return []
        placeholders = ",".join("?" for _ in titles)
        rows = self.db.execute(
            f"SELECT rel_path, title, heading, text, ordinal, tags_json "
            f"FROM chunks WHERE title IN ({placeholders}) ORDER BY rel_path, ordinal",
            titles,
        ).fetchall()
        return [
            Chunk(
                rel_path=r["rel_path"],
                title=r["title"],
                heading=r["heading"],
                text=r["text"],
                ordinal=r["ordinal"],
                tags=json.loads(r["tags_json"] or "[]"),
            )
            for r in rows
        ]

    def links_of(self, rel_paths: list[str]) -> list[str]:
        """Return the union of wikilink targets declared by the given notes."""
        if not rel_paths:
            return []
        placeholders = ",".join("?" for _ in rel_paths)
        rows = self.db.execute(
            f"SELECT links_json FROM notes WHERE rel_path IN ({placeholders})", rel_paths
        ).fetchall()
        out: dict[str, None] = {}
        for r in rows:
            for link in json.loads(r["links_json"] or "[]"):
                out.setdefault(link, None)
        return list(out)

    def stats(self) -> dict[str, int]:
        n_notes = self.db.execute("SELECT COUNT(*) AS c FROM notes").fetchone()["c"]
        n_chunks = self.db.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
        return {"notes": n_notes, "chunks": n_chunks, "dim": self._dim() or 0}

    def close(self) -> None:
        self.db.close()
