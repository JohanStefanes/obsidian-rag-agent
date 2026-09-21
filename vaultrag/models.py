"""Core data structures shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Note:
    """A single hand-authored vault note."""

    rel_path: str          # path relative to the vault root, e.g. "School Modules/M183 ....md"
    title: str             # filename stem, used as the wikilink target name
    frontmatter: dict      # parsed YAML frontmatter (may be empty)
    body: str              # markdown body with frontmatter stripped
    tags: list[str] = field(default_factory=list)   # frontmatter tags, normalised to str
    links: list[str] = field(default_factory=list)  # [[wikilink]] targets found in the body
    mtime: float = 0.0     # filesystem mtime, for incremental indexing
    content_hash: str = "" # sha256 of the raw file, for change detection


@dataclass
class Chunk:
    """A retrievable slice of a note."""

    rel_path: str          # source note
    title: str             # source note title
    heading: str           # heading path within the note, e.g. "Reflexion" or "" for the intro
    text: str              # chunk text (prefixed with title + heading for embedding context)
    ordinal: int           # position of the chunk within its note
    tags: list[str] = field(default_factory=list)


@dataclass
class Hit:
    """A retrieval result: a chunk plus its distance to the query."""

    chunk: Chunk
    distance: float        # cosine distance from sqlite-vec (lower = closer)
