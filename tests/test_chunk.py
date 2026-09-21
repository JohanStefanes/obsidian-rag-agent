from __future__ import annotations

from vaultrag.chunk import chunk_note
from vaultrag.models import Note


def _note(body: str) -> Note:
    return Note(rel_path="n.md", title="Note", frontmatter={}, body=body, tags=["t"])


def test_splits_on_headings_and_prefixes_title() -> None:
    note = _note("## Alpha\nfirst section text here\n\n## Beta\nsecond section text here\n")
    chunks = chunk_note(note)
    headings = {c.heading for c in chunks}
    assert headings == {"Alpha", "Beta"}
    assert all(c.text.startswith("Note > ") for c in chunks)
    assert all(c.tags == ["t"] for c in chunks)


def test_oversized_section_is_split() -> None:
    big = "\n\n".join(["paragraph " * 60 for _ in range(6)])  # well over the char budget
    chunks = chunk_note(_note(f"## Big\n{big}\n"))
    assert len(chunks) >= 2


def test_empty_body_yields_no_chunks() -> None:
    assert chunk_note(_note("   \n\n")) == []
