"""Heading-aware chunking.

Notes here are short (mostly 1-10 KB) and structured by `##`/`###` headings.
We split on headings, keep a title + heading prefix on each chunk so the
embedding has context, and merge tiny sections so we don't embed one-liners.
"""

from __future__ import annotations

import re

from .models import Chunk, Note

# ~1800 chars is roughly 400-500 tokens for German/English prose. Good enough
# without pulling a tokenizer dependency; oversized sections are split on blanks.
_MAX_CHARS = 1800
_MIN_CHARS = 200

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_by_heading(body: str) -> list[tuple[str, str]]:
    """Return (heading, text) sections. The pre-heading intro has heading ''."""
    sections: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []
    for line in body.splitlines():
        m = _HEADING.match(line)
        if m:
            if buf:
                sections.append((heading, "\n".join(buf).strip()))
                buf = []
            heading = m.group(2).strip()
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf).strip()))
    return [(h, t) for h, t in sections if t]


def _pack(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Merge adjacent small sections and hard-split oversized ones."""
    packed: list[tuple[str, str]] = []
    for heading, text in sections:
        if len(text) <= _MAX_CHARS:
            packed.append((heading, text))
            continue
        # Oversized: split on blank lines, greedily filling up to _MAX_CHARS.
        block: list[str] = []
        size = 0
        for para in re.split(r"\n\s*\n", text):
            if size + len(para) > _MAX_CHARS and block:
                packed.append((heading, "\n\n".join(block)))
                block, size = [], 0
            block.append(para)
            size += len(para)
        if block:
            packed.append((heading, "\n\n".join(block)))

    # Merge consecutive tiny chunks that share a heading context.
    merged: list[tuple[str, str]] = []
    for heading, text in packed:
        if merged and len(merged[-1][1]) < _MIN_CHARS and merged[-1][0] == heading:
            prev_h, prev_t = merged.pop()
            merged.append((prev_h, f"{prev_t}\n\n{text}"))
        else:
            merged.append((heading, text))
    return merged


def chunk_note(note: Note) -> list[Chunk]:
    """Turn a Note into embeddable chunks."""
    sections = _split_by_heading(note.body)
    if not sections:
        return []
    chunks: list[Chunk] = []
    for ordinal, (heading, text) in enumerate(_pack(sections)):
        prefix = note.title if not heading else f"{note.title} > {heading}"
        chunks.append(
            Chunk(
                rel_path=note.rel_path,
                title=note.title,
                heading=heading,
                text=f"{prefix}\n\n{text}",
                ordinal=ordinal,
                tags=list(note.tags),
            )
        )
    return chunks
