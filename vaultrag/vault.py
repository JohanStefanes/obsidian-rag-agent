"""Walk an Obsidian vault and turn hand-authored notes into Note objects.

Hard-excludes generated / config directories (Graphify, .obsidian, ...) and
skips empty stub files, so only the real "second brain" is indexed.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterator

import frontmatter

from .config import EXCLUDE_DIRS
from .models import Note

_WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")


def _normalise_tags(value: object) -> list[str]:
    """Frontmatter tags come as a list or a comma/space string; normalise to list[str]."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,\s]+", value.strip())
        return [p.lstrip("#") for p in parts if p]
    if isinstance(value, (list, tuple)):
        return [str(v).lstrip("#") for v in value if str(v).strip()]
    return [str(value)]


def _extract_links(body: str) -> list[str]:
    """Return the distinct wikilink targets (without heading/alias parts)."""
    seen: dict[str, None] = {}
    for match in _WIKILINK.finditer(body):
        target = match.group(1).strip()
        if target:
            seen.setdefault(target, None)
    return list(seen)


def iter_note_paths(vault_path: Path) -> Iterator[Path]:
    """Yield paths to indexable .md files under the vault."""
    for path in vault_path.rglob("*.md"):
        # Skip anything inside an excluded directory.
        if any(part in EXCLUDE_DIRS for part in path.relative_to(vault_path).parts[:-1]):
            continue
        try:
            if path.stat().st_size == 0:  # 0-byte stub notes
                continue
        except OSError:
            continue
        yield path


def load_note(path: Path, vault_path: Path) -> Note | None:
    """Parse one markdown file into a Note. Returns None if it cannot be read."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    content_hash = hashlib.sha256(raw).hexdigest()
    try:
        post = frontmatter.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, Exception):  # noqa: BLE001 - frontmatter can raise broadly
        return None

    body = post.content or ""
    return Note(
        rel_path=str(path.relative_to(vault_path)),
        title=path.stem,
        frontmatter=dict(post.metadata),
        body=body,
        tags=_normalise_tags(post.metadata.get("tags")),
        links=_extract_links(body),
        mtime=path.stat().st_mtime,
        content_hash=content_hash,
    )


def load_vault(vault_path: Path) -> list[Note]:
    """Load every indexable note in the vault."""
    if not vault_path.is_dir():
        raise FileNotFoundError(f"Vault path not found: {vault_path}")
    notes: list[Note] = []
    for path in iter_note_paths(vault_path):
        note = load_note(path, vault_path)
        if note is not None:
            notes.append(note)
    return notes
