"""Runtime configuration, loaded from the environment / a local .env file.

Kept dependency-free (no python-dotenv): a tiny parser reads .env so the CLI
works with `uv run vaultrag ...` without extra setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Repo root = parent of the vaultrag package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories never walked when indexing (Graphify is 10k+ generated notes).
EXCLUDE_DIRS = frozenset({"Graphify", ".obsidian", ".git", ".trash", ".claude"})


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env file without overriding real env vars."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Config:
    vault_path: Path
    ollama_host: str
    embed_model: str
    chat_model: str
    db_path: Path
    top_k: int

    @classmethod
    def load(cls) -> "Config":
        _load_dotenv(REPO_ROOT / ".env")

        # Real path comes from .env / the environment; this is only a fallback.
        vault = os.environ.get("VAULT_PATH", str(Path.home() / "Obsidian"))
        db_raw = os.environ.get("DB_PATH", "data/vault-index.db")
        db_path = Path(db_raw)
        if not db_path.is_absolute():
            db_path = REPO_ROOT / db_path

        return cls(
            vault_path=Path(vault).expanduser(),
            ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            embed_model=os.environ.get("EMBED_MODEL", "bge-m3"),
            chat_model=os.environ.get("CHAT_MODEL", "qwen2.5:7b-instruct"),
            db_path=db_path,
            top_k=int(os.environ.get("TOP_K", "6")),
        )
