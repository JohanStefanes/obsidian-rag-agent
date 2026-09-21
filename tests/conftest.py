"""Shared fixtures: a tiny on-disk vault and a deterministic fake embedder."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

VAULT_FILES = {
    "00_Index.md": "---\ntags: [index]\n---\n# Index\n\nLinks to [[M183 Applikationssicherheit]].\n",
    "School Modules/M183 Applikationssicherheit.md": (
        "---\n"
        "name: Applikationssicherheit\n"
        "module_code: M183\n"
        "tags:\n  - school-module\n  - security\n"
        "status: active\n"
        "---\n"
        "# M183 Applikationssicherheit\n\n"
        "## Kompetenz\nSichere Applikationen entwickeln. Related: [[00_Index]].\n\n"
        "## Projekt\nVulnerApp mit SQL-Injection und XSS Tests.\n"
    ),
    "Lernjournale/LJ-KW16-2026.md": (
        "---\nkw: 16\nyear: 2026\ntags: [lernjournal]\nmodules: [M183]\n---\n"
        "# LJ KW16\n\n## Montag\n**Taetigkeiten:** An [[M183 Applikationssicherheit]] gearbeitet.\n"
    ),
    # Excluded: must never be indexed.
    "Graphify/notch-usage/symbol_1.md": "---\ntags: [graphify/code]\n---\n# gen\nnoise\n",
    # Stub: zero bytes, must be skipped.
    "Cubbo.md": "",
}


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    for rel, content in VAULT_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return root


class FakeEmbedder:
    """Deterministic embeddings from a hash, so tests need no Ollama."""

    dim = 16

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            out.append([h[i % len(h)] / 255.0 for i in range(self.dim)])
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
