from __future__ import annotations

from pathlib import Path

from vaultrag.vault import load_vault


def test_excludes_graphify_and_stubs(vault: Path) -> None:
    notes = load_vault(vault)
    paths = {n.rel_path for n in notes}
    assert not any("Graphify" in p for p in paths), "Graphify must be excluded"
    assert "Cubbo.md" not in paths, "0-byte stub must be skipped"
    assert len(notes) == 3


def test_frontmatter_tags_and_links(vault: Path) -> None:
    notes = {n.title: n for n in load_vault(vault)}
    m183 = notes["M183 Applikationssicherheit"]
    assert m183.frontmatter["module_code"] == "M183"
    assert "school-module" in m183.tags and "security" in m183.tags
    assert "00_Index" in m183.links
    assert m183.content_hash  # populated


def test_wikilink_alias_and_heading_stripped(vault: Path) -> None:
    notes = {n.title: n for n in load_vault(vault)}
    lj = notes["LJ-KW16-2026"]
    assert lj.links == ["M183 Applikationssicherheit"]
