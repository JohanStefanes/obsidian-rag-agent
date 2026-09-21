# Obsidian Vault RAG Agent

Ask your Obsidian notes questions in plain language and get answers with links back to
the exact notes they came from. Everything runs **on your own machine** through
[Ollama](https://ollama.com) — no cloud, no API keys, no data leaving your laptop.

Built as a portfolio project. There is a command line, a web page, and a double-click
macOS app, all backed by the same small Python core.

```
You:  Was habe ich in M183 zu Applikationssicherheit gemacht?
App:  Du hast die VulnerApp abgesichert: Login, Session-Auth und Rollen (RBAC) ...
      Quellen:
        - School Modules/M183 Applikationssicherheit.md
        - Current Projects.md
```

## What it does

- **Reads your vault** (Markdown notes, tags, and `[[links]]`), ignoring generated
  folders like `Graphify/`.
- **Finds the relevant notes** for a question using local embeddings + a vector search.
- **Answers in the question's language** (the vault is mostly German) and always cites
  its sources so you can verify and jump straight to the note.

## How it works (in one picture)

```
your notes ──▶ split into chunks ──▶ embed (bge-m3) ──▶ store in SQLite (sqlite-vec)
                                                                  │
your question ──▶ embed ──▶ find the closest chunks ◀─────────────┘
                                     │
                                     ▼
                        local LLM writes the answer + cites [[notes]]
```

That is the whole idea: turn notes into vectors, find the closest ones to your question,
and let a local model answer using only those notes.

## Quick start

You need [`uv`](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).

```bash
# 1. Get the models (one time)
ollama pull bge-m3                 # turns text into vectors (multilingual)
ollama pull qwen2.5:7b-instruct    # writes the answers (see "Which model" below)

# 2. Install and point it at your vault
uv sync
cp .env.example .env               # edit VAULT_PATH inside .env

# 3. Build the index, then ask
uv run vaultrag index
uv run vaultrag ask "your question here"
```

## Three ways to use it

**Command line**
```bash
uv run vaultrag ask "Was habe ich diese Woche gemacht?"
uv run vaultrag status             # how many notes/chunks are indexed
```

**Web page** (a simple chat UI in your browser)
```bash
uv sync --extra web
uv run vaultrag serve              # open http://127.0.0.1:8000
```

**Mac app** (a real window you can double-click)
```bash
uv sync --extra web --extra app
bash packaging/make_app.sh         # builds dist/VaultRAG.app
open dist/VaultRAG.app             # or drag it into /Applications
```

In the web page and the app, answers stream in live and every source is a link that
opens the note in Obsidian.

## The code

Small on purpose — each file does one thing:

| File | Job |
|------|-----|
| `vaultrag/vault.py` | read the vault: notes, frontmatter, tags, `[[links]]` |
| `vaultrag/chunk.py` | split notes into bite-size pieces |
| `vaultrag/embed.py` | turn text into vectors with Ollama |
| `vaultrag/store.py` | save/search vectors in one SQLite file |
| `vaultrag/retrieve.py` | find the closest pieces to a question |
| `vaultrag/answer.py` | ask the local model, cite the sources |
| `vaultrag/cli.py` | the `index` / `ask` / `status` / `serve` / `app` commands |
| `vaultrag/server.py` | the web page + streaming API |
| `vaultrag/desktop.py` | the native Mac window |

## Which model

Set the chat model in `.env` (`CHAT_MODEL=`). Bigger is smarter but slower; pick by how
much memory you have:

| Machine | Suggested chat model | Notes |
|---------|----------------------|-------|
| 8-16 GB | `qwen2.5:7b-instruct` | fast, solid German, the default |
| 24 GB+  | `qwen2.5:14b-instruct` | more complete answers, cleaner `[[citations]]`, ~12 tok/s on an M1 Pro |
| 32 GB+, want the ceiling | `qwen3:30b-a3b` | mixture-of-experts: smarter, stays fast |

Embeddings stay on `bge-m3` (multilingual) regardless. Any Ollama model works — these are
just the ones that did best here on German notes.

## Tests

```bash
uv run pytest                      # unit tests, no Ollama needed
uv run python evals/run_eval.py    # retrieval quality (hit@k) on a small question set
```

## Roadmap

- [x] **Phase 1** — ask your vault (done): CLI + web + Mac app.
- [ ] **Phase 2** — suggest tags and links for new notes, as a review diff.
- [ ] **Phase 3** — draft daily journal entries in the vault's own format.

## Why these choices

- **Local only:** the vault is private and mostly German, so on-device models keep it that
  way and cost nothing per question.
- **SQLite instead of a big vector database:** the corpus is small (~600 notes), so one
  file you can open and inspect is simpler and easier to trust.
- **No heavy frameworks:** the pipeline is short enough to read top to bottom.

Anything the app writes into the vault (later phases) is opt-in, shown for review first,
and never touches generated folders.
