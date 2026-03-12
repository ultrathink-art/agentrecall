# Changelog

## 0.1.2 (2026-03-12)

Renamed PyPI package from `agentrecall-memory` to `agent-cerebro` (v0.1.0). Python import name (`from agentrecall import ...`) and CLI commands (`agentrecall`/`cerebro`) are unchanged.

## 0.1.1 (2026-03-12)

Renamed PyPI package from `agentrecall` to `agentrecall-memory` (PyPI rejected `agentrecall` as too similar to existing `agent-recall`). Python import name (`from agentrecall import ...`) and CLI commands (`agentrecall store/search/...`) are unchanged.

## 0.1.0 (2026-03-11)

Initial release (as agentrecall, renamed from agentmemory).

- Two-tier memory: short-term markdown files + long-term SQLite with OpenAI embeddings
- CLI: `agentrecall store/search/list/check/init/migrate`
- Semantic dedup via cosine similarity (threshold >0.92)
- Keyword fallback when no OpenAI API key
- Short-term memory file validation (80-line limit, session log pruning)
- JSONL to SQLite migration tool
- Agent Skills packaging (`skill/agent-recall/`)
- Zero required dependencies (SQLite is stdlib)
- Full pytest suite
