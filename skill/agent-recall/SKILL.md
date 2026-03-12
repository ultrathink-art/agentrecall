---
name: agent-memory
description: Persistent two-tier memory for AI agents. Use when storing learnings, searching past work, checking memory health, or following the session start/end memory protocol.
allowed-tools: Bash(agentrecall *), Bash(python *), Read, Glob
---

# Agent Recall

Persistent two-tier memory for AI agents. Short-term markdown files (always loaded) + long-term SQLite with semantic search (queried on-demand).

## Setup

```bash
# Install (zero required deps — SQLite is stdlib)
pip install agentrecall

# Optional: enable semantic search/dedup
pip install agentrecall[embeddings]
export OPENAI_API_KEY="sk-..."

# Initialize memory directory
agentrecall init
```

## Quick Reference

### Store a memory
```bash
agentrecall store <role> <category> "text" --tags tag1,tag2
```

### Search memories
```bash
agentrecall search <role> <category> "query"
# Exit 0 = matches found, exit 1 = no matches
```

### List categories
```bash
agentrecall list <role>
```

### Check health
```bash
agentrecall check              # Short-term files
agentrecall check --long-term  # DB health
agentrecall check --all        # Both
agentrecall check --fix        # Auto-prune oversized files
```

## Protocol

1. **Session start**: Read `memory/<role>.md` — your accumulated knowledge
2. **Before acting**: `agentrecall search <role> <category> "concept"` to check past work
3. **After acting**: `agentrecall store <role> <category> "what you learned"`
4. **Session end**: Update `memory/<role>.md` with mistakes/learnings, run `agentrecall check`

## References

- [Memory Directive](references/memory-directive.md) — full agent protocol
- [Best Practices](references/best-practices.md) — patterns that work
- [Categories Guide](references/categories-guide.md) — suggested categories by role

## Scripts

- `scripts/store.py` — programmatic store
- `scripts/search.py` — programmatic search
- `scripts/check.py` — programmatic health check
- `scripts/setup.py` — initialize for a new project
