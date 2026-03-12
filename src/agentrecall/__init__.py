"""AgentRecall — Persistent two-tier memory for AI agents."""

__version__ = "0.1.0"

from agentrecall.core.store import MemoryStore, DuplicateError
from agentrecall.core.search import MemorySearch
from agentrecall.core.embeddings import cosine_similarity, get_embedding
from agentrecall.core.result import Result
from agentrecall.core.schema import ensure_schema, get_connection

__all__ = [
    "MemoryStore",
    "MemorySearch",
    "DuplicateError",
    "cosine_similarity",
    "get_embedding",
    "Result",
    "ensure_schema",
    "get_connection",
]
