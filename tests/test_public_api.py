"""Tests that the public API (from agentmemory import ...) works correctly."""
import pytest

from agentmemory import DuplicateError, MemoryStore
from agentmemory.core.store import DuplicateError as CoreDuplicateError
from tests.conftest import null_embed_fn


class TestDuplicateErrorIdentity:
    def test_public_api_exports_same_class_as_core(self):
        assert DuplicateError is CoreDuplicateError

    def test_public_api_catches_store_duplicate(self, tmp_db):
        store = MemoryStore(db_path=tmp_db)
        store.store("r", "c", "some text", embed_fn=null_embed_fn)

        with pytest.raises(DuplicateError):
            store.store("r", "c", "some text", embed_fn=null_embed_fn)
        store.close()

    def test_duplicate_error_in_all(self):
        import agentmemory
        assert "DuplicateError" in agentmemory.__all__
