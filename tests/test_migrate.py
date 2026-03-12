"""Tests for longterm/migrate.py — JSONL→SQLite migration + rebuild."""
import json
import os
import sqlite3

import pytest

from agentrecall.core.schema import ensure_schema, get_connection
from agentrecall.core.embeddings import pack_embedding, unpack_embedding, EMBEDDING_DIMS
from agentrecall.longterm.migrate import load_jsonl, run_migrate, run_rebuild
from conftest import fake_embedding


def _make_jsonl(memory_dir, role, category, entries):
    """Helper: write a JSONL file at memory_dir/role/category.jsonl."""
    role_dir = os.path.join(memory_dir, role)
    os.makedirs(role_dir, exist_ok=True)
    path = os.path.join(role_dir, f"{category}.jsonl")
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return path


def _null_batch_embed(texts):
    """Batch embed that returns None (no API key)."""
    return None


def _fake_batch_embed(texts):
    """Batch embed returning deterministic vectors per text."""
    return [fake_embedding(seed=0.1 * (i + 1)) for i in range(len(texts))]


class TestLoadJsonl:
    def test_load_valid_entries(self, tmp_path):
        path = str(tmp_path / "data.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"text": "one"}) + "\n")
            f.write(json.dumps({"text": "two", "tags": ["a"]}) + "\n")

        entries = load_jsonl(path)
        assert len(entries) == 2
        assert entries[0]["text"] == "one"
        assert entries[1]["tags"] == ["a"]

    def test_load_skips_malformed_lines(self, tmp_path):
        path = str(tmp_path / "data.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"text": "good"}) + "\n")
            f.write("not json at all\n")
            f.write(json.dumps({"text": "also good"}) + "\n")

        entries = load_jsonl(path)
        assert len(entries) == 2

    def test_load_skips_blank_lines(self, tmp_path):
        path = str(tmp_path / "data.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"text": "one"}) + "\n")
            f.write("\n")
            f.write("   \n")
            f.write(json.dumps({"text": "two"}) + "\n")

        entries = load_jsonl(path)
        assert len(entries) == 2

    def test_load_missing_file(self):
        entries = load_jsonl("/nonexistent/path.jsonl")
        assert entries == []


class TestRunMigrate:
    def test_migrate_creates_entries(self, memory_dir, tmp_db, monkeypatch):
        """JSONL files → SQLite entries created."""
        monkeypatch.setattr(
            "agentrecall.longterm.migrate.get_embeddings_batch",
            _null_batch_embed,
        )

        _make_jsonl(memory_dir, "coder", "gotchas", [
            {"text": "kamal spawns new container", "tags": ["kamal"]},
            {"text": "sqlite wal corruption risk", "tags": ["sqlite"]},
        ])

        result = run_migrate(memory_dir, db_path=tmp_db)
        assert result == 0

        conn = sqlite3.connect(tmp_db)
        rows = conn.execute(
            "SELECT role, category, text, tags FROM entries ORDER BY id"
        ).fetchall()
        conn.close()

        assert len(rows) == 2
        assert rows[0][0] == "coder"
        assert rows[0][1] == "gotchas"
        assert rows[0][2] == "kamal spawns new container"
        assert json.loads(rows[0][3]) == ["kamal"]
        assert rows[1][2] == "sqlite wal corruption risk"

    def test_migrate_multiple_roles(self, memory_dir, tmp_db, monkeypatch):
        """Multiple role dirs each create entries under correct role."""
        monkeypatch.setattr(
            "agentrecall.longterm.migrate.get_embeddings_batch",
            _null_batch_embed,
        )

        _make_jsonl(memory_dir, "coder", "gotchas", [
            {"text": "entry from coder"},
        ])
        _make_jsonl(memory_dir, "social", "exhausted", [
            {"text": "entry from social"},
        ])

        run_migrate(memory_dir, db_path=tmp_db)

        conn = sqlite3.connect(tmp_db)
        rows = conn.execute(
            "SELECT role, text FROM entries ORDER BY role"
        ).fetchall()
        conn.close()

        assert len(rows) == 2
        assert rows[0] == ("coder", "entry from coder")
        assert rows[1] == ("social", "entry from social")

    def test_migrate_preserves_created_at(self, memory_dir, tmp_db, monkeypatch):
        monkeypatch.setattr(
            "agentrecall.longterm.migrate.get_embeddings_batch",
            _null_batch_embed,
        )

        _make_jsonl(memory_dir, "coder", "fixes", [
            {"text": "old entry", "created_at": "2025-01-15"},
        ])

        run_migrate(memory_dir, db_path=tmp_db)

        conn = sqlite3.connect(tmp_db)
        row = conn.execute("SELECT created_at FROM entries").fetchone()
        conn.close()

        assert row[0] == "2025-01-15"

    def test_migrate_with_embeddings(self, memory_dir, tmp_db, monkeypatch):
        """When embed function returns vectors, they're stored as blobs."""
        monkeypatch.setattr(
            "agentrecall.longterm.migrate.get_embeddings_batch",
            _fake_batch_embed,
        )

        _make_jsonl(memory_dir, "coder", "gotchas", [
            {"text": "test embedding storage"},
        ])

        run_migrate(memory_dir, db_path=tmp_db)

        conn = sqlite3.connect(tmp_db)
        row = conn.execute("SELECT embedding FROM entries").fetchone()
        conn.close()

        assert row[0] is not None
        vec = unpack_embedding(row[0])
        assert len(vec) == EMBEDDING_DIMS

    def test_migrate_no_jsonl_files(self, memory_dir, tmp_db, capsys):
        """Empty directory → returns 0, prints message."""
        result = run_migrate(memory_dir, db_path=tmp_db)
        assert result == 0

        output = capsys.readouterr().out
        assert "No JSONL files" in output

    def test_migrate_skips_root_level_files(self, memory_dir, tmp_db, monkeypatch):
        """JSONL files at root (no role dir) are skipped (< 2 path parts)."""
        monkeypatch.setattr(
            "agentrecall.longterm.migrate.get_embeddings_batch",
            _null_batch_embed,
        )

        path = os.path.join(memory_dir, "orphan.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"text": "should be skipped"}) + "\n")

        run_migrate(memory_dir, db_path=tmp_db)

        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        conn.close()
        assert count == 0


class TestRunMigrateDryRun:
    def test_dry_run_no_writes(self, memory_dir, tmp_db, monkeypatch, capsys):
        """--dry-run shows what would be migrated without writing."""
        monkeypatch.setattr(
            "agentrecall.longterm.migrate.get_embeddings_batch",
            _null_batch_embed,
        )

        _make_jsonl(memory_dir, "coder", "gotchas", [
            {"text": "entry one"},
            {"text": "entry two"},
        ])

        result = run_migrate(memory_dir, dry_run=True, db_path=tmp_db)
        assert result == 0

        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        conn.close()
        assert count == 0

        output = capsys.readouterr().out
        assert "dry-run" in output
        assert "2 entries" in output

    def test_dry_run_reports_summary(self, memory_dir, tmp_db, monkeypatch, capsys):
        """Dry run still prints summary with file/entry counts."""
        monkeypatch.setattr(
            "agentrecall.longterm.migrate.get_embeddings_batch",
            _null_batch_embed,
        )

        _make_jsonl(memory_dir, "coder", "gotchas", [{"text": "a"}])
        _make_jsonl(memory_dir, "social", "stories", [{"text": "b"}, {"text": "c"}])

        run_migrate(memory_dir, dry_run=True, db_path=tmp_db)

        output = capsys.readouterr().out
        assert "Migration Summary" in output
        assert "Total entries: 3" in output


class TestRunMigrateSkipsDuplicates:
    def test_skips_exact_duplicate_text(self, memory_dir, tmp_db, monkeypatch):
        """run_migrate skips entries already in DB with same role/category/text."""
        monkeypatch.setattr(
            "agentrecall.longterm.migrate.get_embeddings_batch",
            _null_batch_embed,
        )

        _make_jsonl(memory_dir, "coder", "gotchas", [
            {"text": "already exists"},
            {"text": "new entry"},
        ])

        # Pre-insert one entry
        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "INSERT INTO entries (role, category, text, tags, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("coder", "gotchas", "already exists", "[]", "2025-01-01"),
        )
        conn.commit()
        conn.close()

        run_migrate(memory_dir, db_path=tmp_db)

        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        conn.close()
        assert count == 2  # 1 pre-existing + 1 new (duplicate skipped)

    def test_skips_duplicate_reports_count(self, memory_dir, tmp_db, monkeypatch, capsys):
        """Summary shows skipped count."""
        monkeypatch.setattr(
            "agentrecall.longterm.migrate.get_embeddings_batch",
            _null_batch_embed,
        )

        _make_jsonl(memory_dir, "coder", "gotchas", [
            {"text": "dup1"},
            {"text": "dup2"},
            {"text": "new"},
        ])

        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "INSERT INTO entries (role, category, text, tags, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("coder", "gotchas", "dup1", "[]", "2025-01-01"),
        )
        conn.execute(
            "INSERT INTO entries (role, category, text, tags, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("coder", "gotchas", "dup2", "[]", "2025-01-01"),
        )
        conn.commit()
        conn.close()

        run_migrate(memory_dir, db_path=tmp_db)

        output = capsys.readouterr().out
        assert "Skipped: 2" in output

    def test_rerun_migration_is_idempotent(self, memory_dir, tmp_db, monkeypatch):
        """Running migrate twice on same data creates no duplicates."""
        monkeypatch.setattr(
            "agentrecall.longterm.migrate.get_embeddings_batch",
            _null_batch_embed,
        )

        _make_jsonl(memory_dir, "coder", "gotchas", [
            {"text": "entry one"},
            {"text": "entry two"},
        ])

        run_migrate(memory_dir, db_path=tmp_db)
        run_migrate(memory_dir, db_path=tmp_db)

        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        conn.close()
        assert count == 2


class TestRunRebuild:
    def test_rebuild_embeds_null_entries(self, tmp_db, monkeypatch):
        """run_rebuild re-embeds entries with NULL embeddings."""
        monkeypatch.setattr(
            "agentrecall.longterm.migrate.get_embeddings_batch",
            _fake_batch_embed,
        )

        # Insert entries without embeddings
        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "INSERT INTO entries (role, category, text, embedding, tags, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("coder", "gotchas", "needs embedding", None, "[]", "2025-01-01"),
        )
        conn.execute(
            "INSERT INTO entries (role, category, text, embedding, tags, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("social", "stories", "also needs embedding", None, "[]", "2025-01-02"),
        )
        conn.commit()
        conn.close()

        result = run_rebuild(db_path=tmp_db)
        assert result == 0

        conn = sqlite3.connect(tmp_db)
        rows = conn.execute(
            "SELECT embedding FROM entries ORDER BY id"
        ).fetchall()
        conn.close()

        for row in rows:
            assert row[0] is not None
            vec = unpack_embedding(row[0])
            assert len(vec) == EMBEDDING_DIMS

    def test_rebuild_skips_already_embedded(self, tmp_db, monkeypatch):
        """Entries with existing embeddings are untouched."""
        call_count = {"n": 0}
        original_embed = _fake_batch_embed

        def counting_embed(texts):
            call_count["n"] += 1
            return original_embed(texts)

        monkeypatch.setattr(
            "agentrecall.longterm.migrate.get_embeddings_batch",
            counting_embed,
        )

        # Insert one with embedding, one without
        existing_emb = pack_embedding(fake_embedding(seed=0.5))
        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "INSERT INTO entries (role, category, text, embedding, tags, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("coder", "gotchas", "has embedding", existing_emb, "[]", "2025-01-01"),
        )
        conn.execute(
            "INSERT INTO entries (role, category, text, embedding, tags, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("coder", "gotchas", "null embedding", None, "[]", "2025-01-02"),
        )
        conn.commit()
        conn.close()

        run_rebuild(db_path=tmp_db)

        # Only called once — for the 1 NULL entry
        assert call_count["n"] == 1

        conn = sqlite3.connect(tmp_db)
        rows = conn.execute(
            "SELECT text, embedding FROM entries ORDER BY id"
        ).fetchall()
        conn.close()

        # First entry embedding unchanged (compare blobs to avoid float precision)
        assert rows[0][1] == existing_emb
        # Second entry now has embedding
        assert rows[1][1] is not None

    def test_rebuild_no_null_entries(self, tmp_db, capsys):
        """All entries already embedded → prints message, returns 0."""
        existing_emb = pack_embedding(fake_embedding(seed=0.5))
        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "INSERT INTO entries (role, category, text, embedding, tags, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("coder", "gotchas", "fully embedded", existing_emb, "[]", "2025-01-01"),
        )
        conn.commit()
        conn.close()

        result = run_rebuild(db_path=tmp_db)
        assert result == 0

        output = capsys.readouterr().out
        assert "All entries have embeddings" in output

    def test_rebuild_returns_1_on_embed_failure(self, tmp_db, monkeypatch):
        """Embedding API failure → returns 1."""
        def failing_embed(texts):
            raise RuntimeError("API down")

        monkeypatch.setattr(
            "agentrecall.longterm.migrate.get_embeddings_batch",
            failing_embed,
        )

        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "INSERT INTO entries (role, category, text, embedding, tags, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("coder", "gotchas", "needs embedding", None, "[]", "2025-01-01"),
        )
        conn.commit()
        conn.close()

        result = run_rebuild(db_path=tmp_db)
        assert result == 1

    def test_rebuild_returns_1_when_embed_returns_none(self, tmp_db, monkeypatch):
        """No API key (embed returns None) → returns 1."""
        monkeypatch.setattr(
            "agentrecall.longterm.migrate.get_embeddings_batch",
            _null_batch_embed,
        )

        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "INSERT INTO entries (role, category, text, embedding, tags, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("coder", "gotchas", "needs embedding", None, "[]", "2025-01-01"),
        )
        conn.commit()
        conn.close()

        result = run_rebuild(db_path=tmp_db)
        assert result == 1


class TestRunRebuildDryRun:
    def test_dry_run_no_writes(self, tmp_db, capsys):
        """--dry-run shows count without writing embeddings."""
        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "INSERT INTO entries (role, category, text, embedding, tags, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("coder", "gotchas", "null entry", None, "[]", "2025-01-01"),
        )
        conn.commit()
        conn.close()

        result = run_rebuild(dry_run=True, db_path=tmp_db)
        assert result == 0

        # Verify no embedding was written
        conn = sqlite3.connect(tmp_db)
        row = conn.execute("SELECT embedding FROM entries").fetchone()
        conn.close()
        assert row[0] is None

        output = capsys.readouterr().out
        assert "dry-run" in output
        assert "1 entries" in output

    def test_dry_run_reports_correct_count(self, tmp_db, capsys):
        """Dry run reports exact number of NULL-embedding entries."""
        conn = sqlite3.connect(tmp_db)
        for i in range(5):
            conn.execute(
                "INSERT INTO entries (role, category, text, embedding, tags, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("r", "c", f"entry {i}", None, "[]", "2025-01-01"),
            )
        existing_emb = pack_embedding(fake_embedding(seed=0.5))
        conn.execute(
            "INSERT INTO entries (role, category, text, embedding, tags, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("r", "c", "has embedding", existing_emb, "[]", "2025-01-01"),
        )
        conn.commit()
        conn.close()

        run_rebuild(dry_run=True, db_path=tmp_db)

        output = capsys.readouterr().out
        assert "5 entries" in output
