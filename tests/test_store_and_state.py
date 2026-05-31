"""Tests for ResearchGoal, state initialisation, and the SQLite document store."""
import sqlite3
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from mariana.state import ResearchGoal, _parse_duration


# ── ResearchGoal ──────────────────────────────────────────────────────────────

def test_research_goal_defaults():
    g = ResearchGoal()
    assert g.max_runtime is None
    assert g.max_sources is None
    assert g.target_words is None
    assert g.required_topics == []
    assert g.max_depth == 2


def test_research_goal_to_dict():
    g = ResearchGoal(max_runtime=timedelta(hours=1), max_sources=50, target_words=3000)
    d = g.to_dict()
    assert d["max_runtime_seconds"] == 3600
    assert d["max_sources"] == 50
    assert d["target_words"] == 3000


def test_parse_duration_hours():
    assert _parse_duration("2h") == timedelta(hours=2)


def test_parse_duration_minutes():
    assert _parse_duration("30m") == timedelta(minutes=30)


def test_parse_duration_combined():
    assert _parse_duration("1h30m") == timedelta(hours=1, minutes=30)


def test_parse_duration_none():
    assert _parse_duration(None) is None


def test_parse_duration_invalid_returns_none():
    assert _parse_duration("bad") is None


# ── SQLite document store ─────────────────────────────────────────────────────

@pytest.fixture()
def tmp_store(monkeypatch, tmp_path):
    """Point the store at a temp directory for isolation."""
    import mariana.utils.store as store_mod
    store_mod._DB_PATH = tmp_path / "test.db"
    yield store_mod
    store_mod._DB_PATH = None


def test_init_db_creates_tables(tmp_store):
    tmp_store.init_db()
    with tmp_store.get_db() as db:
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert {"documents", "toc_nodes", "section_content", "search_queue"}.issubset(tables)


def test_create_and_get_document(tmp_store):
    tmp_store.init_db()
    doc_id = tmp_store.create_document("test query", "Test Title")
    doc = tmp_store.get_document(doc_id)
    assert doc is not None
    assert doc["query"] == "test query"
    assert doc["title"] == "Test Title"


def test_create_toc_node(tmp_store):
    tmp_store.init_db()
    doc_id = tmp_store.create_document("query")
    node_id = tmp_store.create_toc_node(doc_id, "Section 1", order_index=0)
    toc = tmp_store.get_toc(doc_id)
    assert len(toc) == 1
    assert toc[0]["title"] == "Section 1"
    assert toc[0]["id"] == node_id


def test_save_and_load_section_content(tmp_store):
    tmp_store.init_db()
    doc_id = tmp_store.create_document("query")
    node_id = tmp_store.create_toc_node(doc_id, "Section 1")
    tmp_store.save_section_content(node_id, "Hello world this is content")
    loaded = tmp_store.load_section_content(node_id)
    assert loaded == "Hello world this is content"


def test_update_node_summary(tmp_store):
    tmp_store.init_db()
    doc_id = tmp_store.create_document("query")
    node_id = tmp_store.create_toc_node(doc_id, "Section")
    tmp_store.update_node_summary(node_id, "A key finding here.")
    toc = tmp_store.get_toc(doc_id)
    assert toc[0]["summary"] == "A key finding here."


def test_build_collapsed_toc(tmp_store):
    tmp_store.init_db()
    doc_id = tmp_store.create_document("query")
    n1 = tmp_store.create_toc_node(doc_id, "Alpha", order_index=0)
    n2 = tmp_store.create_toc_node(doc_id, "Beta", order_index=1)
    tmp_store.update_node_status(n1, "complete")
    tmp_store.update_node_summary(n1, "Alpha covers solar panels.")
    result = tmp_store.build_collapsed_toc(doc_id, current_node_id=n2)
    assert "Alpha" in result
    assert "Beta" in result
    assert "WRITING NOW" in result


def test_word_count_rolls_up_to_document(tmp_store):
    tmp_store.init_db()
    doc_id = tmp_store.create_document("query")
    node_id = tmp_store.create_toc_node(doc_id, "Section 1")
    tmp_store.save_section_content(node_id, "one two three four five")
    doc = tmp_store.get_document(doc_id)
    assert doc["total_words"] == 5
