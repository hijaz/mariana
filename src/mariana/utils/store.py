"""
Persistent document store — SQLite-backed.
Tables: documents, toc_nodes, section_content, search_queue
"""
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from mariana.utils.config import load_config

_DB_PATH: Path | None = None


def _get_db_path() -> Path:
    global _DB_PATH
    if _DB_PATH is None:
        cfg = load_config()
        db_dir = Path(cfg.output_dir)
        db_dir.mkdir(parents=True, exist_ok=True)
        _DB_PATH = db_dir / "mariana.db"
    return _DB_PATH


@contextmanager
def get_db():
    db_path = _get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id           TEXT PRIMARY KEY,
                query        TEXT NOT NULL,
                title        TEXT NOT NULL DEFAULT '',
                created_at   TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
                total_words  INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS toc_nodes (
                id           TEXT PRIMARY KEY,
                document_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                parent_id    TEXT REFERENCES toc_nodes(id) ON DELETE CASCADE,
                title        TEXT NOT NULL,
                summary      TEXT NOT NULL DEFAULT '',
                status       TEXT NOT NULL DEFAULT 'pending',
                depth        INTEGER NOT NULL DEFAULT 0,
                order_index  INTEGER NOT NULL DEFAULT 0,
                word_count   INTEGER NOT NULL DEFAULT 0,
                created_at   TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS section_content (
                id           TEXT PRIMARY KEY,
                node_id      TEXT NOT NULL UNIQUE REFERENCES toc_nodes(id) ON DELETE CASCADE,
                content      TEXT NOT NULL DEFAULT '',
                updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS search_queue (
                id           TEXT PRIMARY KEY,
                document_id  TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                query        TEXT NOT NULL,
                node_id      TEXT REFERENCES toc_nodes(id) ON DELETE SET NULL,
                generation   INTEGER NOT NULL DEFAULT 0,
                priority     REAL NOT NULL DEFAULT 1.0,
                status       TEXT NOT NULL DEFAULT 'pending',
                created_at   TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_toc_document
                ON toc_nodes(document_id, parent_id, order_index);
            CREATE INDEX IF NOT EXISTS idx_toc_status
                ON toc_nodes(document_id, status);
            CREATE INDEX IF NOT EXISTS idx_queue_pending
                ON search_queue(document_id, status, priority DESC);
        """)


# ── Document helpers ──────────────────────────────────────────────────────────

def create_document(query: str, title: str = "") -> str:
    doc_id = str(uuid.uuid4())
    with get_db() as db:
        db.execute(
            "INSERT INTO documents (id, query, title) VALUES (?, ?, ?)",
            (doc_id, query, title or query),
        )
    return doc_id


def update_document_title(doc_id: str, title: str) -> None:
    with get_db() as db:
        db.execute(
            "UPDATE documents SET title=?, updated_at=datetime('now') WHERE id=?",
            (title, doc_id),
        )


def get_document(doc_id: str) -> dict | None:
    with get_db() as db:
        row = db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    return dict(row) if row else None


# ── ToC node helpers ──────────────────────────────────────────────────────────

def create_toc_node(
    doc_id: str,
    title: str,
    parent_id: str | None = None,
    depth: int = 0,
    order_index: int = 0,
) -> str:
    node_id = str(uuid.uuid4())
    with get_db() as db:
        db.execute(
            """INSERT INTO toc_nodes
               (id, document_id, parent_id, title, depth, order_index)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (node_id, doc_id, parent_id, title, depth, order_index),
        )
    return node_id


def update_node_status(node_id: str, status: str) -> None:
    with get_db() as db:
        db.execute(
            "UPDATE toc_nodes SET status=?, updated_at=datetime('now') WHERE id=?",
            (status, node_id),
        )


def update_node_summary(node_id: str, summary: str) -> None:
    with get_db() as db:
        db.execute(
            "UPDATE toc_nodes SET summary=?, updated_at=datetime('now') WHERE id=?",
            (summary, node_id),
        )


def get_toc(doc_id: str) -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            """SELECT * FROM toc_nodes
               WHERE document_id=?
               ORDER BY depth, order_index""",
            (doc_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Section content helpers ───────────────────────────────────────────────────

def save_section_content(node_id: str, content: str) -> None:
    word_count = len(content.split())
    with get_db() as db:
        db.execute(
            """INSERT INTO section_content (id, node_id, content, updated_at)
               VALUES (?, ?, ?, datetime('now'))
               ON CONFLICT(node_id) DO UPDATE
               SET content=excluded.content, updated_at=excluded.updated_at""",
            (str(uuid.uuid4()), node_id, content),
        )
        db.execute(
            "UPDATE toc_nodes SET word_count=?, updated_at=datetime('now') WHERE id=?",
            (word_count, node_id),
        )
        # Roll up total_words to parent document
        db.execute(
            """UPDATE documents SET
               total_words=(SELECT COALESCE(SUM(word_count),0) FROM toc_nodes WHERE document_id=documents.id),
               updated_at=datetime('now')
               WHERE id=(SELECT document_id FROM toc_nodes WHERE id=?)""",
            (node_id,),
        )


def load_section_content(node_id: str) -> str:
    with get_db() as db:
        row = db.execute(
            "SELECT content FROM section_content WHERE node_id=?", (node_id,)
        ).fetchone()
    return row["content"] if row else ""


# ── Search queue helpers ──────────────────────────────────────────────────────

def enqueue_query(
    doc_id: str,
    query: str,
    node_id: str | None,
    parent_id: str | None,
    generation: int = 0,
    priority: float = 1.0,
) -> str:
    entry_id = str(uuid.uuid4())
    with get_db() as db:
        db.execute(
            """INSERT INTO search_queue
               (id, document_id, query, node_id, generation, priority, status)
               VALUES (?, ?, ?, ?, ?, ?, 'pending')""",
            (entry_id, doc_id, query, node_id, generation, priority),
        )
    return entry_id


def dequeue_next(doc_id: str) -> dict | None:
    with get_db() as db:
        row = db.execute(
            """SELECT * FROM search_queue
               WHERE document_id=? AND status='pending'
               ORDER BY priority DESC, created_at ASC
               LIMIT 1""",
            (doc_id,),
        ).fetchone()
        if row:
            db.execute(
                "UPDATE search_queue SET status='running' WHERE id=?", (row["id"],)
            )
    return dict(row) if row else None


def mark_queue_done(entry_id: str) -> None:
    with get_db() as db:
        db.execute(
            "UPDATE search_queue SET status='done' WHERE id=?", (entry_id,)
        )


def pending_count(doc_id: str) -> int:
    with get_db() as db:
        row = db.execute(
            "SELECT COUNT(*) as n FROM search_queue WHERE document_id=? AND status='pending'",
            (doc_id,),
        ).fetchone()
    return row["n"] if row else 0


# ── ToC rendering ─────────────────────────────────────────────────────────────

def get_pending_sections(doc_id: str) -> list[dict]:
    """Return toc_nodes that are still pending or interrupted (active)."""
    with get_db() as db:
        rows = db.execute(
            """SELECT * FROM toc_nodes
               WHERE document_id=? AND status IN ('pending', 'active')
               ORDER BY order_index""",
            (doc_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def build_collapsed_toc(doc_id: str, current_node_id: str | None = None) -> str:
    nodes = get_toc(doc_id)
    STATUS_ICON = {
        "complete": "✓",
        "active": "→",
        "pending": "○",
        "needs_revision": "!",
    }
    lines = []
    for node in nodes:
        icon = STATUS_ICON.get(node["status"], "○")
        indent = "  " * node["depth"]
        is_current = node["id"] == current_node_id
        words = f"({node['word_count']}w)" if node["word_count"] > 0 else ""
        marker = "  ← WRITING NOW" if is_current else ""

        if is_current:
            lines.append(f"{indent}{icon} {node['title']} {words}{marker}".rstrip())
        elif node["summary"] and node["status"] == "complete":
            short = (node["summary"][:80] + "…") if len(node["summary"]) > 80 else node["summary"]
            lines.append(f"{indent}{icon} {node['title']} {words} — {short}".rstrip())
        else:
            lines.append(f"{indent}{icon} {node['title']} {words}".rstrip())

    return "\n".join(lines)
