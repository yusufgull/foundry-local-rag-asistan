import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "knowledge_base.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def clear_db():
    conn = get_connection()
    conn.execute("DELETE FROM chunks")
    conn.commit()
    conn.close()


def insert_chunk(source: str, content: str, embedding: list[float]):
    conn = get_connection()
    conn.execute(
        "INSERT INTO chunks (source, content, embedding) VALUES (?, ?, ?)",
        (source, content, json.dumps(embedding)),
    )
    conn.commit()
    conn.close()


def get_all_chunks():
    conn = get_connection()
    rows = conn.execute("SELECT source, content, embedding FROM chunks").fetchall()
    conn.close()
    result = []
    for row in rows:
        result.append(
            {
                "source": row["source"],
                "content": row["content"],
                "embedding": json.loads(row["embedding"]),
            }
        )
    return result


def count_chunks() -> int:
    conn = get_connection()
    n = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
    conn.close()
    return n
