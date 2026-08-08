"""
db.py
-----
SQLite ile calisan basit bir "vektor depolama" katmani.
Her dokuman parcasini (chunk) ve onun embedding vektorunu saklariz.
Embedding vektorlerini JSON string olarak TEXT sutununda tutuyoruz
(kucuk olcekli projeler icin bu yeterlidir, ayri bir vektor
veritabanina gerek yoktur).
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "knowledge_base.db"


def get_connection():
    """Veritabani baglantisi acar."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Tablo yoksa olusturur. Program her calistiginda cagirmak guvenlidir."""
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
    """Tum chunk'lari siler (yeniden ingest etmeden once kullanilir)."""
    conn = get_connection()
    conn.execute("DELETE FROM chunks")
    conn.commit()
    conn.close()


def insert_chunk(source: str, content: str, embedding: list[float]):
    """Bir dokuman parcasini ve embedding'ini veritabanina yazar."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO chunks (source, content, embedding) VALUES (?, ?, ?)",
        (source, content, json.dumps(embedding)),
    )
    conn.commit()
    conn.close()


def get_all_chunks():
    """
    Tum chunk'lari (source, content, embedding) olarak dondurur.
    embedding alani tekrar Python listesine cevrilir.
    """
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
