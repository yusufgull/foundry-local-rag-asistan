"""
search.py
---------
Vektor benzerligi (cosine similarity) hesaplama ve en alakali
chunk'lari bulma fonksiyonlari.

Resmi Microsoft "Tutorial: Build a RAG application" ornegindeki
mantigin aynisidir; biz sadece SQLite'tan okuyacak sekilde uyarladik.
"""

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Iki vektor arasindaki kosinus benzerligini hesaplar (0-1 arasi, 1'e yakin = cok benzer)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def find_relevant(query_embedding: list[float], chunks: list[dict], top_k: int = 3):
    """
    chunks: db.get_all_chunks() ile alinan liste
    Donen deger: en alakali top_k chunk (skoruyla birlikte), buyukten kucuge siralanmis
    """
    scored = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]
