from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

import db

DOCS_DIR = Path(__file__).parent / "documents"
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"


def chunk_text(text: str, min_chunk_chars: int = 40) -> list[str]:
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    buffer = ""
    for para in raw_paragraphs:
        buffer = (buffer + "\n" + para).strip() if buffer else para
        if len(buffer) >= min_chunk_chars:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        if chunks:
            chunks[-1] = chunks[-1] + "\n" + buffer
        else:
            chunks.append(buffer)
    return chunks


def load_documents() -> list[tuple[str, str]]:
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"'{DOCS_DIR}' klasoru bulunamadi.")

    files = sorted(DOCS_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"'{DOCS_DIR}' icinde .txt dosyasi bulunamadi. "
            "Once documents/ klasorune en az bir .txt dosyasi ekleyin."
        )

    docs = []
    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        docs.append((file_path.name, content))
    return docs


def main():
    print("=== INGEST: Dokumanlar okunuyor ve embedding'ler uretiliyor ===\n")

    documents = load_documents()
    print(f"{len(documents)} dosya bulundu: {[name for name, _ in documents]}")

    all_chunks = []
    for name, content in documents:
        pieces = chunk_text(content)
        print(f"  - {name}: {len(pieces)} parcaya bolundu")
        for piece in pieces:
            all_chunks.append((name, piece))

    print(f"\nToplam {len(all_chunks)} chunk olusturuldu. Embedding uretiliyor...\n")

    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    embedding_model.download(
        lambda p: print(f"\rEmbedding modeli indiriliyor: {p:.1f}%", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    texts = [c for _, c in all_chunks]
    response = embedding_client.generate_embeddings(texts)
    embeddings = [item.embedding for item in response.data]

    db.init_db()
    db.clear_db()
    for (source, content), embedding in zip(all_chunks, embeddings):
        db.insert_chunk(source, content, embedding)

    embedding_model.unload()

    print(f"\nBitti! {db.count_chunks()} chunk 'knowledge_base.db' icine kaydedildi.")
    print("Simdi 'python main.py' calistirarak soru sorabilirsiniz.")


if __name__ == "__main__":
    main()
