from foundry_local_sdk import Configuration, FoundryLocalManager

import db
import search

EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
CHAT_MODEL_ALIAS = "phi-3.5-mini"

TOP_K = 3

SYSTEM_PROMPT_TEMPLATE = (
    "Sen kisa ve net cevaplar veren bir Turkce asistansin. "
    "SADECE asagidaki BAGLAM icindeki bilgiyi kullan, baska hicbir sey ekleme. "
    "Cevabini 2-3 cumleyle sinirla, tekrar etme. "
    "Eger cevap baglamda yoksa, sadece 'Bu bilgi elimdeki dokumanlarda yok.' yaz, baska bir sey ekleme.\n\n"
    "BAGLAM:\n{context}"
)


def build_context(results) -> tuple[str, list[str]]:
    lines = []
    sources = []
    for score, chunk in results:
        lines.append(f"- ({chunk['source']}) {chunk['content']}")
        if chunk["source"] not in sources:
            sources.append(chunk["source"])
    return "\n".join(lines), sources


def main():
    if db.count_chunks() == 0:
        print(
            "Veritabani bos! Once 'python ingest.py' calistirip "
            "dokumanlarinizi isleyin."
        )
        return

    print("=== Yerel RAG Soru-Cevap Asistani ===")
    print(f"Veritabaninda {db.count_chunks()} bilgi parcasi var.\n")

    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    print("Embedding modeli yukleniyor...")
    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    embedding_model.download(
        lambda p: print(f"\r  indiriliyor: {p:.1f}%", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    print("Sohbet (chat) modeli yukleniyor...")
    chat_model = manager.catalog.get_model(CHAT_MODEL_ALIAS)
    chat_model.download(
        lambda p: print(f"\r  indiriliyor: {p:.1f}%", end="", flush=True)
    )
    print()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    all_chunks = db.get_all_chunks()

    print("\nModeller hazir! Sorularinizi yazabilirsiniz.")
    print('Cikmak icin "quit" yazin.\n')

    while True:
        query = input("Soru: ").strip()
        if not query or query.lower() == "quit":
            break

        query_response = embedding_client.generate_embedding(query)
        query_embedding = query_response.data[0].embedding

        results = search.find_relevant(query_embedding, all_chunks, top_k=TOP_K)
        context, sources = build_context(results)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_TEMPLATE.format(context=context),
            },
            {"role": "user", "content": query},
        ]

        print("Cevap: ", end="", flush=True)
        for chunk in chat_client.complete_streaming_chat(messages):
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
        print(f"\n(Kaynaklar: {', '.join(sources)})\n")

    embedding_model.unload()
    chat_model.unload()
    print("Gorusmek uzere!")


if __name__ == "__main__":
    main()
