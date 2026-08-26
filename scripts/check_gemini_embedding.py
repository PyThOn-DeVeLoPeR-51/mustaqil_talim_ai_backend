from __future__ import annotations

from app.rag.embeddings import GeminiEmbeddingProvider


def main() -> None:
    provider = GeminiEmbeddingProvider()
    documents = provider.embed_documents(
        [
            "Muhandislik grafikasi bo‘yicha o‘quv materiali.",
            "Mustaqil ta’lim topshirig‘i uchun qisqa sinov matni.",
        ]
    )
    query = provider.embed_query("Mustaqil ta’lim nima?")

    print(f"Provider: {provider.provider_name}")
    print(f"Model: {provider.model_name}")
    print(f"Document vectors: {len(documents)}")
    print(f"Dimensions: {len(documents[0]) if documents else 0}")
    print(f"Query dimensions: {len(query)}")
    print("Gemini embedding: OK")


if __name__ == "__main__":
    main()
