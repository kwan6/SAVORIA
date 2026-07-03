"""
test_retrieval.py
Script sederhana untuk MENGUJI apakah retrieval dari ChromaDB sudah benar,
sebelum dipakai di dalam agent LangGraph.

Cara jalankan:
    python test_retrieval.py
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(_THIS_DIR, "..", "chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Beberapa pertanyaan uji coba per divisi
TEST_QUERIES = {
    "collection_inventory": [
        "Kapan bahan baku harus direstock?",
        "Bagaimana cara mencegah bahan baku terbuang?",
    ],
    "collection_order": [
        "Apa yang harus dilakukan saat jam sibuk?",
        "Bagaimana urutan prioritas pesanan online vs offline?",
    ],
    "collection_hr": [
        "Berapa minimal staf shift malam saat weekend?",
        "Apa aturan kalau jadwal shift bentrok?",
    ],
    "collection_finance": [
        "Berapa batas toleransi selisih rekonsiliasi?",
        "Apa penyebab umum selisih omzet?",
    ],
}


def main():
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    for collection_name, queries in TEST_QUERIES.items():
        print(f"\n===== Collection: {collection_name} =====")
        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": 2})

        for q in queries:
            print(f"\nQ: {q}")
            results = retriever.invoke(q)
            for i, doc in enumerate(results, 1):
                snippet = doc.page_content.replace("\n", " ")[:200]
                print(f"  [{i}] {snippet}...")


if __name__ == "__main__":
    main()
