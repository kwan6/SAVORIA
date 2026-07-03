"""
build_vectorstore.py
Script untuk membaca dokumen SOP, memecahnya jadi chunk, embed, dan
menyimpannya ke ChromaDB sebagai knowledge base untuk RAG.

Setiap divisi (inventory, order, hr, finance) punya collection Chroma sendiri,
supaya agent masing-masing divisi hanya retrieve dari knowledge base-nya sendiri.

Cara jalankan:
    python build_vectorstore.py
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ------------------------------------------------------------------
# KONFIGURASI
# ------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SOP_DIR = os.path.join(_THIS_DIR, "..", "sop_docs")          # folder berisi file .txt SOP
PERSIST_DIR = os.path.join(_THIS_DIR, "..", "chroma_db")     # tempat ChromaDB menyimpan data secara lokal

# Mapping nama file SOP -> nama collection per divisi
DIVISI_MAP = {
    "sop_inventory.txt": "collection_inventory",
    "sop_order.txt": "collection_order",
    "sop_hr.txt": "collection_hr",
    "sop_finance.txt": "collection_finance",
}

# Model embedding open-source, ringan, jalan di CPU
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_vectorstore_for_division(filename: str, collection_name: str, embeddings):
    """Load 1 file SOP, split jadi chunk, dan simpan ke collection Chroma tersendiri."""
    filepath = os.path.join(SOP_DIR, filename)

    if not os.path.exists(filepath):
        print(f"[SKIP] File tidak ditemukan: {filepath}")
        return

    print(f"\n[PROSES] {filename} -> collection '{collection_name}'")

    # 1. Load dokumen
    loader = TextLoader(filepath, encoding="utf-8")
    documents = loader.load()

    # 2. Split jadi chunk kecil (supaya retrieval lebih presisi)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"  -> Dipecah menjadi {len(chunks)} chunk")

    # 3. Embed & simpan ke ChromaDB (collection terpisah per divisi)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=PERSIST_DIR,
    )
    # Catatan: di package langchain_chroma versi baru, data otomatis
    # tersimpan ke persist_directory saat from_documents() dipanggil,
    # jadi tidak perlu lagi memanggil vectordb.persist() secara manual.
    print(f"  -> Tersimpan ke ChromaDB (persist_directory='{PERSIST_DIR}')")


def main():
    print("=== BUILD VECTOR STORE SAVORIA ===")
    print(f"Menggunakan model embedding: {EMBEDDING_MODEL_NAME}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    for filename, collection_name in DIVISI_MAP.items():
        build_vectorstore_for_division(filename, collection_name, embeddings)

    print("\n=== SELESAI ===")
    print("Semua SOP sudah di-embed dan disimpan ke ChromaDB.")


if __name__ == "__main__":
    main()
