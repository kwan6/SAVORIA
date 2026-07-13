"""
agents.py
Definisi 4 Agent Divisi (Inventory, Order, HR, Finance) untuk Savoria.

Setiap agent:
1. Retrieve konteks SOP relevan dari ChromaDB (RAG)
2. Ambil data aktual dari CSV lewat data_tools.py
3. Gabungkan keduanya jadi prompt, lalu panggil LLM (Ollama - llama3.2)
   untuk menghasilkan jawaban akhir yang eksplainable (menyebutkan sumbernya)
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

import data_tools as dt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(_THIS_DIR, "..", "chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.2"

_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
_llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0.2)

# CATATAN (hasil evaluate.py): jarak retrieval Chroma TIDAK memisahkan
# dengan bersih antara SOP yang benar-benar relevan vs yang cuma mirip
# bahasanya (dites di 21 kasus, jarak tumpang tindih 0.62-0.91 baik untuk
# jawaban yang grounded maupun yang berhalusinasi). Karena itu threshold
# ini HANYA dipakai sebagai lapis pertahanan tambahan untuk kasus retrieval
# yang benar-benar buruk (jarak sangat tinggi), BUKAN solusi utama.
# Solusi utama ada di instruksi anti-fabrikasi pada _build_prompt().
RETRIEVAL_DISTANCE_THRESHOLD = 1.0


def _get_retriever(collection_name: str, k: int = 2):
    """Dipertahankan untuk kompatibilitas dengan kode lain (mis. test_retrieval.py)."""
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=_embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vectordb.as_retriever(search_kwargs={"k": k})


def _retrieve_with_scores(collection_name: str, question: str, k: int = 2):
    """Ambil dokumen SOP beserta skor jaraknya (lebih kecil = lebih mirip)."""
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=_embeddings,
        persist_directory=PERSIST_DIR,
    )
    scored = vectordb.similarity_search_with_score(question, k=k)
    docs = [doc for doc, _ in scored]
    distances = [float(score) for _, score in scored]
    return docs, distances


def _get_source_filenames(docs: list) -> list:
    """Ambil nama file SOP ASLI dari metadata dokumen (diisi otomatis oleh
    TextLoader di build_vectorstore.py sebagai metadata['source']).

    Dipakai supaya LLM tahu nama dokumen yang BENAR untuk dikutip, dan tidak
    perlu (atau tidak boleh) mengarang nama dokumen sendiri seperti
    "SOP Divisi X - Savoria Resto Group" yang terdengar resmi tapi fiktif."""
    names = []
    for d in docs:
        src = d.metadata.get("source") if hasattr(d, "metadata") else None
        if src:
            names.append(os.path.basename(src))
    # unik tapi tetap urut
    seen = set()
    unique_names = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique_names.append(n)
    return unique_names


def _build_prompt(divisi: str, sop_context: str, data_context: str, question: str,
                   retrieval_uncertain: bool = False, source_filenames: list = None) -> str:
    warning_block = ""
    if retrieval_uncertain:
        warning_block = """
PERHATIAN KHUSUS: Sistem tidak yakin dokumen SOP di bawah ini benar-benar
relevan dengan pertanyaan. Jika isinya memang tidak menjawab pertanyaan,
katakan secara eksplisit bahwa SOP terkait topik ini belum tersedia,
JANGAN memaksakan jawaban dari pengetahuan umum di luar konteks."""

    source_filenames = source_filenames or []
    if source_filenames:
        source_names_str = ", ".join(f'"{n}"' for n in source_filenames)
        source_rule = f"""5. Nama SOP yang BOLEH kamu sebut HANYA ini: {source_names_str}.
   JANGAN PERNAH mengarang nama dokumen lain seperti "SOP Divisi {divisi} -
   Savoria Resto Group" atau judul resmi apa pun yang tidak ada di daftar ini,
   walaupun terdengar masuk akal. Kalau mau menyebut sumber, sebut PERSIS
   salah satu nama file di atas, atau cukup tulis "data aktual" untuk data
   dari sistem (bukan dari SOP)."""
    else:
        source_rule = """5. Tidak ada nama dokumen SOP yang bisa dikutip untuk pertanyaan ini.
   JANGAN mengarang nama dokumen atau judul SOP apa pun."""

    return f"""Kamu adalah Agent AI Divisi {divisi} di restoran Savoria.
Jawab pertanyaan manajer HANYA berdasarkan konteks SOP dan data aktual di bawah ini.

ATURAN KETAT (wajib dipatuhi):
1. Jangan mengarang informasi, prosedur, angka, atau langkah yang tidak ada di konteks.
2. JANGAN menyebut nomor pasal, nomor SOP, atau judul dokumen spesifik KECUALI
   nomor/judul tersebut memang tertulis persis di KONTEKS SOP di bawah. Kalau
   konteks tidak menyebutkan nomor pasal, jangan mengarang nomor pasal sendiri.
3. Jika KONTEKS SOP di bawah tidak membahas topik yang ditanyakan, katakan dengan
   jelas: "SOP terkait hal ini belum tersedia di knowledge base" - jangan
   menjawab seolah-olah kamu tahu prosedurnya dari sumber lain.
4. Sebutkan secara singkat sumber informasimu (SOP atau data aktual) di akhir
   jawaban, HANYA jika sumber itu benar-benar kamu pakai.
{source_rule}
{warning_block}
=== KONTEKS SOP (Divisi {divisi}) ===
{sop_context}

=== DATA AKTUAL ===
{data_context}

=== PERTANYAAN MANAJER ===
{question}

Jawaban (singkat, jelas, actionable):"""


def _run_agent(divisi: str, collection_name: str, data_context: str, question: str) -> dict:
    docs, distances = _retrieve_with_scores(collection_name, question)
    sop_context = "\n---\n".join(d.page_content for d in docs)
    source_filenames = _get_source_filenames(docs)
    best_distance = min(distances) if distances else None
    retrieval_uncertain = best_distance is None or best_distance > RETRIEVAL_DISTANCE_THRESHOLD

    prompt = _build_prompt(divisi, sop_context, data_context, question, retrieval_uncertain, source_filenames)
    response = _llm.invoke(prompt)

    return {
        "agent": divisi,
        "answer": response.content,
        "sop_sources": [d.page_content[:100] + "..." for d in docs],
        "source_filenames": source_filenames,
        "data_used": data_context,
        "retrieval_uncertain": retrieval_uncertain,
        "best_retrieval_distance": best_distance,
    }


# ------------------------------------------------------------------
# AGENT INVENTORY
# ------------------------------------------------------------------
def agent_inventory(question: str, branch_id: str = None) -> dict:
    data_context = dt.get_critical_stock(branch_id=branch_id)
    return _run_agent("Inventory", "collection_inventory", data_context, question)


# ------------------------------------------------------------------
# AGENT ORDER
# ------------------------------------------------------------------
def agent_order(question: str, branch_id: str = None) -> dict:
    top_menu = dt.get_top_menu(branch_id=branch_id)
    channel_dist = dt.get_channel_distribution(branch_id=branch_id)
    data_context = f"{top_menu}\n\n{channel_dist}"
    return _run_agent("Order", "collection_order", data_context, question)


# ------------------------------------------------------------------
# AGENT HR
# ------------------------------------------------------------------
def agent_hr(question: str, branch_id: str = None) -> dict:
    data_context = dt.get_shift_gaps(branch_id=branch_id)
    return _run_agent("HR", "collection_hr", data_context, question)


# ------------------------------------------------------------------
# AGENT FINANCE
# ------------------------------------------------------------------
def agent_finance(question: str, branch_id: str = None) -> dict:
    data_context = dt.get_finance_discrepancy(branch_id=branch_id)
    return _run_agent("Finance", "collection_finance", data_context, question)


AGENT_MAP = {
    "inventory": agent_inventory,
    "order": agent_order,
    "hr": agent_hr,
    "finance": agent_finance,
}