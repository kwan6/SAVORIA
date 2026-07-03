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


def _get_retriever(collection_name: str, k: int = 2):
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=_embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vectordb.as_retriever(search_kwargs={"k": k})


def _build_prompt(divisi: str, sop_context: str, data_context: str, question: str) -> str:
    return f"""Kamu adalah Agent AI Divisi {divisi} di restoran Savoria.
Jawab pertanyaan manajer HANYA berdasarkan konteks SOP dan data aktual di bawah ini.
Jangan mengarang informasi yang tidak ada di konteks. Sebutkan secara singkat
sumber informasimu (SOP atau data aktual) di akhir jawaban.

=== KONTEKS SOP (Divisi {divisi}) ===
{sop_context}

=== DATA AKTUAL ===
{data_context}

=== PERTANYAAN MANAJER ===
{question}

Jawaban (singkat, jelas, actionable):"""


def _run_agent(divisi: str, collection_name: str, data_context: str, question: str) -> dict:
    retriever = _get_retriever(collection_name)
    docs = retriever.invoke(question)
    sop_context = "\n---\n".join(d.page_content for d in docs)

    prompt = _build_prompt(divisi, sop_context, data_context, question)
    response = _llm.invoke(prompt)

    return {
        "agent": divisi,
        "answer": response.content,
        "sop_sources": [d.page_content[:100] + "..." for d in docs],
        "data_used": data_context,
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
