"""
graph_savoria.py
Membangun graph LangGraph untuk Savoria Multi-Agent System:

    User Input -> Supervisor (klasifikasi divisi) -> Agent Divisi -> Output

Supervisor menggunakan LLM (Ollama) untuk mengklasifikasikan pertanyaan
ke salah satu dari: inventory, order, hr, finance, atau general.

Cara jalankan (contoh interaktif):
    python graph_savoria.py
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

import agents

LLM_MODEL_NAME = "llama3.2"
_llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0)

VALID_DIVISIONS = ["inventory", "order", "hr", "finance", "general"]


# ------------------------------------------------------------------
# STATE
# ------------------------------------------------------------------
class SavoriaState(TypedDict):
    question: str
    branch_id: Optional[str]
    divisi: Optional[str]
    result: Optional[dict]


# ------------------------------------------------------------------
# NODE: SUPERVISOR (klasifikasi divisi)
# ------------------------------------------------------------------
def supervisor_node(state: SavoriaState) -> SavoriaState:
    question = state["question"]

    classification_prompt = f"""Kamu adalah Supervisor Agent di sistem Savoria.
Klasifikasikan pertanyaan berikut ke SATU kategori berikut saja (jawab satu kata):
- inventory  (soal stok bahan baku, restock, bahan habis)
- order      (soal menu terlaris, pesanan, kanal online/offline, jam sibuk)
- hr         (soal jadwal shift, karyawan, staf)
- finance    (soal omzet, keuangan, selisih rekonsiliasi)
- general    (kalau tidak cocok dengan kategori manapun)

Pertanyaan: "{question}"

Jawab HANYA dengan satu kata kategori di atas, tanpa penjelasan tambahan."""

    response = _llm.invoke(classification_prompt)
    divisi = response.content.strip().lower()

    # fallback kalau LLM menjawab di luar kategori yang valid
    if divisi not in VALID_DIVISIONS:
        divisi = "general"

    print(f"[Supervisor] Pertanyaan diklasifikasikan sebagai: '{divisi}'")
    state["divisi"] = divisi
    return state


# ------------------------------------------------------------------
# NODE: AGENT DIVISI
# ------------------------------------------------------------------
def agent_node(state: SavoriaState) -> SavoriaState:
    divisi = state["divisi"]
    question = state["question"]
    branch_id = state.get("branch_id")

    if divisi == "general":
        state["result"] = {
            "agent": "General",
            "answer": (
                "Maaf, pertanyaan ini tidak termasuk ke divisi Inventory, Order, "
                "HR, atau Finance. Coba ajukan pertanyaan yang lebih spesifik "
                "terkait salah satu divisi tersebut."
            ),
            "sop_sources": [],
            "data_used": "-",
        }
        return state

    agent_fn = agents.AGENT_MAP[divisi]
    state["result"] = agent_fn(question, branch_id=branch_id)
    return state


# ------------------------------------------------------------------
# BUILD GRAPH
# ------------------------------------------------------------------
def build_graph():
    graph = StateGraph(SavoriaState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("agent", agent_node)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "agent")
    graph.add_edge("agent", END)

    return graph.compile()


# ------------------------------------------------------------------
# MODE INTERAKTIF UNTUK TESTING
# ------------------------------------------------------------------
def main():
    app = build_graph()
    print("=== SAVORIA MULTI-AGENT SYSTEM ===")
    print("Ketik pertanyaan (atau 'exit' untuk keluar)\n")

    while True:
        question = input("Pertanyaan Manajer: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        state = {"question": question, "branch_id": None, "divisi": None, "result": None}
        final_state = app.invoke(state)
        result = final_state["result"]

        print(f"\n[Agent {result['agent']}] menjawab:")
        print(result["answer"])
        if result["sop_sources"]:
            print("\nSumber SOP yang dipakai:")
            for src in result["sop_sources"]:
                print(f"  - {src}")
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
