"""
app.py
Dashboard Streamlit untuk Savoria Multi-Agent System.

Terdiri dari 3 tab:
1. Chat        - tanya jawab langsung ke sistem multi-agent
2. Monitoring  - visualisasi kondisi tiap cabang (stok, omzet, shift)
3. Evaluasi    - placeholder untuk skor evaluasi model (diisi di tahap berikutnya)

Cara jalankan (dari folder savoria_project):
    streamlit run app.py
"""

import sys
import os
import time
import json
import pandas as pd
import streamlit as st

# Supaya bisa import modul dari folder scripts/
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

import graph_savoria  # noqa: E402
import data_tools as dt  # noqa: E402

st.set_page_config(page_title="Savoria Command Center", page_icon="🍽️", layout="wide")

# ------------------------------------------------------------------
# CACHE: build graph & load data sekali saja
# ------------------------------------------------------------------
@st.cache_resource
def load_graph():
    return graph_savoria.build_graph()


@st.cache_data
def load_branches():
    return dt._load("branches.csv")


@st.cache_data
def load_all_data():
    return {
        "transactions": dt._load("transactions.csv"),
        "inventory": dt._load("inventory_daily.csv"),
        "finance": dt._load("finance_daily.csv"),
        "shifts": dt._load("shifts.csv"),
        "menu": dt._load("menu.csv"),
        "ingredients": dt._load("ingredients.csv"),
    }


app_graph = load_graph()
branches_df = load_branches()
data = load_all_data()

st.title("Savoria Command Center")
st.caption("Dashboard Multi-Agent AI untuk Manajemen Savoria Resto Group")

tab_chat, tab_monitor, tab_eval = st.tabs(["💬 Chat", "📊 Monitoring Cabang", "✅ Evaluasi Model"])

# ==================================================================
# TAB 1: CHAT
# ==================================================================
with tab_chat:
    st.subheader("Tanya ke Savoria AI Assistant")
    st.caption("Sistem akan otomatis merutekan pertanyaanmu ke Agent Divisi yang tepat "
               "(Inventory / Order / HR / Finance).")

    branch_filter = st.selectbox(
        "Filter cabang (opsional)",
        options=["Semua Cabang"] + branches_df["branch_name"].tolist(),
        key="chat_branch_filter",
    )
    branch_id_selected = None
    if branch_filter != "Semua Cabang":
        branch_id_selected = branches_df[branches_df["branch_name"] == branch_filter]["branch_id"].values[0]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Lihat sumber SOP yang dipakai"):
                    for src in msg["sources"]:
                        st.markdown(f"- {src}")

    question = st.chat_input("Tulis pertanyaan, misal: 'Bahan apa yang paling kritis stoknya?'")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Menganalisis dan merutekan ke agent yang tepat..."):
                start = time.time()
                state = {
                    "question": question,
                    "branch_id": branch_id_selected,
                    "divisi": None,
                    "result": None,
                }
                final_state = app_graph.invoke(state)
                elapsed = time.time() - start
                result = final_state["result"]

            st.markdown(f"**[Agent {result['agent']}]**")
            if result.get("retrieval_uncertain"):
                st.caption("⚠️ SOP yang ditemukan mungkin kurang relevan dengan pertanyaan ini — cek ulang jawaban di bawah.")
            st.markdown(result["answer"])
            st.caption(f"⏱️ Waktu respons: {elapsed:.1f} detik")

            if result["sop_sources"]:
                src_files = result.get("source_filenames") or []
                expander_title = (
                    f"Lihat sumber SOP yang dipakai ({', '.join(src_files)})"
                    if src_files else "Lihat sumber SOP yang dipakai"
                )
                with st.expander(expander_title):
                    for src in result["sop_sources"]:
                        st.markdown(f"- {src}")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"**[Agent {result['agent']}]**\n\n{result['answer']}",
            "sources": result["sop_sources"],
        })

    if st.session_state.chat_history:
        if st.button("🗑️ Bersihkan Riwayat Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ==================================================================
# TAB 2: MONITORING
# ==================================================================
with tab_monitor:
    st.subheader("Monitoring Kondisi Cabang")

    selected_branch_name = st.selectbox(
        "Pilih cabang",
        options=branches_df["branch_name"].tolist(),
        key="monitor_branch_filter",
    )
    selected_branch_id = branches_df[branches_df["branch_name"] == selected_branch_name]["branch_id"].values[0]

    col1, col2, col3 = st.columns(3)

    # --- Metrik ringkas ---
    trx = data["transactions"]
    trx_branch = trx[trx["branch_id"] == selected_branch_id]
    total_omzet = trx_branch["total_price"].sum()

    inv = data["inventory"]
    inv_branch = inv[inv["branch_id"] == selected_branch_id]
    latest_date = inv_branch["date"].max()
    inv_latest = inv_branch[inv_branch["date"] == latest_date].copy()
    inv_latest["pct"] = inv_latest["stock_remaining"] / inv_latest["stock_start"] * 100
    n_kritis = (inv_latest["pct"] < 10).sum()

    fin = data["finance"]
    fin_branch = fin[fin["branch_id"] == selected_branch_id]
    total_discrepancy = fin_branch["discrepancy"].abs().sum()

    col1.metric("Total Omzet (90 hari)", f"Rp {total_omzet:,.0f}")
    col2.metric("Bahan Baku Status Kritis (hari ini)", f"{n_kritis} item")
    col3.metric("Total Selisih Keuangan (90 hari)", f"Rp {total_discrepancy:,.0f}")

    st.divider()

    # --- Grafik omzet harian ---
    st.markdown("**Tren Omzet Harian**")
    omzet_harian = trx_branch.groupby("date")["total_price"].sum()
    st.line_chart(omzet_harian)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Menu Terlaris**")
        menu_df = data["menu"]
        top_menu = (
            trx_branch.groupby("menu_id")["qty"].sum()
            .reset_index().sort_values("qty", ascending=False).head(5)
            .merge(menu_df, on="menu_id")
        )
        st.bar_chart(top_menu.set_index("menu_name")["qty"])

    with col_b:
        st.markdown("**Status Stok Bahan Baku (Hari Ini)**")
        ing_df = data["ingredients"]
        inv_display = inv_latest.merge(ing_df, on="ingredient_id")[["ingredient_name", "pct"]]
        inv_display = inv_display.sort_values("pct")
        st.bar_chart(inv_display.set_index("ingredient_name")["pct"])

    st.markdown("**Distribusi Kanal Pemesanan**")
    channel_dist = trx_branch["order_channel"].value_counts()
    st.bar_chart(channel_dist)

# ==================================================================
# TAB 3: EVALUASI MODEL
# ==================================================================
with tab_eval:
    st.subheader("Evaluasi Model Multi-Agent Savoria")

    EVAL_DIR = os.path.join(os.path.dirname(__file__), "evaluation_results")
    summary_path = os.path.join(EVAL_DIR, "summary.json")
    detail_path = os.path.join(EVAL_DIR, "detail.csv")
    cm_path = os.path.join(EVAL_DIR, "confusion_matrix.png")
    metrics_png_path = os.path.join(EVAL_DIR, "metrics_summary.png")

    if not os.path.exists(summary_path):
        st.warning(
            "Belum ada hasil evaluasi. Jalankan dulu script evaluasi dari terminal "
            "(pastikan Ollama sedang berjalan):"
        )
        st.code("python scripts\\evaluate.py", language="bash")
        st.caption(
            "Script ini akan menguji sistem dengan sejumlah pertanyaan berlabel "
            "(ground-truth), lalu menghitung skor Accuracy, Effectiveness, "
            "Efficiency, Explainability, dan Hallucination Rate."
        )
    else:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        rtr = summary.get("run_to_run_variability")
        n_runs = rtr["n_runs"] if rtr else 1
        n_pooled = summary.get('n_test_cases', 0)
        n_per_run = n_pooled // n_runs if n_runs else n_pooled

        st.caption(f"Terakhir dievaluasi: {summary.get('generated_at', '-')} "
                   f"| {n_runs} run x {n_per_run} test case (pooled: {n_pooled}) "
                   f"({summary.get('n_errors', 0)} error)")

        # ---------------- RINGKASAN 5 METRIK UTAMA ----------------
        col1, col2, col3, col4, col5 = st.columns(5)

        def _std_delta(path_keys):
            if not rtr:
                return None
            node = rtr
            for k in path_keys:
                node = node.get(k) if isinstance(node, dict) else None
                if node is None:
                    return None
            # node di sini SUDAH berupa nilai std (float), karena path_keys
            # sudah diakhiri dengan "std" -> tidak perlu .get() lagi.
            std = node
            return f"± {std:.1%} antar-run" if isinstance(std, (int, float)) else None

        acc = summary["accuracy"]["overall_accuracy"]
        col1.metric("1️⃣ Accuracy (Routing)", f"{acc:.1%}", _std_delta(["accuracy", "std"]), delta_color="off")

        eff = summary["effectiveness"]["avg_context_coverage_recall"]
        col2.metric("2️⃣ Effectiveness", f"{eff:.1%}" if eff is not None else "N/A",
                     _std_delta(["effectiveness_coverage", "std"]), delta_color="off")

        avg_latency = summary["efficiency"]["avg_latency_total_sec"]
        col3.metric("3️⃣ Efficiency", f"{avg_latency:.1f}s/query" if avg_latency is not None else "N/A")

        citation_rate = summary["explainability"]["citation_rate"]
        col4.metric("4️⃣ Explainability", f"{citation_rate:.1%}" if citation_rate is not None else "N/A",
                     _std_delta(["explainability_citation_rate", "std"]), delta_color="off")

        halluc_rate = summary["hallucination"]["hallucination_rate"]
        col5.metric("5️⃣ Hallucination Rate", f"{halluc_rate:.1%}" if halluc_rate is not None else "N/A",
                     _std_delta(["hallucination_rate", "std"]), delta_color="inverse")

        if n_runs > 1:
            st.caption(
                f"Skor di atas dihitung dari data GABUNGAN (pooled) {n_runs} run terpisah "
                f"untuk stabilitas statistik. Angka '±' menunjukkan seberapa besar variasi "
                f"antar-run individual (murni akibat randomness LLM, temperature=0.2)."
            )

        st.divider()

        if os.path.exists(metrics_png_path):
            st.image(metrics_png_path, caption="Ringkasan skor 5 dimensi evaluasi (data pooled)")

        if rtr and rtr.get("n_runs", 1) > 1:
            with st.expander("📊 Variansi Antar-Run (stabilitas statistik)", expanded=False):
                st.caption(rtr.get("note", ""))
                var_rows = []
                labels_map = {
                    "accuracy": "Accuracy",
                    "effectiveness_coverage": "Effectiveness (coverage)",
                    "efficiency_avg_latency_sec": "Efficiency (avg latency, detik)",
                    "explainability_citation_rate": "Explainability (citation rate)",
                    "hallucination_rate": "Hallucination rate",
                }
                for key, label in labels_map.items():
                    node = rtr.get(key, {})
                    values = node.get("values_per_run", [])
                    std = node.get("std")
                    var_rows.append({
                        "Metrik": label,
                        **{f"Run {i+1}": v for i, v in enumerate(values)},
                        "Std Dev": std,
                    })
                st.dataframe(pd.DataFrame(var_rows))

        # ---------------- DETAIL PER DIMENSI ----------------
        with st.expander("📌 1. Accuracy — Detail Routing Supervisor", expanded=False):
            st.markdown(
                "Mengukur seberapa tepat Supervisor (fine-tuned TF-IDF+LogReg dengan "
                "fallback LLM) merutekan pertanyaan ke divisi yang benar."
            )
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix - Supervisor Routing")

            per_class_df = pd.DataFrame(summary["accuracy"]["per_class"]).T
            per_class_df.index.name = "divisi"
            st.dataframe(per_class_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1": "{:.2f}"}))

        with st.expander("📌 2. Effectiveness — Coverage Fakta Data Aktual", expanded=False):
            st.markdown(
                "Mengukur seberapa banyak fakta penting (angka, nama bahan/menu/cabang) "
                "dari data aktual yang benar-benar disebutkan di jawaban akhir agent."
            )
            per_div = summary["effectiveness"].get("per_division", {})
            if per_div:
                st.bar_chart(pd.Series(per_div, name="coverage_recall"))
            st.caption(f"Dihitung dari {summary['effectiveness'].get('n_cases_evaluated', 0)} test case "
                       f"(kategori 'general' tidak dinilai karena tidak ada data/SOP acuan).")

        with st.expander("📌 3. Efficiency — Latency Sistem", expanded=False):
            e = summary["efficiency"]
            ecol1, ecol2, ecol3 = st.columns(3)
            ecol1.metric("Avg latency total", f"{e['avg_latency_total_sec']:.2f}s" if e['avg_latency_total_sec'] else "N/A")
            ecol2.metric("Median latency total", f"{e['median_latency_total_sec']:.2f}s" if e['median_latency_total_sec'] else "N/A")
            ecol3.metric("P95 latency total", f"{e['p95_latency_total_sec']:.2f}s" if e['p95_latency_total_sec'] else "N/A")
            st.metric("Avg latency Supervisor (fine-tuned) saja", f"{e['avg_latency_supervisor_sec']*1000:.0f} ms" if e['avg_latency_supervisor_sec'] else "N/A")
            st.metric("Fallback ke LLM rate", f"{e['fallback_rate']:.1%}" if e['fallback_rate'] is not None else "N/A")
            st.caption("Supervisor fine-tuned jauh lebih cepat dari LLM; semakin rendah fallback rate, "
                       "semakin efisien pipeline routing secara keseluruhan.")

        with st.expander("📌 4. Explainability — Transparansi Sumber Jawaban", expanded=False):
            ex = summary["explainability"]
            xcol1, xcol2, xcol3 = st.columns(3)
            xcol1.metric("Citation rate", f"{ex['citation_rate']:.1%}" if ex['citation_rate'] is not None else "N/A")
            xcol2.metric("Avg dokumen SOP diretrieve", f"{ex['avg_sop_docs_retrieved']:.1f}" if ex['avg_sop_docs_retrieved'] is not None else "N/A")
            xcol3.metric("Avg jarak retrieval", f"{ex['avg_retrieval_distance']:.3f}" if ex['avg_retrieval_distance'] is not None else "N/A")
            st.caption(ex.get("note", ""))

        with st.expander("📌 5. Hallucination — Fakta yang Tidak Terbukti di Konteks", expanded=False):
            h = summary["hallucination"]
            hcol1, hcol2 = st.columns(2)
            hcol1.metric("Avg grounding precision", f"{h['avg_grounding_precision']:.1%}" if h['avg_grounding_precision'] is not None else "N/A")
            hcol2.metric("Hallucination rate", f"{h['hallucination_rate']:.1%}" if h['hallucination_rate'] is not None else "N/A")
            st.caption(
                f"Threshold: jawaban dianggap 'berhalusinasi' jika grounding precision < {h['threshold_used']:.0%}. "
                f"{h.get('note', '')}"
            )
            breakdown = h.get("detection_breakdown", {})
            if breakdown:
                bcol1, bcol2 = st.columns(2)
                bcol1.metric("Dinilai via fact-grounding", breakdown.get("n_cases_fact_based", 0))
                bcol2.metric("Dinilai via LLM-as-judge", breakdown.get("n_cases_llm_judge_based", 0))
                verdicts = breakdown.get("llm_judge_verdicts", {})
                if verdicts:
                    st.caption(f"Rincian verdict LLM-as-judge: {verdicts}")

        st.divider()

        # ---------------- DETAIL PER TEST CASE ----------------
        if os.path.exists(detail_path):
            with st.expander("📋 Lihat detail hasil per pertanyaan uji (semua run)"):
                detail_df = pd.read_csv(detail_path)
                show_cols = [
                    "run", "question", "expected_divisi", "predicted_divisi", "correct",
                    "latency_total_sec", "citation_present", "coverage_recall",
                    "grounding_precision", "hallucinated", "hallucination_method",
                    "llm_judge_verdict",
                ]
                show_cols = [c for c in show_cols if c in detail_df.columns]
                st.dataframe(detail_df[show_cols])

        st.button("🔄 Muat Ulang Hasil Evaluasi", on_click=lambda: st.rerun())