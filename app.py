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

st.title("🍽️ Savoria Command Center")
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
            st.markdown(result["answer"])
            st.caption(f"⏱️ Waktu respons: {elapsed:.1f} detik")

            if result["sop_sources"]:
                with st.expander("Lihat sumber SOP yang dipakai"):
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
# TAB 3: EVALUASI (placeholder, diisi di tahap berikutnya)
# ==================================================================
with tab_eval:
    st.subheader("Evaluasi Model")
    st.info(
        "Tab ini akan menampilkan skor evaluasi sistem multi-agent: "
        "Accuracy, Effectiveness, Efficiency, Explainability, dan Hallucination Rate.\n\n"
        "Bagian ini akan diisi setelah skrip evaluasi (`evaluate.py`) dijalankan "
        "dan hasilnya disimpan ke file CSV."
    )
