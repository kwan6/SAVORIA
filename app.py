"""
app.py
Dashboard Streamlit untuk Savoria Multi-Agent System.
"""

import sys
import os
import time
import datetime
import pandas as pd
import streamlit as st

# Supaya bisa import modul dari folder scripts/
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

import graph_savoria  # noqa: E402
import data_tools as dt  # noqa: E402

st.set_page_config(
    page_title="SAVORIA",
    layout="wide"
)

def load_css():
    with open("assets/styles.css", encoding="utf-8") as f:
        st.markdown(
            f"""
            <style>
            {f.read()}
            </style>
            """,
            unsafe_allow_html=True,
        )
load_css()

def get_greeting() -> str:
    """Sapaan dinamis sesuai jam, gaya seperti 'Good afternoon'."""
    hour = datetime.datetime.now().hour
    if 4 <= hour < 11:
        return "Selamat pagi"
    elif 11 <= hour < 15:
        return "Selamat siang"
    elif 15 <= hour < 19:
        return "Selamat sore"
    else:
        return "Selamat malam"

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

# ==================================================================
# HEADER: PENJELASAN VERTIKAL MAJU KE BAWAH & SEJAJAR TENGAH PERFECT
# ==================================================================
# Menambah space kosong yang cukup dari batas atas browser agar layout bernafas
st.markdown("<div style='margin-top: 55px;'></div>", unsafe_allow_html=True)

# Membuat kolom pembagi judul dan dropdown
top_col1, top_col2 = st.columns([4, 1.2])

with top_col1:
    st.markdown(
        """
        <h1 style="font-size: 2.2rem; font-weight: 700; color: #F2E9DD; margin: 0; padding: 0; line-height: 1.2;">
            Savoria Command Center
        </h1>
        """,
        unsafe_allow_html=True,
    )

with top_col2:
    # Memaksa dropdown turun lebih banyak ke bawah dengan margin-top 14px agar selaras dengan baseline teks judul
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    branch_filter = st.selectbox(
        "Cabang",
        options=["Semua Cabang"] + branches_df["branch_name"].tolist(),
        key="chat_branch_filter",
        label_visibility="collapsed",
    )

branch_id_selected = None
if branch_filter != "Semua Cabang":
    branch_id_selected = branches_df[branches_df["branch_name"] == branch_filter]["branch_id"].values[0]

# Memberikan sedikit ruang sebelum masuk ke komponen Tab navigasi
st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

tab_chat, tab_monitor, tab_eval = st.tabs([
    ":material/chat: Chat",
    ":material/monitoring: Monitoring",
    ":material/query_stats: Evaluasi"
])

# ==================================================================
# TAB 1: CHAT
# ==================================================================
SUGGESTIONS = [
    (
        ":material/inventory_2:",
        "Cek Stok Kritis",
        "Lihat bahan baku yang stoknya hampir habis.",
        "Bahan apa yang paling kritis stoknya hari ini?"
    ),
    (
        ":material/trending_up:",
        "Analisis Omzet",
        "Analisis performa penjualan cabang.",
        "Bagaimana tren omzet 7 hari terakhir?"
    ),
    (
        ":material/groups:",
        "Info Shift",
        "Informasi jadwal karyawan.",
        "Siapa saja yang shift hari ini?"
    ),
    (
        ":material/description:",
        "Cari SOP",
        "Temukan SOP yang relevan.",
        "Apa SOP untuk penanganan selisih kas?"
    )
]

def process_question(question: str, branch_id_selected):
    st.session_state.chat_history.append({"role": "user", "content": question})
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

    st.session_state.chat_history.append({
            "role":"assistant",
            "agent": result["agent"],
            "content": result["answer"],
            "sources": result["sop_sources"],
            "elapsed": elapsed,
        })

with tab_chat:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ---------------------------------------------------------
    # LAYAR AWAL: Greeting + Perbaikan Chip Icon Profesional
    # ---------------------------------------------------------
    if not st.session_state.chat_history:
        st.markdown(
            f"""
            <div class="sv-greeting-wrap">
                <div class="sv-greeting-title">{get_greeting()}, Manajer</div>
                <div class="sv-agent-list">
                    <span class="sv-chip"><i class="material-icons">inventory_2</i> Inventory</span>
                    <span class="sv-chip"><i class="material-icons">payments</i> Finance</span>
                    <span class="sv-chip"><i class="material-icons">badge</i> HR</span>
                    <span class="sv-chip"><i class="material-icons">description</i> SOP</span>
                    <span class="sv-chip"><i class="material-icons">local_shipping</i> Order</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cols = st.columns(len(SUGGESTIONS))
        clicked_question = None

        for col, (icon, label, desc, prompt_text) in zip(cols, SUGGESTIONS):
            with col:
                if st.button(
                    label,
                    icon=icon,
                    key=f"sugg_{label}",
                    use_container_width=True,
                    help=desc,
                ):
                    clicked_question = prompt_text

        question = st.chat_input("Tanya apa saja...") or clicked_question
        if question:
            process_question(question, branch_id_selected)
            st.rerun()

    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    if msg.get("elapsed") is not None:
                        st.caption(f":material/smart_toy: {msg.get('agent', 'AI Agent')} · {msg['elapsed']:.1f}s")
                    if msg.get("sources"):
                        with st.expander("Lihat sumber SOP yang dipakai"):
                            for src in msg["sources"]:
                                st.markdown(f"- {src}")

        question = st.chat_input("Tanya apa saja...")
        if question:
            process_question(question, branch_id_selected)
            st.rerun()

        if st.button("Mulai Percakapan Baru", icon=":material/refresh:"):
            st.session_state.chat_history = []
            st.rerun()

# ==================================================================
# TAB 2 & 3 TETAP SAMA (tidak disentuh logika bisnisnya)
# ==================================================================
with tab_monitor:
    st.subheader("Monitoring Kondisi Cabang")
    selected_branch_name = st.selectbox("Pilih cabang", options=branches_df["branch_name"].tolist(), key="monitor_branch_filter")
    selected_branch_id = branches_df[branches_df["branch_name"] == selected_branch_name]["branch_id"].values[0]

    col1, col2, col3 = st.columns(3)
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
    st.markdown("**Tren Omzet Harian**")
    omzet_harian = trx_branch.groupby("date")["total_price"].sum()
    st.line_chart(omzet_harian)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Menu Terlaris**")
        menu_df = data["menu"]
        top_menu = (trx_branch.groupby("menu_id")["qty"].sum().reset_index().sort_values("qty", ascending=False).head(5).merge(menu_df, on="menu_id"))
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

with tab_eval:
    st.subheader("Evaluasi Model")
    st.info("Tab ini akan menampilkan skor evaluasi sistem multi-agent...")