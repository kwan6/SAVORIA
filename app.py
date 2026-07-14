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
import datetime
import pandas as pd
import streamlit as st

# Supaya bisa import modul dari folder scripts/
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

import graph_savoria  # noqa: E402
import data_tools as dt  # noqa: E402

st.set_page_config(page_title="SAVORIA", page_icon="🍽️", layout="wide")

# ------------------------------------------------------------------
# STYLE: palet warna Savoria (tetap dipakai, tidak diganti)
#   Primary   : #8B5E3C (Coffee Brown)
#   Secondary : #DCC7AA (Cream)
#   Success   : #2E7D32
#   Warning   : #F57C00
#   Danger    : #C62828
#   Background: #F8F6F3
# ------------------------------------------------------------------
CUSTOM_CSS = """
<style>
:root{
    --sv-primary: #8B5E3C;
    --sv-primary-dark: #6E4A2E;
    --sv-secondary: #DCC7AA;
    --sv-bg: #F8F6F3;
    --sv-success: #2E7D32;
    --sv-warning: #F57C00;
    --sv-danger: #C62828;
}

.stApp{
    background: radial-gradient(circle at 20% 0%, #F3EAE0 0%, var(--sv-bg) 45%, #EFE6DA 100%);
}

/* Padding rapi di sekeliling konten */
.block-container{
    padding-top: 2.8rem;
    max-width: 1100px;
}

/* ---------------- Header judul dashboard ---------------- */
.sv-header{
    padding-bottom: 0.6rem;
    margin-bottom: 0.4rem;
    border-bottom: 1px solid rgba(139, 94, 60, 0.15);
}
.sv-header-title{
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--sv-primary-dark);
    line-height: 1.3;
}
.sv-header-caption{
    font-size: 0.88rem;
    color: #9c8c7c;
    margin-top: 0.1rem;
}

/* ---------------- Filter cabang: dropdown kecil di kanan ---------------- */
div[data-testid="stSelectbox"]{
    margin-top: 0.15rem;
}
div[data-testid="stSelectbox"] > div > div{
    border-radius: 8px;
    border: 1px solid var(--sv-secondary);
    background-color: #FFFFFF;
    font-size: 0.85rem;
    min-height: 2.1rem;
}

/* ---------------- Greeting screen (state kosong) ---------------- */
.sv-greeting-wrap{
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    text-align:center;
    padding: 2rem 1rem 1.2rem 1rem;
}
.sv-greeting-title{
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--sv-primary-dark);
    line-height: 1.35;
    margin-bottom: 0.2rem;
}
.sv-greeting-sub{
    font-size: 1.05rem;
    color: #8a7b6c;
    margin-bottom: 1.8rem;
}

/* ---------------- Suggestion pill buttons ---------------- */
div[data-testid="stHorizontalBlock"] div.stButton > button{
    background-color: #FFFFFF;
    border: 1px solid var(--sv-secondary);
    color: var(--sv-primary-dark);
    border-radius: 999px;
    padding: 0.5rem 1.1rem;
    font-size: 0.85rem;
    font-weight: 500;
    box-shadow: 0 1px 3px rgba(139, 94, 60, 0.08);
    transition: all 0.15s ease-in-out;
    width: 100%;
}
div[data-testid="stHorizontalBlock"] div.stButton > button:hover{
    background-color: var(--sv-secondary);
    border-color: var(--sv-primary);
    color: var(--sv-primary-dark);
}

/* ---------------- Chat input: bulat mirip "Ask anything" ---------------- */
div[data-testid="stChatInput"]{
    border: 1px solid var(--sv-secondary);
    border-radius: 28px;
    background-color: #FFFFFF;
    box-shadow: 0 4px 16px rgba(139, 94, 60, 0.10);
}
div[data-testid="stChatInput"] textarea{
    color: var(--sv-primary-dark) !important;
}

/* ---------------- Chat bubbles ---------------- */
div[data-testid="stChatMessage"]{
    background-color: #FFFFFF;
    border: 1px solid rgba(220, 199, 170, 0.6);
    border-radius: 16px;
    padding: 0.4rem 0.8rem;
    box-shadow: 0 1px 4px rgba(139, 94, 60, 0.06);
}

/* ---------------- Tabs ---------------- */
button[data-baseweb="tab"]{
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"]{
    color: var(--sv-primary) !important;
}
div[data-baseweb="tab-highlight"]{
    background-color: var(--sv-primary) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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

st.markdown(
    """
    <div class="sv-header">
        <div class="sv-header-title">Savoria Command Center</div>
        <div class="sv-header-caption">Dashboard Multi-Agent AI untuk Manajemen Savoria Resto Group</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_monitor, tab_eval = st.tabs(["💬 Chat", "📊 Monitoring Cabang", "✅ Evaluasi Model"])

# ==================================================================
# TAB 1: CHAT
# ==================================================================
SUGGESTIONS = [
    ("📦", "Cek stok kritis", "Bahan apa yang paling kritis stoknya hari ini?"),
    ("📈", "Analisis omzet", "Bagaimana tren omzet 7 hari terakhir?"),
    ("👥", "Info shift karyawan", "Siapa saja yang shift hari ini?"),
    ("📋", "Cari SOP terkait", "Apa SOP untuk penanganan selisih kas?"),
]


def process_question(question: str, branch_id_selected):
    """Kirim pertanyaan ke graph multi-agent & simpan hasilnya ke riwayat chat."""
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
        "role": "assistant",
        "content": f"**[Agent {result['agent']}]**\n\n{result['answer']}",
        "sources": result["sop_sources"],
        "elapsed": elapsed,
    })


with tab_chat:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    header_left, header_right = st.columns([5, 1.4])
    with header_right:
        branch_filter = st.selectbox(
            "Cabang",
            options=["Semua Cabang"] + branches_df["branch_name"].tolist(),
            key="chat_branch_filter",
            label_visibility="collapsed",
        )
    branch_id_selected = None
    if branch_filter != "Semua Cabang":
        branch_id_selected = branches_df[branches_df["branch_name"] == branch_filter]["branch_id"].values[0]

    # ---------------------------------------------------------
    # LAYAR AWAL (belum ada percakapan): greeting + ask-bar + saran
    # ---------------------------------------------------------
    if not st.session_state.chat_history:
        st.markdown(
            f"""
            <div class="sv-greeting-wrap">
                <div class="sv-greeting-title">{get_greeting()}, Manajer 👋<br>Ada yang bisa dibantu hari ini?</div>
                <div class="sv-greeting-sub">Tanya apa saja soal stok, omzet, shift, atau SOP cabang Savoria.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        question = st.chat_input("Tanya apa saja...")

        cols = st.columns(len(SUGGESTIONS))
        for col, (icon, label, prompt_text) in zip(cols, SUGGESTIONS):
            with col:
                if st.button(f"{icon} {label}", key=f"sugg_{label}", use_container_width=True):
                    question = prompt_text

        if question:
            process_question(question, branch_id_selected)
            st.rerun()

    # ---------------------------------------------------------
    # LAYAR PERCAKAPAN (sudah ada riwayat chat)
    # ---------------------------------------------------------
    else:
        st.caption("Sistem akan otomatis merutekan pertanyaanmu ke Agent Divisi yang tepat "
                   "(Inventory / Order / HR / Finance).")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    if msg.get("elapsed") is not None:
                        st.caption(f"⏱️ Waktu respons: {msg['elapsed']:.1f} detik")
                    if msg.get("sources"):
                        with st.expander("Lihat sumber SOP yang dipakai"):
                            for src in msg["sources"]:
                                st.markdown(f"- {src}")

        question = st.chat_input("Tanya apa saja...")
        if question:
            process_question(question, branch_id_selected)
            st.rerun()

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