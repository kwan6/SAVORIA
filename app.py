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

from pathlib import Path

BASE_DIR = Path(__file__).parent
CSS_FILE = BASE_DIR / "assets" / "styles.css"

css = Path("assets/styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


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

        <div class="sv-header-title">
            Savoria Command Center
        </div>

        <div class="sv-header-caption">
            Dashboard Multi-Agent AI untuk Manajemen Operasional Savoria Resto Group
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_monitor, tab_eval = st.tabs([
    "Chat",
    "Monitoring",
    "Evaluasi"
])

# ==================================================================
# TAB 1: CHAT
# ==================================================================
SUGGESTIONS = [

    (
        "📦",
        "Cek stok kritis",
        "Lihat bahan baku yang stoknya hampir habis.",
        "Bahan apa yang paling kritis stoknya hari ini?"
    ),

    (
        "📈",
        "Analisis omzet",
        "Analisis performa penjualan cabang.",
        "Bagaimana tren omzet 7 hari terakhir?"
    ),

    (
        "👥",
        "Info Shift",
        "Informasi jadwal karyawan.",
        "Siapa saja yang shift hari ini?"
    ),

    (
        "📋",
        "Cari SOP",
        "Temukan SOP yang relevan.",
        "Apa SOP untuk penanganan selisih kas?"
    )

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

            <div class="sv-greeting-title">
                {get_greeting()}, Manajer
            </div>

            <div class="sv-greeting-sub">
                AI siap membantu operasional seluruh cabang Savoria.<br>
                Analisis stok, omzet, shift, SOP, hingga performa bisnis dalam satu dashboard.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
            f"""
            <div class="sv-dashboard-grid">
                <div class="sv-card">
                    <div class="sv-card-label">
                        TOTAL CABANG
                    </div>
                    <div class="sv-card-value">
                        {len(branches_df)}
                    </div>
                    <div class="sv-card-sub">
                        Cabang Aktif
                    </div>

                </div>

                <div class="sv-card">
                    <div class="sv-card-label">
                        AI AGENT
                    </div>
                    <div class="sv-card-value">
                        5
                    </div>
                    <div class="sv-card-sub">
                        Agent Online
                    </div>

                </div>

                <div class="sv-card">
                    <div class="sv-card-label">
                        DOKUMEN SOP
                    </div>
                    <div class="sv-card-value">
                        248
                    </div>
                    <div class="sv-card-sub">
                        SOP Tersedia
                    </div>

                </div>

                <div class="sv-card">
                    <div class="sv-card-label">
                        STATUS
                    </div>
                    <div class="sv-card-value">
                        Online
                    </div>
                    <div class="sv-card-sub">
                        Semua sistem normal
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
            )
    
    st.markdown(
        """
        <div style="
        text-align:center;
        margin-top:8px;
        margin-bottom:26px;
        color:#8a7b6c;
        font-size:14px;
        ">

        Seluruh agent AI aktif dan siap membantu analisis operasional.

        </div>
        """,
        unsafe_allow_html=True
        )
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
                """
        <div style="
        text-align:center;
        font-size:15px;
        color:#8a7b6c;
        margin-bottom:14px;
        ">

        Apa yang ingin Anda analisis hari ini?

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="
        text-align:center;
        font-size:15px;
        font-weight:600;
        color:#8B5E3C;
        margin-bottom:18px;
        ">

        Aksi Cepat

        </div>
        """,
        unsafe_allow_html=True
        )
    cols = st.columns(len(SUGGESTIONS))
    for col, (icon, label, prompt_text) in zip(cols, SUGGESTIONS):

            with col:

                if st.button(
                    f"{icon} {label}",
                    key=f"sugg_{label}",
                    use_container_width=True
                ):
                    question = prompt_text

    st.markdown("<br>", unsafe_allow_html=True)

    question = st.chat_input(
            "Contoh: Bagaimana omzet minggu ini di Cabang Malioboro?"
        )

    for col, (icon, label, prompt_text) in zip(cols, SUGGESTIONS):

        with col:

            if st.button(
                f"{icon} {label}",
                key=f"sugg_{label}",
                use_container_width=True
            ):

                question = prompt_text

    if question:

        process_question(
            question,
            branch_id_selected
        )

        st.rerun()

    # ---------------------------------------------------------
    # LAYAR PERCAKAPAN (sudah ada riwayat chat)
    # ---------------------------------------------------------
    else:
        st.info(
            "AI akan memilih agent yang paling sesuai secara otomatis berdasarkan konteks pertanyaan Anda."
        )

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