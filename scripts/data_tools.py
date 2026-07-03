"""
data_tools.py
Kumpulan fungsi untuk mengambil data AKTUAL dari file CSV (bukan dari SOP/RAG).
Fungsi-fungsi ini dipakai oleh masing-masing agent sebagai "tools" untuk
menjawab pertanyaan berbasis data real, dikombinasikan dengan aturan dari SOP (RAG).
"""

import pandas as pd
import os

# Path dihitung berdasarkan LOKASI FILE ini (bukan working directory saat
# script dijalankan), supaya tetap benar baik dipanggil dari scripts/
# (mis. graph_savoria.py) maupun dari root project (mis. app.py).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(_THIS_DIR, "..", "..", "savoria_dataset")


def _load(filename: str) -> pd.DataFrame:
    path = os.path.join(DATASET_DIR, filename)
    return pd.read_csv(path)


# ------------------------------------------------------------------
# INVENTORY TOOLS
# ------------------------------------------------------------------
def get_critical_stock(branch_id: str = None, top_n: int = 5) -> str:
    """Cari bahan baku dengan stok paling kritis (persentase sisa terendah)."""
    df = _load("inventory_daily.csv")
    ingredients = _load("ingredients.csv")
    branches = _load("branches.csv")

    latest_date = df["date"].max()
    df_latest = df[df["date"] == latest_date].copy()

    if branch_id:
        df_latest = df_latest[df_latest["branch_id"] == branch_id]

    df_latest["pct_remaining"] = (df_latest["stock_remaining"] / df_latest["stock_start"] * 100).round(1)
    df_latest = df_latest.merge(ingredients, on="ingredient_id").merge(branches, on="branch_id")
    df_latest = df_latest.sort_values("pct_remaining").head(top_n)

    if df_latest.empty:
        return f"Tidak ada data stok untuk tanggal {latest_date}."

    lines = [f"Data stok per {latest_date}:"]
    for _, row in df_latest.iterrows():
        status = "KRITIS" if row["pct_remaining"] < 10 else ("WASPADA" if row["pct_remaining"] < 25 else "AMAN")
        lines.append(
            f"- {row['ingredient_name']} di {row['branch_name']}: "
            f"sisa {row['stock_remaining']} {row['unit']} ({row['pct_remaining']}%) -> status {status}"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------
# ORDER TOOLS
# ------------------------------------------------------------------
def get_top_menu(branch_id: str = None, top_n: int = 5) -> str:
    """Cari menu paling laris berdasarkan jumlah qty terjual."""
    df = _load("transactions.csv")
    menu = _load("menu.csv")
    branches = _load("branches.csv")

    if branch_id:
        df = df[df["branch_id"] == branch_id]

    top = (
        df.groupby("menu_id")["qty"]
        .sum()
        .reset_index()
        .sort_values("qty", ascending=False)
        .head(top_n)
        .merge(menu, on="menu_id")
    )

    branch_name = branches[branches["branch_id"] == branch_id]["branch_name"].values[0] if branch_id else "Semua Cabang"
    lines = [f"Menu terlaris di {branch_name}:"]
    for _, row in top.iterrows():
        lines.append(f"- {row['menu_name']}: {row['qty']} porsi terjual")
    return "\n".join(lines)


def get_channel_distribution(branch_id: str = None) -> str:
    """Distribusi order berdasarkan kanal (Offline/GoFood/GrabFood/ShopeeFood)."""
    df = _load("transactions.csv")
    if branch_id:
        df = df[df["branch_id"] == branch_id]

    dist = df["order_channel"].value_counts(normalize=True).mul(100).round(1)
    lines = ["Distribusi kanal pemesanan:"]
    for channel, pct in dist.items():
        lines.append(f"- {channel}: {pct}%")
    return "\n".join(lines)


# ------------------------------------------------------------------
# HR TOOLS
# ------------------------------------------------------------------
def get_shift_gaps(branch_id: str = None) -> str:
    """Deteksi hari dengan jumlah staf shift malam di bawah minimum (khusus weekend)."""
    shifts = _load("shifts.csv")
    branches = _load("branches.csv")

    if branch_id:
        shifts = shifts[shifts["branch_id"] == branch_id]

    shifts["date"] = pd.to_datetime(shifts["date"])
    shifts["is_weekend"] = shifts["date"].dt.weekday >= 4  # Jumat=4,Sabtu=5,Minggu=6
    malam = shifts[shifts["shift"].str.contains("Malam") & shifts["is_weekend"]]

    counts = malam.groupby(["date", "branch_id"]).size().reset_index(name="jumlah_staf")
    gaps = counts[counts["jumlah_staf"] < 5]  # minimal 2 chef + 3 waiter = 5
    gaps = gaps.merge(branches, on="branch_id")

    if gaps.empty:
        return "Tidak ditemukan kekurangan staf shift malam weekend pada periode data."

    lines = ["Ditemukan kekurangan staf shift malam saat weekend:"]
    for _, row in gaps.iterrows():
        lines.append(
            f"- {row['date'].strftime('%Y-%m-%d')} di {row['branch_name']}: "
            f"hanya {row['jumlah_staf']} staf (minimal 5)"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------
# FINANCE TOOLS
# ------------------------------------------------------------------
def get_finance_discrepancy(branch_id: str = None, threshold: int = 50000) -> str:
    """Cari hari dengan selisih rekonsiliasi (discrepancy) di atas ambang batas."""
    df = _load("finance_daily.csv")
    branches = _load("branches.csv")

    if branch_id:
        df = df[df["branch_id"] == branch_id]

    flagged = df[df["discrepancy"].abs() > threshold].merge(branches, on="branch_id")

    if flagged.empty:
        return f"Tidak ada selisih di atas Rp {threshold:,} pada periode data."

    lines = [f"Selisih rekonsiliasi di atas Rp {threshold:,}:"]
    for _, row in flagged.iterrows():
        lines.append(
            f"- {row['date']} di {row['branch_name']}: selisih Rp {row['discrepancy']:,}"
        )
    return "\n".join(lines)


def get_branch_id_by_name(name_query: str) -> str:
    """Helper: cari branch_id dari nama cabang (partial match, case-insensitive)."""
    branches = _load("branches.csv")
    match = branches[branches["branch_name"].str.contains(name_query, case=False, na=False)]
    if match.empty:
        return None
    return match.iloc[0]["branch_id"]
