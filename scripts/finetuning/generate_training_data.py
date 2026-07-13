"""
generate_training_data.py
Membuat dataset berlabel untuk fine-tuning model klasifikasi kategori
pertanyaan (inventory/order/hr/finance/general).

Dataset ini dibuat manual berdasarkan variasi pertanyaan yang realistis
diajukan manajer ke sistem Savoria. Total ~200 contoh, cukup untuk melatih
model klasifikasi sederhana (TF-IDF + Logistic Regression).

Cara jalankan:
    python generate_training_data.py
"""

import pandas as pd
import random

random.seed(42)

# ------------------------------------------------------------------
# TEMPLATE PERTANYAAN PER KATEGORI
# Dibuat bervariasi (formal/informal, dengan/tanpa nama cabang) supaya
# model belajar pola bahasa yang beragam, bukan cuma hafal kata kunci.
# ------------------------------------------------------------------

branches = ["Malioboro", "Kaliurang", "Jakal", "Sudirman", "Prawirotaman", ""]

inventory_templates = [
    "Bahan apa yang paling kritis stoknya {b}?",
    "Stok {item} di cabang {b} masih cukup tidak?",
    "Kapan kita harus restock {item}?",
    "Ada bahan baku yang mau habis {b}?",
    "Tolong cek status stok gudang {b}",
    "Bahan baku apa saja yang perlu segera dipesan ulang?",
    "Berapa sisa stok minyak goreng di {b}?",
    "Apakah ada risiko kehabisan bahan baku minggu ini?",
    "Cek dong stok ayam di cabang {b}",
    "Bahan mana yang statusnya waspada sekarang?",
    "Info stok bahan baku terbaru dong",
    "Ada bahan yang mendekati kedaluwarsa tidak {b}?",
    "Perlu restock apa saja hari ini?",
    "Gudang {b} kekurangan bahan apa?",
    "Sisa stok beras berapa ya di {b}?",
]

order_templates = [
    "Menu apa yang paling laris {b}?",
    "Bagaimana distribusi order online vs offline {b}?",
    "Jam berapa paling ramai pesanan di {b}?",
    "Menu mana yang penjualannya menurun?",
    "Berapa banyak pesanan dari GoFood hari ini?",
    "Apa menu favorit pelanggan di cabang {b}?",
    "Ada komplain soal pesanan yang telat tidak?",
    "Bagaimana performa order lewat ShopeeFood?",
    "Menu terlaris minggu ini apa saja?",
    "Order dine-in atau online yang lebih banyak {b}?",
    "Cek dong volume pesanan jam sibuk {b}",
    "Apakah ada pembatalan pesanan online hari ini?",
    "Bagaimana tren penjualan menu baru?",
    "Kanal mana yang paling banyak dipakai pelanggan?",
    "Menu apa yang sebaiknya dipromosikan?",
]

hr_templates = [
    "Ada kekurangan staf shift malam {b}?",
    "Siapa saja yang jaga weekend ini {b}?",
    "Jadwal shift bentrok tidak minggu ini?",
    "Berapa jumlah karyawan yang libur hari ini?",
    "Cek dong komposisi staf shift pagi {b}",
    "Apakah shift malam Jumat sudah terisi penuh?",
    "Karyawan mana yang sudah 6 hari kerja berturut-turut?",
    "Butuh tambahan staf tidak untuk weekend ini?",
    "Bagaimana rotasi shift bulan ini?",
    "Siapa Chef yang bertugas hari ini {b}?",
    "Ada gap jadwal shift yang belum terisi?",
    "Berapa total karyawan aktif di cabang {b}?",
    "Cek jadwal cuti karyawan minggu ini",
    "Apakah komposisi staf sudah sesuai minimum SOP?",
    "Waiter mana saja yang shift siang hari ini?",
]

finance_templates = [
    "Ada selisih keuangan yang mencurigakan {b}?",
    "Berapa total omzet bulan ini {b}?",
    "Cek dong rekonsiliasi bank hari ini",
    "Apakah ada discrepancy di atas Rp 50.000?",
    "Bagaimana tren omzet dibanding bulan lalu?",
    "Kenapa laporan keuangan cabang {b} telat?",
    "Berapa total pendapatan dari GrabFood minggu ini?",
    "Cek selisih antara recorded sales dan bank deposit",
    "Apakah audit keuangan bulan ini sudah dilakukan?",
    "Omzet cabang mana yang paling tinggi?",
    "Ada potongan biaya platform yang belum tercatat?",
    "Bagaimana performa keuangan cabang {b} bulan ini?",
    "Cek laporan rekonsiliasi harian dong",
    "Berapa besar selisih pembayaran hari ini?",
    "Apakah ada indikasi kebocoran keuangan?",
]

general_templates = [
    "Halo, kamu siapa?",
    "Apa kabar hari ini?",
    "Bisa bantu apa saja kamu?",
    "Terima kasih atas bantuannya",
    "Restoran Savoria buka jam berapa?",
    "Bagaimana cuaca hari ini?",
    "Ceritakan sejarah kopi",
    "Apa rekomendasi liburan akhir pekan?",
    "Siapa pendiri Savoria?",
    "Apa filosofi nama Savoria?",
    "Bisa rekomendasikan resep masakan?",
    "Bagaimana cara membuat kue coklat?",
    "Apa itu machine learning?",
    "Tolong jelaskan cara kerja AI",
    "Apa menu favoritmu?",
]

ingredients_list = ["ayam", "beras", "minyak goreng", "daging sapi", "bumbu dasar", "telur", "gula", "kopi"]

CATEGORY_TEMPLATES = {
    "inventory": inventory_templates,
    "order": order_templates,
    "hr": hr_templates,
    "finance": finance_templates,
    "general": general_templates,
}


def fill_template(template: str) -> str:
    text = template
    if "{b}" in text:
        branch = random.choice(branches)
        branch_phrase = f"di Savoria {branch}" if branch else ""
        text = text.replace("{b}", branch_phrase)
    if "{item}" in text:
        text = text.replace("{item}", random.choice(ingredients_list))
    return " ".join(text.split())  # rapikan spasi ganda


def main():
    rows = []
    # Setiap template digandakan beberapa kali dengan variasi random,
    # supaya total dataset cukup untuk training (~200+ baris)
    n_variations = 6

    for category, templates in CATEGORY_TEMPLATES.items():
        for template in templates:
            for _ in range(n_variations):
                question = fill_template(template)
                rows.append({"question": question, "label": category})

    df = pd.DataFrame(rows).drop_duplicates(subset="question").reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    df.to_csv("training_data.csv", index=False)
    print(f"Dataset training berhasil dibuat: {len(df)} baris")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
