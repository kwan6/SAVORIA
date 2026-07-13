"""
evaluate.py
Script evaluasi kuantitatif untuk Savoria Multi-Agent System.

Mengukur 5 dimensi sesuai rubrik UAS:
    1. Accuracy       -> akurasi routing Supervisor (fine-tuned + fallback LLM)
                         vs label ground-truth (pakai test set berlabel)
    2. Effectiveness  -> coverage recall: seberapa banyak fakta penting dari
                         data aktual yang benar-benar disebutkan di jawaban
    3. Efficiency     -> latency supervisor vs latency total pipeline,
                         rate fallback ke LLM
    4. Explainability -> apakah jawaban menyebut sumber (SOP/data aktual),
                         jumlah dokumen SOP yang diretrieve, skor jarak retrieval
    5. Hallucination  -> grounding precision terbalik (fact-based) untuk jawaban
                         dengan cukup fakta angka/entitas, dilengkapi LLM-as-judge
                         untuk jawaban dengan fakta sedikit/tanpa fakta

Karena LLM (llama3.2, temperature=0.2) tidak deterministik, seluruh test set
dijalankan berkali-kali (default 3x, atur lewat --runs) lalu SEMUA hasilnya
digabung (pooled) sebelum dihitung metrik utamanya -> lebih stabil secara
statistik dibanding 1x run. Variansi antar-run tetap dilaporkan terpisah
di summary.json (key "run_to_run_variability") supaya transparan.

Cara jalankan (dari folder savoria_project, dengan Ollama sudah running):
    python scripts\\evaluate.py
    python scripts\\evaluate.py --runs 5      (opsional, ubah jumlah run)

Output disimpan di folder `evaluation_results/` (sejajar dengan chroma_db/):
    - detail.csv              -> hasil per test case, GABUNGAN semua run (kolom 'run' menandai asal run)
    - detail_run{N}.csv       -> hasil mentah per run individual
    - summary_run{N}.json     -> ringkasan metrik per run individual
    - summary.json            -> ringkasan metrik pooled (gabungan semua run) + variansi antar-run
    - confusion_matrix.png    -> visual confusion matrix routing (dari data pooled)
    - metrics_summary.png     -> bar chart ringkasan skor 5 dimensi (dari data pooled)

File-file ini lalu dibaca oleh tab "Evaluasi Model" di app.py.
"""

import os
import sys
import re
import json
import time
import statistics
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")  # supaya bisa jalan tanpa display (headless/Windows terminal)
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# SETUP PATH & IMPORT MODUL PROJECT
# ------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_THIS_DIR)
sys.path.append(os.path.join(_THIS_DIR, "finetuning"))

import agents                    # noqa: E402
import data_tools as dt          # noqa: E402
import graph_savoria             # noqa: E402
import supervisor_finetuned      # noqa: E402

RESULTS_DIR = os.path.join(_THIS_DIR, "..", "evaluation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

VALID_DIVISIONS = ["inventory", "order", "hr", "finance", "general"]

COLLECTION_MAP = {
    "inventory": "collection_inventory",
    "order": "collection_order",
    "hr": "collection_hr",
    "finance": "collection_finance",
}

# Kata kunci yang menandakan jawaban menyebutkan sumbernya (untuk Explainability)
CITATION_KEYWORDS = [
    "sop", "data aktual", "berdasarkan data", "berdasarkan sop", "sumber",
]

# Threshold grounding precision di bawah ini dianggap "berhalusinasi"
HALLUCINATION_THRESHOLD = 0.7

# Jawaban dengan jumlah fakta angka/entitas <= nilai ini dianggap TIDAK CUKUP
# untuk dinilai andal lewat overlap fakta (rawan false-negative, mis. cuma
# kebetulan 1 angka yang cocok padahal isi jawaban sebagian besar dikarang).
# Kasus seperti ini dialihkan ke LLM-as-judge. Sebelumnya threshold ini 0
# (hanya jawaban TANPA fakta sama sekali) - dinaikkan ke 1 setelah ditemukan
# kasus false-negative nyata di data (SOP cuti karyawan yang tetap lolos
# karena kebetulan 1 angka "1 minggu" cocok dengan konteks).
LOW_FACT_COUNT_THRESHOLD = 1

# Berapa kali seluruh evaluasi diulang untuk mengukur variansi antar-run.
# LLM (temperature=0.2 di agents.py) tidak deterministik, jadi 1x run saja
# tidak cukup untuk menyimpulkan suatu perubahan (mis. prompt hardening)
# benar-benar berefek, bukan cuma variasi acak. Semua run digabung (pooled)
# untuk metrik utama, plus dilaporkan variansi per-run secara terpisah.
# Bisa dioverride lewat argumen CLI: python evaluate.py --runs 5
N_EVAL_RUNS = 3


# ==================================================================
# TEST SET
# ------------------------------------------------------------------
# expected_divisi = label ground-truth untuk mengukur Accuracy routing.
# Query diadaptasi dari test_retrieval.py + variasi tambahan, plus
# beberapa query "general" untuk menguji penolakan out-of-scope.
# ==================================================================
def build_test_set(branch_id_sample: str = None) -> list:
    test_set = [
        # --- INVENTORY ---
        {"question": "Bahan apa yang paling kritis stoknya hari ini?", "expected_divisi": "inventory", "branch_id": None},
        {"question": "Kapan bahan baku harus direstock?", "expected_divisi": "inventory", "branch_id": None},
        {"question": "Bagaimana cara mencegah bahan baku terbuang?", "expected_divisi": "inventory", "branch_id": None},
        {"question": "Stok bahan baku mana saja yang statusnya waspada?", "expected_divisi": "inventory", "branch_id": branch_id_sample},
        {"question": "Apa SOP penanganan bahan baku yang mendekati kadaluarsa?", "expected_divisi": "inventory", "branch_id": None},

        # --- ORDER ---
        {"question": "Menu apa yang paling laris bulan ini?", "expected_divisi": "order", "branch_id": None},
        {"question": "Apa yang harus dilakukan saat jam sibuk?", "expected_divisi": "order", "branch_id": None},
        {"question": "Bagaimana urutan prioritas pesanan online vs offline?", "expected_divisi": "order", "branch_id": None},
        {"question": "Bagaimana distribusi kanal pemesanan di cabang ini?", "expected_divisi": "order", "branch_id": branch_id_sample},
        {"question": "Menu apa yang penjualannya paling rendah?", "expected_divisi": "order", "branch_id": None},

        # --- HR ---
        {"question": "Berapa minimal staf shift malam saat weekend?", "expected_divisi": "hr", "branch_id": None},
        {"question": "Apa aturan kalau jadwal shift bentrok?", "expected_divisi": "hr", "branch_id": None},
        {"question": "Apakah ada kekurangan staf shift malam akhir pekan?", "expected_divisi": "hr", "branch_id": branch_id_sample},
        {"question": "Bagaimana SOP pengajuan cuti karyawan?", "expected_divisi": "hr", "branch_id": None},

        # --- FINANCE ---
        {"question": "Berapa batas toleransi selisih rekonsiliasi kas?", "expected_divisi": "finance", "branch_id": None},
        {"question": "Apa penyebab umum selisih omzet harian?", "expected_divisi": "finance", "branch_id": None},
        {"question": "Adakah selisih keuangan yang perlu diwaspadai bulan ini?", "expected_divisi": "finance", "branch_id": branch_id_sample},
        {"question": "Bagaimana prosedur rekonsiliasi kas harian?", "expected_divisi": "finance", "branch_id": None},

        # --- GENERAL / OUT OF SCOPE ---
        {"question": "Bagaimana cuaca hari ini di Yogyakarta?", "expected_divisi": "general", "branch_id": None},
        {"question": "Siapa nama CEO Savoria?", "expected_divisi": "general", "branch_id": None},
        {"question": "Apa resep rahasia saus andalan restoran?", "expected_divisi": "general", "branch_id": None},
    ]
    return test_set


# ==================================================================
# HELPER: EKSTRAKSI FAKTA (angka + entitas bernama) UNTUK
# EFFECTIVENESS & HALLUCINATION
# ==================================================================
_NUMBER_RE = re.compile(r"\d[\d.,]*\d|\d")

# Penanda list bernomor ("1. ", "2) ") di AWAL BARIS -> dibuang dulu sebelum
# ekstraksi angka, supaya nomor urut list tidak ikut dihitung sebagai "fakta".
_LIST_MARKER_RE = re.compile(r"(?m)^\s*\d+[\.\)]\s+")


def _normalize_number(raw: str) -> str:
    """Hilangkan pemisah ribuan/desimal supaya '50.000' dan '50000' dianggap sama."""
    return re.sub(r"[.,]", "", raw)


def load_entity_vocab() -> list:
    """Ambil daftar nama bahan baku, menu, dan cabang sebagai kosakata entitas
    dari dataset aktual, dipakai untuk mendeteksi entitas bernama di jawaban."""
    vocab = set()
    try:
        vocab.update(dt._load("ingredients.csv")["ingredient_name"].astype(str).str.strip())
    except Exception:
        pass
    try:
        vocab.update(dt._load("menu.csv")["menu_name"].astype(str).str.strip())
    except Exception:
        pass
    try:
        vocab.update(dt._load("branches.csv")["branch_name"].astype(str).str.strip())
    except Exception:
        pass
    # Buang string kosong / terlalu pendek (rawan false-positive)
    return [v for v in vocab if len(v) >= 3]


def extract_facts(text: str, vocab: list) -> set:
    """Ekstrak 'fakta' dari sebuah teks: angka (harga, persentase, jumlah)
    dan entitas bernama (bahan/menu/cabang) yang disebut di teks tersebut.

    CATATAN PERBAIKAN: versi awal membuang angka 1 digit untuk menghindari
    noise dari penomoran list ("1. ", "2. "), tapi ini juga ikut membuang
    fakta 1-digit yang justru penting di domain ini (mis. "2 Chef", "3
    Waiter", "minimal 5 staf"). Sekarang penomoran list dibersihkan dulu
    secara eksplisit (_LIST_MARKER_RE), baru semua angka (termasuk 1 digit)
    dihitung sebagai fakta.
    """
    if not text:
        return set()

    text_clean = _LIST_MARKER_RE.sub("", text)

    facts = set()

    # Angka
    for match in _NUMBER_RE.findall(text_clean):
        norm = _normalize_number(match)
        if norm:
            facts.add(f"num:{norm}")

    # Entitas bernama (case-insensitive substring match)
    text_lower = text_clean.lower()
    for entity in vocab:
        if entity.lower() in text_lower:
            facts.add(f"ent:{entity.lower()}")

    return facts


def fact_precision_recall(answer_text: str, context_text: str, vocab: list):
    """
    precision = proporsi fakta di JAWABAN yang benar-benar ada di KONTEKS
                -> dipakai untuk Hallucination (1 - precision)
    recall    = proporsi fakta di KONTEKS yang berhasil disebut di JAWABAN
                -> dipakai untuk Effectiveness (coverage)
    """
    answer_facts = extract_facts(answer_text, vocab)
    context_facts = extract_facts(context_text, vocab)
    overlap = answer_facts & context_facts

    precision = (len(overlap) / len(answer_facts)) if answer_facts else np.nan
    recall = (len(overlap) / len(context_facts)) if context_facts else np.nan

    return precision, recall, len(answer_facts), len(context_facts), len(overlap)


def has_citation(answer_text: str) -> bool:
    text_lower = (answer_text or "").lower()
    return any(kw in text_lower for kw in CITATION_KEYWORDS)


# ------------------------------------------------------------------
# LLM-AS-JUDGE (lapis kedua untuk Hallucination)
# ------------------------------------------------------------------
# Metode fact-based (angka/entitas) TIDAK BISA menilai jawaban yang isinya
# murni naratif/prosedural tanpa angka atau nama spesifik (mis. jawaban
# berupa langkah-langkah SOP yang dikarang tanpa didukung konteks apa pun).
# Untuk kasus seperti ini (n_answer_facts == 0), dipakai LLM sebagai
# evaluator tambahan untuk menilai apakah jawaban benar-benar bersumber
# dari konteks yang diberikan.
_JUDGE_PROMPT_TEMPLATE = """Kamu adalah evaluator yang ketat dan objektif.
Tugasmu: menilai apakah JAWABAN di bawah ini murni didasarkan pada KONTEKS
yang diberikan, atau berisi informasi/prosedur yang TIDAK ada di konteks
tersebut (dikarang oleh model).

CATATAN PENTING: Jika di awal KONTEKS ada baris "(Nama file SOP yang sah
untuk kutipan: ...)", maka menyebut salah satu nama file itu di JAWABAN
BUKAN fabrikasi - itu kutipan yang valid. Yang dianggap fabrikasi adalah
kalau JAWABAN menyebut nama dokumen/judul SOP LAIN yang TIDAK ada di
daftar nama file sah tersebut (mis. membuat judul resmi sendiri).

=== KONTEKS (SOP + data aktual) ===
{context}

=== JAWABAN YANG DINILAI ===
{answer}

Jawab HANYA dengan SATU kata berikut, tanpa penjelasan tambahan:
- GROUNDED     -> jika seluruh isi jawaban bisa ditelusuri ke konteks di atas
                  (termasuk kalau nama file yang disebut memang ada di daftar sah)
- NOT_GROUNDED -> jika ada bagian jawaban yang mengarang informasi/prosedur/
                  nama dokumen yang tidak ada di konteks atau daftar nama sah
- UNCERTAIN    -> jika kamu tidak yakin

Jawaban (satu kata saja):"""


def llm_judge_grounding(answer_text: str, context_text: str) -> str:
    """Return 'grounded', 'not_grounded', atau 'uncertain'."""
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        context=context_text[:2500] if context_text else "(kosong)",
        answer=answer_text,
    )
    try:
        response = agents._llm.invoke(prompt)
        verdict = response.content.strip().upper()
        if "NOT_GROUNDED" in verdict or "NOT GROUNDED" in verdict:
            return "not_grounded"
        elif "GROUNDED" in verdict:
            return "grounded"
        else:
            return "uncertain"
    except Exception:
        return "uncertain"


# ==================================================================
# HELPER: AMBIL KONTEKS PENUH (SOP + DATA) UNTUK 1 TEST CASE
# (dipisah dari agents.py supaya dapat teks SOP LENGKAP, bukan yang
# dipotong 100 karakter seperti di `sop_sources`)
# ==================================================================
def get_full_context(divisi: str, question: str, branch_id: str):
    if divisi not in COLLECTION_MAP:
        return "", 0, []

    collection_name = COLLECTION_MAP[divisi]
    docs, distances = agents._retrieve_with_scores(collection_name, question)
    sop_context_full = "\n---\n".join(d.page_content for d in docs)
    source_filenames = agents._get_source_filenames(docs)

    if divisi == "inventory":
        data_context = dt.get_critical_stock(branch_id=branch_id)
    elif divisi == "order":
        data_context = f"{dt.get_top_menu(branch_id=branch_id)}\n\n{dt.get_channel_distribution(branch_id=branch_id)}"
    elif divisi == "hr":
        data_context = dt.get_shift_gaps(branch_id=branch_id)
    elif divisi == "finance":
        data_context = dt.get_finance_discrepancy(branch_id=branch_id)
    else:
        data_context = ""

    # Nama file SOP yang sah disisipkan eksplisit di awal konteks, supaya
    # LLM-judge tidak salah menganggap kutipan nama file yang VALID sebagai
    # fabrikasi (dulu ini jadi false-positive besar karena judge cuma lihat
    # isi teks chunk, tidak tahu nama file resminya).
    source_note = (
        f"(Nama file SOP yang sah untuk kutipan: {', '.join(source_filenames)})\n\n"
        if source_filenames else ""
    )
    full_context = f"{source_note}{sop_context_full}\n\n{data_context}"
    return full_context, len(docs), distances


# ==================================================================
# MAIN EVALUATION LOOP
# ==================================================================
def run_evaluation():
    print("=== EVALUASI MODEL SAVORIA ===")
    print("Memuat resources (graph, embeddings, vocab)...")

    app_graph = graph_savoria.build_graph()
    vocab = load_entity_vocab()

    try:
        branches_df = dt._load("branches.csv")
        branch_id_sample = branches_df["branch_id"].iloc[0] if not branches_df.empty else None
    except Exception:
        branch_id_sample = None

    test_set = build_test_set(branch_id_sample=branch_id_sample)
    rows = []

    for i, case in enumerate(test_set, 1):
        question = case["question"]
        expected_divisi = case["expected_divisi"]
        branch_id = case["branch_id"]

        print(f"[{i}/{len(test_set)}] {question}")

        row = {
            "question": question,
            "branch_id": branch_id,
            "expected_divisi": expected_divisi,
        }

        try:
            # --- 1. Supervisor fine-tuned saja (untuk ukur efficiency & akurasi tahap awal) ---
            t0 = time.time()
            ft_result = supervisor_finetuned.classify_with_finetuned_model(question)
            latency_supervisor = time.time() - t0

            row["finetuned_divisi"] = ft_result["divisi"]
            row["confidence"] = ft_result["confidence"]
            row["fallback_used"] = ft_result["needs_llm_fallback"]
            row["latency_supervisor_sec"] = latency_supervisor

            # --- 2. Full pipeline (supervisor + agent) ---
            state = {"question": question, "branch_id": branch_id, "divisi": None, "result": None}
            t0 = time.time()
            final_state = app_graph.invoke(state)
            latency_total = time.time() - t0

            result = final_state["result"]
            final_divisi = final_state["divisi"]

            row["predicted_divisi"] = final_divisi
            row["correct"] = int(final_divisi == expected_divisi)
            row["latency_total_sec"] = latency_total
            row["agent"] = result["agent"]
            row["answer"] = result["answer"]
            row["n_sop_docs_preview"] = len(result.get("sop_sources", []))
            row["citation_present"] = has_citation(result["answer"])

            # --- 3. Konteks penuh untuk Effectiveness & Hallucination ---
            if final_divisi in COLLECTION_MAP:
                full_context, n_docs, distances = get_full_context(final_divisi, question, branch_id)
                row["n_sop_docs_full"] = n_docs
                row["avg_retrieval_distance"] = float(np.mean(distances)) if distances else np.nan

                precision, recall, n_ans_facts, n_ctx_facts, n_overlap = fact_precision_recall(
                    result["answer"], full_context, vocab
                )
                row["grounding_precision"] = precision
                row["coverage_recall"] = recall
                row["n_answer_facts"] = n_ans_facts
                row["n_context_facts"] = n_ctx_facts
                row["n_overlap_facts"] = n_overlap

                if n_ans_facts <= LOW_FACT_COUNT_THRESHOLD:
                    # Blind spot metode fact-based: jawaban dengan fakta
                    # angka/entitas yang SEDIKIT (termasuk 0 atau cuma 1 yang
                    # kebetulan cocok) tidak bisa dinilai secara andal lewat
                    # overlap fakta. Pakai LLM-as-judge di sini.
                    verdict = llm_judge_grounding(result["answer"], full_context)
                    row["llm_judge_verdict"] = verdict
                    row["hallucination_method"] = "llm_judge"
                    row["hallucinated"] = (
                        1 if verdict == "not_grounded"
                        else (0 if verdict == "grounded" else np.nan)
                    )
                else:
                    row["llm_judge_verdict"] = None
                    row["hallucination_method"] = "fact_grounding"
                    row["hallucinated"] = (
                        int(precision < HALLUCINATION_THRESHOLD) if not np.isnan(precision) else np.nan
                    )
            else:
                # divisi "general" -> tidak ada konteks SOP/data untuk dinilai
                row["n_sop_docs_full"] = 0
                row["avg_retrieval_distance"] = np.nan
                row["grounding_precision"] = np.nan
                row["coverage_recall"] = np.nan
                row["n_answer_facts"] = np.nan
                row["n_context_facts"] = np.nan
                row["n_overlap_facts"] = np.nan
                row["llm_judge_verdict"] = None
                row["hallucination_method"] = "no_context"
                row["hallucinated"] = np.nan

            row["error"] = None

        except Exception as e:
            print(f"  [ERROR] {e}")
            row["error"] = str(e)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ==================================================================
# AGREGASI 5 DIMENSI
# ==================================================================
def aggregate_metrics(df: pd.DataFrame) -> dict:
    df_ok = df[df["error"].isna()].copy()

    # ---------------- 1. ACCURACY (routing) ----------------
    y_true = df_ok["expected_divisi"]
    y_pred = df_ok["predicted_divisi"]

    overall_accuracy = accuracy_score(y_true, y_pred)
    labels_present = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_present, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels_present)
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)

    accuracy_metrics = {
        "overall_accuracy": float(overall_accuracy),
        "per_class": per_class,
        "confusion_matrix": {"labels": labels_present, "matrix": cm.tolist()},
    }

    # ---------------- 2. EFFECTIVENESS (coverage recall) ----------------
    coverage_valid = df_ok["coverage_recall"].dropna()
    effectiveness_metrics = {
        "avg_context_coverage_recall": float(coverage_valid.mean()) if not coverage_valid.empty else None,
        "per_division": {
            div: float(g["coverage_recall"].dropna().mean())
            for div, g in df_ok.groupby("expected_divisi")
            if not g["coverage_recall"].dropna().empty
        },
        "n_cases_evaluated": int(coverage_valid.shape[0]),
    }

    # ---------------- 3. EFFICIENCY (latency) ----------------
    lat_total = df_ok["latency_total_sec"].dropna()
    lat_sup = df_ok["latency_supervisor_sec"].dropna()
    efficiency_metrics = {
        "avg_latency_total_sec": float(lat_total.mean()) if not lat_total.empty else None,
        "median_latency_total_sec": float(lat_total.median()) if not lat_total.empty else None,
        "p95_latency_total_sec": float(np.percentile(lat_total, 95)) if not lat_total.empty else None,
        "avg_latency_supervisor_sec": float(lat_sup.mean()) if not lat_sup.empty else None,
        "fallback_rate": float(df_ok["fallback_used"].mean()) if "fallback_used" in df_ok else None,
    }

    # ---------------- 4. EXPLAINABILITY (citation & retrieval) ----------------
    div_cases = df_ok[df_ok["expected_divisi"] != "general"]
    explainability_metrics = {
        "citation_rate": float(df_ok["citation_present"].mean()) if not df_ok.empty else None,
        "avg_sop_docs_retrieved": float(div_cases["n_sop_docs_full"].mean()) if not div_cases.empty else None,
        "avg_retrieval_distance": (
            float(div_cases["avg_retrieval_distance"].dropna().mean())
            if not div_cases["avg_retrieval_distance"].dropna().empty else None
        ),
        "note": "avg_retrieval_distance: semakin KECIL semakin relevan (jarak vektor Chroma).",
    }

    # ---------------- 5. HALLUCINATION (grounding precision + LLM-as-judge) ----------------
    precision_valid = df_ok["grounding_precision"].dropna()
    hallucinated_valid = df_ok["hallucinated"].dropna()

    fact_based = df_ok[df_ok.get("hallucination_method") == "fact_grounding"]
    judge_based = df_ok[df_ok.get("hallucination_method") == "llm_judge"]
    judge_verdicts = judge_based["llm_judge_verdict"].value_counts().to_dict() if not judge_based.empty else {}

    hallucination_metrics = {
        "avg_grounding_precision": float(precision_valid.mean()) if not precision_valid.empty else None,
        "hallucination_rate": float(hallucinated_valid.mean()) if not hallucinated_valid.empty else None,
        "threshold_used": HALLUCINATION_THRESHOLD,
        "n_cases_evaluated": int(hallucinated_valid.shape[0]),
        "detection_breakdown": {
            "n_cases_fact_based": int(fact_based.shape[0]),
            "n_cases_llm_judge_based": int(judge_based.shape[0]),
            "llm_judge_verdicts": judge_verdicts,
        },
        "note": (
            "Jawaban dengan >=1 fakta angka/entitas dinilai via overlap fakta "
            "(fact_grounding); jawaban tanpa fakta numerik/entitas sama sekali "
            "(mis. jawaban prosedural murni) dinilai via LLM-as-judge."
        ),
    }

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_test_cases": int(df.shape[0]),
        "n_errors": int(df["error"].notna().sum()),
        "accuracy": accuracy_metrics,
        "effectiveness": effectiveness_metrics,
        "efficiency": efficiency_metrics,
        "explainability": explainability_metrics,
        "hallucination": hallucination_metrics,
    }
    return summary


# ==================================================================
# VISUALISASI
# ==================================================================
def plot_confusion_matrix(summary: dict, out_path: str):
    labels = summary["accuracy"]["confusion_matrix"]["labels"]
    matrix = np.array(summary["accuracy"]["confusion_matrix"]["matrix"])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Divisi")
    ax.set_ylabel("Expected Divisi")
    ax.set_title("Confusion Matrix - Supervisor Routing")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                     color="white" if matrix[i, j] > matrix.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_summary(summary: dict, out_path: str):
    dims = ["Accuracy", "Effectiveness", "Efficiency*", "Explainability", "Hallucination**"]
    values = [
        summary["accuracy"]["overall_accuracy"],
        summary["effectiveness"]["avg_context_coverage_recall"] or 0,
        None,  # efficiency ditampilkan terpisah (bukan skor 0-1)
        summary["explainability"]["citation_rate"] or 0,
        1 - (summary["hallucination"]["hallucination_rate"] or 0),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_dims = [d for d, v in zip(dims, values) if v is not None]
    plot_vals = [v for v in values if v is not None]

    bars = ax.bar(plot_dims, plot_vals, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Skor (0-1)")
    ax.set_title("Ringkasan Skor Evaluasi Model Savoria")
    for bar, val in zip(bars, plot_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center")

    ax.text(
        0, -0.22,
        "*Efficiency diukur dalam detik latency (lihat summary.json), bukan skor 0-1.\n"
        "**Hallucination ditampilkan sebagai (1 - hallucination_rate) = tingkat kebenaran fakta.",
        transform=ax.transAxes, fontsize=8, color="gray",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ==================================================================
# ENTRY POINT
# ==================================================================
def _variability_stats(per_run_summaries: list, path: list):
    """Ambil satu nilai numerik dari tiap summary per-run (mengikuti `path`,
    mis. ['accuracy', 'overall_accuracy']), lalu hitung list nilai + std-nya."""
    values = []
    for s in per_run_summaries:
        v = s
        for key in path:
            v = v.get(key) if isinstance(v, dict) else None
            if v is None:
                break
        if v is not None:
            values.append(v)

    std = float(statistics.stdev(values)) if len(values) >= 2 else None
    return {"values_per_run": values, "std": std}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluasi Model Savoria")
    parser.add_argument(
        "--runs", type=int, default=N_EVAL_RUNS,
        help=f"Jumlah pengulangan evaluasi untuk stabilitas statistik (default: {N_EVAL_RUNS})",
    )
    args = parser.parse_args()
    n_runs = max(1, args.runs)

    print(f"Evaluasi akan dijalankan {n_runs}x dan hasilnya digabung (pooled) "
          f"untuk mengurangi variasi akibat temperature LLM > 0.\n")

    all_dfs = []
    per_run_summaries = []

    for run_idx in range(1, n_runs + 1):
        print(f"\n########## RUN {run_idx}/{n_runs} ##########")
        df_run = run_evaluation()
        df_run["run"] = run_idx
        all_dfs.append(df_run)

        run_summary = aggregate_metrics(df_run)
        per_run_summaries.append(run_summary)

        df_run.to_csv(os.path.join(RESULTS_DIR, f"detail_run{run_idx}.csv"), index=False)
        with open(os.path.join(RESULTS_DIR, f"summary_run{run_idx}.json"), "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2, ensure_ascii=False)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    detail_path = os.path.join(RESULTS_DIR, "detail.csv")
    combined_df.to_csv(detail_path, index=False)
    print(f"\nDetail hasil (gabungan {n_runs} run) disimpan -> {detail_path}")

    # Metrik utama dihitung dari data GABUNGAN (pooled) semua run -> lebih
    # stabil secara statistik dibanding cuma 1 run.
    summary = aggregate_metrics(combined_df)

    # Tambahan: variasi antar-run untuk metrik-metrik utama, supaya transparan
    # seberapa besar fluktuasi yang murni disebabkan temperature LLM > 0.
    summary["run_to_run_variability"] = {
        "n_runs": n_runs,
        "accuracy": _variability_stats(per_run_summaries, ["accuracy", "overall_accuracy"]),
        "effectiveness_coverage": _variability_stats(per_run_summaries, ["effectiveness", "avg_context_coverage_recall"]),
        "efficiency_avg_latency_sec": _variability_stats(per_run_summaries, ["efficiency", "avg_latency_total_sec"]),
        "explainability_citation_rate": _variability_stats(per_run_summaries, ["explainability", "citation_rate"]),
        "hallucination_rate": _variability_stats(per_run_summaries, ["hallucination", "hallucination_rate"]),
        "note": (
            "Nilai per-dimensi di atas dihitung dari POOLED data (semua run "
            "digabung), sementara 'values_per_run' & 'std' di sini menunjukkan "
            "seberapa jauh tiap run individual menyimpang -> std besar berarti "
            "hasil masih cukup sensitif terhadap randomness LLM."
        ),
    }

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Ringkasan metrik (pooled, {n_runs} run) disimpan -> {summary_path}")

    try:
        plot_confusion_matrix(summary, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
        plot_metrics_summary(summary, os.path.join(RESULTS_DIR, "metrics_summary.png"))
        print("Grafik disimpan -> confusion_matrix.png, metrics_summary.png")
    except Exception as e:
        print(f"[WARNING] Gagal membuat grafik: {e}")

    print(f"\n=== RINGKASAN EVALUASI (POOLED, {n_runs} RUN) ===")
    print(f"1. Accuracy (routing)         : {summary['accuracy']['overall_accuracy']:.2%}")
    eff = summary["effectiveness"]["avg_context_coverage_recall"]
    print(f"2. Effectiveness (coverage)   : {eff:.2%}" if eff is not None else "2. Effectiveness (coverage)   : N/A")
    print(f"3. Efficiency (avg latency)   : {summary['efficiency']['avg_latency_total_sec']:.2f} detik/query")
    print(f"   Fallback ke LLM rate       : {summary['efficiency']['fallback_rate']:.2%}")
    print(f"4. Explainability (citation)  : {summary['explainability']['citation_rate']:.2%}")
    hr = summary["hallucination"]["hallucination_rate"]
    print(f"5. Hallucination rate         : {hr:.2%}" if hr is not None else "5. Hallucination rate         : N/A")

    hr_std = summary["run_to_run_variability"]["hallucination_rate"]["std"]
    if hr_std is not None:
        print(f"\n   (variasi antar-run untuk hallucination rate: std = {hr_std:.2%} "
              f"dari {n_runs} run terpisah)")

    print("\nSelesai. Buka tab 'Evaluasi Model' di app.py untuk melihat visualisasinya.")


if __name__ == "__main__":
    main()