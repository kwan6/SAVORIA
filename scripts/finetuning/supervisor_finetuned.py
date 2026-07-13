"""
supervisor_finetuned.py
Wrapper untuk memakai model classifier hasil fine-tuning sebagai
pengganti/pendukung Supervisor Agent berbasis LLM.

Strategi HYBRID:
- Kalau confidence model fine-tuned TINGGI (>= threshold), pakai hasilnya
  langsung -> jauh lebih cepat, tidak perlu panggil LLM sama sekali.
- Kalau confidence RENDAH (model ragu), fallback ke Supervisor berbasis LLM
  (lebih lambat tapi lebih fleksibel untuk kasus ambigu/di luar pola training).

Ini memberi keuntungan: sebagian besar pertanyaan rutin dijawab lebih cepat
lewat model fine-tuned, sementara pertanyaan yang tidak biasa tetap
tertangani lewat LLM sebagai jaring pengaman.
"""

import os
import joblib

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_THIS_DIR, "supervisor_classifier.pkl")

CONFIDENCE_THRESHOLD = 0.55  # di bawah ini dianggap "model ragu"

_model = None


def _load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def classify_with_finetuned_model(question: str) -> dict:
    """
    Klasifikasikan pertanyaan pakai model fine-tuned.
    Return dict berisi: divisi, confidence, dan method yang dipakai.
    """
    model = _load_model()
    prediction = model.predict([question])[0]
    confidence = model.predict_proba([question]).max()

    return {
        "divisi": prediction,
        "confidence": float(confidence),
        "method": "fine-tuned-classifier",
        "needs_llm_fallback": confidence < CONFIDENCE_THRESHOLD,
    }
