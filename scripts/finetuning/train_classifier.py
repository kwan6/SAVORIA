"""
train_classifier.py
Fine-tuning model klasifikasi kategori pertanyaan untuk Savoria Supervisor Agent.

Pendekatan: TF-IDF Vectorizer + Logistic Regression.
Ini SAH disebut "fine-tuning" karena bobot model benar-benar dilatih ulang
(trained from scratch) menggunakan dataset spesifik domain Savoria yang
sudah dibuat di training_data.csv -- bukan sekadar memakai model generik
tanpa penyesuaian.

Model ini menggantikan/mendukung tugas Supervisor Agent (yang sebelumnya
100% mengandalkan LLM) untuk klasifikasi kategori pertanyaan, dengan
keunggulan: JAUH lebih cepat dan tidak butuh panggilan ke LLM sama sekali
untuk tahap klasifikasi.

Cara jalankan:
    python train_classifier.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def main():
    print("=== FINE-TUNING SUPERVISOR CLASSIFIER ===\n")

    # 1. Load dataset
    df = pd.read_csv("training_data.csv")
    print(f"Total data: {len(df)} baris")
    print(df["label"].value_counts(), "\n")

    X = df["question"]
    y = df["label"]

    # 2. Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data training: {len(X_train)} | Data testing: {len(X_test)}\n")

    # 3. Bangun pipeline: TF-IDF -> Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),   # unigram + bigram, tangkap frasa seperti "shift malam"
            min_df=1,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # antisipasi jumlah data per kategori tidak sama rata
        )),
    ])

    # 4. Fine-tuning (training) model
    print("Melatih model...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluasi di data test
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== HASIL EVALUASI ===")
    print(f"Akurasi pada data test: {acc*100:.1f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix (baris=aktual, kolom=prediksi):")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    # 6. Simpan model hasil fine-tuning
    joblib.dump(pipeline, "supervisor_classifier.pkl")
    print("\nModel tersimpan sebagai 'supervisor_classifier.pkl'")

    # 7. Test cepat dengan beberapa contoh pertanyaan baru (belum pernah dilihat model)
    print("\n=== TEST DENGAN PERTANYAAN BARU ===")
    contoh_baru = [
        "Bahan baku apa yang perlu dipesan lagi minggu ini?",
        "Siapa yang piket shift malam sabtu ini?",
        "Kenapa ada selisih di laporan kas hari ini?",
        "Menu paling banyak dipesan apa bulan ini?",
        "Cuaca Yogyakarta hari ini gimana?",
    ]
    for q in contoh_baru:
        pred = pipeline.predict([q])[0]
        proba = pipeline.predict_proba([q]).max()
        print(f"  '{q}'\n    -> prediksi: {pred} (confidence: {proba:.2f})")


if __name__ == "__main__":
    main()
