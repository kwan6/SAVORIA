# SAVORIA 🍽️

SAVORIA merupakan proyek UAS yang mengimplementasikan konsep **Agentic AI** pada domain **Food and Beverage (F&B)**.

Sistem ini menggunakan pendekatan **multi-agent** dengan dukungan **Retrieval-Augmented Generation (RAG)** untuk membantu operasional restoran berdasarkan SOP dan data aktual yang tersimpan dalam file CSV.

Saat ini sistem masih menggunakan data dummy sebagai simulasi operasional restoran.

---

# Struktur Folder

```bash
savoria_project/
├── sop_docs/                    # Dokumen SOP per divisi (bahan RAG)
│   ├── sop_inventory.txt
│   ├── sop_order.txt
│   ├── sop_hr.txt
│   └── sop_finance.txt
│
├── scripts/
│   ├── build_vectorstore.py     # Embed SOP -> simpan ke ChromaDB
│   ├── test_retrieval.py        # Pengujian retrieval
│   ├── agents.py                # Implementasi agent per divisi
│   ├── data_tools.py            # Query data CSV
│   └── graph_savoria.py         # Multi-agent workflow (LangGraph)
│
├── chroma_db/                   # Database vector lokal (otomatis dibuat)
├── app.py                       # Dashboard Streamlit
├── requirements.txt
└── README.md
```

---

# Persiapan Dataset

Letakkan folder dataset (`savoria_dataset`) pada level yang sama dengan folder project.

Contoh struktur:

```bash
D:\SAVORIA\
├── savoria_dataset\
├── savoria_project\
│   ├── sop_docs\
│   ├── scripts\
│   └── ...
```

---

# Instalasi

Masuk ke folder project:

```powershell
cd savoria_project
```

Install seluruh dependency:

```powershell
python -m pip install -r requirements.txt
```

---

# Membangun Vector Store

Jalankan script berikut satu kali untuk membuat vector database dari dokumen SOP.

```powershell
cd scripts
python build_vectorstore.py
```

Folder `chroma_db/` akan otomatis dibuat dan berisi collection untuk masing-masing divisi:

- Inventory
- Order
- HR
- Finance

---

# Pengujian Retrieval

Untuk memastikan proses RAG berjalan dengan benar:

```powershell
python test_retrieval.py
```

Jika hasil retrieval sesuai dengan isi SOP, maka sistem RAG telah berhasil dibangun.

---

# Menjalankan Sistem Multi-Agent

Pastikan package berikut sudah terinstall:

```powershell
python -m pip install langchain-ollama
```

Pastikan juga **Ollama** sudah berjalan dan model `llama3.2` telah tersedia:

```powershell
ollama pull llama3.2
```

Selanjutnya jalankan sistem:

```powershell
cd scripts
python graph_savoria.py
```

Contoh pertanyaan:

- *Bahan apa yang paling kritis stoknya sekarang?*
- *Menu apa yang paling laris?*
- *Ada cabang yang kekurangan staf shift malam weekend?*
- *Ada selisih keuangan yang mencurigakan?*

Sistem akan mengklasifikasikan pertanyaan melalui **Supervisor Agent**, kemudian meneruskannya ke agent yang sesuai (**Inventory, Order, HR, atau Finance**).

Setiap agent memanfaatkan kombinasi:

- SOP perusahaan (RAG)
- Data aktual (CSV)
- LLM melalui Ollama

---

# Struktur Kode

| File | Deskripsi |
|-------|------------|
| `data_tools.py` | Fungsi query terhadap data CSV aktual |
| `agents.py` | Implementasi agent untuk tiap divisi |
| `graph_savoria.py` | Workflow multi-agent menggunakan LangGraph |
| `build_vectorstore.py` | Membuat vector database dari SOP |
| `test_retrieval.py` | Pengujian retrieval pada vector database |

---

# Menjalankan Dashboard

Dashboard dibangun menggunakan **Streamlit** dan telah terintegrasi dengan sistem multi-agent.

Jalankan:

```powershell
cd D:\SAVORIA\savoria_project
streamlit run app.py
```

Aplikasi akan berjalan pada:

```text
http://localhost:8501
```

Dashboard terdiri dari beberapa tab:

### Chat

- Interaksi dengan sistem multi-agent
- Mendukung filter per cabang
- Menampilkan sumber SOP yang digunakan dalam jawaban

### Monitoring Cabang

- Metrik omzet
- Status stok
- Menu terlaris
- Distribusi kanal pemesanan

### Evaluasi Model

Masih berupa placeholder dan akan dikembangkan pada tahap berikutnya.

---

# Pengembangan Selanjutnya

Beberapa fitur yang direncanakan:

- Evaluasi model (accuracy, effectiveness, efficiency, explainability, hallucination)
- Fine-tuning model klasifikasi pertanyaan
- Penggunaan dataset operasional nyata
- Peningkatan kolaborasi antar agent

---

# Catatan

Setiap pertanyaan pada tab Chat membutuhkan waktu beberapa detik untuk diproses karena sistem menggunakan LLM lokal melalui Ollama. Hal ini merupakan perilaku yang normal.
