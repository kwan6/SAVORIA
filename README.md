# Savoria Project — Struktur Folder

```
savoria_project/
├── sop_docs/                    # Dokumen SOP per divisi (bahan RAG)
│   ├── sop_inventory.txt
│   ├── sop_order.txt
│   ├── sop_hr.txt
│   └── sop_finance.txt
├── scripts/
│   ├── build_vectorstore.py     # Embed SOP -> simpan ke ChromaDB
│   └── test_retrieval.py        # Test apakah retrieval sudah benar
├── chroma_db/                   # (akan otomatis terbuat) database vector lokal
└── requirements.txt
```

## Cara Pakai

1. **Taruh folder `savoria_dataset` (CSV) di level yang sama dengan folder ini**, jadi strukturnya:
   ```
   D:\SAVORIA\
   ├── savoria_dataset\
   ├── savoria_project\
   │   ├── sop_docs\
   │   ├── scripts\
   │   └── ...
   ```

2. **Install dependency** (kalau belum):
   ```powershell
   cd savoria_project
   python -m pip install -r requirements.txt
   ```

3. **Build vector store** (jalankan sekali untuk generate ChromaDB dari SOP):
   ```powershell
   cd scripts
   python build_vectorstore.py
   ```
   Ini akan membuat folder `chroma_db/` berisi 4 collection (inventory, order, hr, finance).

4. **Test retrieval** untuk memastikan RAG bekerja dengan benar:
   ```powershell
   python test_retrieval.py
   ```
   Kamu akan lihat hasil pencarian untuk beberapa pertanyaan contoh per divisi.
   Kalau hasil retrieve-nya relevan dengan isi SOP, berarti tahap RAG sudah beres.

## Langkah Selanjutnya
Setelah retrieval berhasil, lanjut ke tahap Agent:

5. **Install package tambahan untuk koneksi ke Ollama:**
   ```powershell
   python -m pip install langchain-ollama
   ```

6. **Pastikan Ollama sedang berjalan** dan model `llama3.2` sudah ter-pull
   (`ollama pull llama3.2`).

7. **Jalankan sistem multi-agent secara interaktif:**
   ```powershell
   cd scripts
   python graph_savoria.py
   ```
   Coba tanyakan hal-hal seperti:
   - "Bahan apa yang paling kritis stoknya sekarang?"
   - "Menu apa yang paling laris?"
   - "Ada cabang yang kekurangan staf shift malam weekend?"
   - "Ada selisih keuangan yang mencurigakan?"

   Sistem akan otomatis mengklasifikasikan pertanyaan ke divisi yang tepat
   (Inventory/Order/HR/Finance) lewat Supervisor Agent, lalu agent divisi
   terkait akan menjawab menggunakan kombinasi SOP (RAG) + data aktual (CSV).

## Struktur File Kode
- `data_tools.py` — fungsi-fungsi query data CSV aktual (dipakai tiap agent)
- `agents.py` — 4 fungsi agent (Inventory, Order, HR, Finance), masing-masing
  menggabungkan retrieval SOP + data CSV + LLM Ollama
- `graph_savoria.py` — LangGraph: Supervisor (klasifikasi) -> Agent Divisi -> Output

## Langkah Berikutnya (belum ada di paket ini)
- Skrip evaluasi (accuracy, effectiveness, efficiency, explainability, hallucination)
- Komponen fine-tuning kecil (misal klasifikasi kategori pertanyaan pakai scikit-learn)

## Menjalankan Dashboard

Dashboard dibuat dengan Streamlit dan sudah menggabungkan sistem multi-agent
(chat) dengan visualisasi monitoring per cabang.

```powershell
cd D:\SAVORIA\savoria_project
streamlit run app.py
```

Browser akan otomatis terbuka ke `http://localhost:8501`. Dashboard punya 3 tab:
- **Chat** — tanya jawab ke sistem multi-agent, bisa difilter per cabang, dan
  bisa lihat sumber SOP yang dipakai untuk menjawab
- **Monitoring Cabang** — metrik omzet, status stok, menu terlaris, dan
  distribusi kanal pemesanan per cabang
- **Evaluasi Model** — placeholder, akan diisi setelah skrip evaluasi dibuat

Catatan: setiap pertanyaan di tab Chat butuh waktu beberapa detik karena harus
memanggil LLM Ollama secara lokal — ini normal.


