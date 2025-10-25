INSTRUKSI MENJALANKAN APLIKASI
Sentiment Analyzer ChatBot Review (NLP) — Petunjuk Lengkap untuk Dosen

Catatan ringkas:
- Project ini dibuat dengan Python (3.9+). 
- Lokasi file utama:
  - repo root/
    - data/reviews.csv    ← dataset contoh
    - models/sentiment_model.joblib ← model TF-IDF+NaiveBayes (dibuat oleh train.py)
    - src/
      - preprocess.py
      - train.py
      - app.py
      - bert_train_optional.py (opsional IndoBERT)
    - templates/index.html

LANGKAH-LANGKAH (Windows dan macOS/Linux)

1) Clone repo (atau download ZIP) dan buka folder project:
   - Contoh:
     git clone https://github.com/PutYourBae/sentimen-analyzer.git
     cd <repo>

2) Buat Virtual Environment & aktifkan
   - Windows:
     python -m venv venv
     venv\Scripts\activate
   - macOS / Linux:
     python3 -m venv venv
     source venv/bin/activate

3) Install dependensi
   - Jika ada file requirements.txt:
     pip install -r requirements.txt
   - Jika tidak, jalankan:
     pip install numpy pandas scikit-learn nltk flask joblib sastrawi emoji matplotlib

   - Opsional (kalau mau IndoBERT / transformer):
     pip install datasets transformers torch
     (Catatan: instalasi torch berbeda-beda untuk GPU/CPU; lihat https://pytorch.org bila butuh versi GPU)

4) Download NLTK resources (jalankan satu kali)
   - Jalankan di terminal:
     python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

5) Cek dataset
   - Pastikan file `data/reviews.csv` ada dan memiliki dua kolom: `text,label`
   - Contoh baris:
     "chatbot ini sangat membantu",positive

6) Latih model TF-IDF + Naive Bayes (membuat models/sentiment_model.joblib)
   - Pindah ke folder src dan jalankan:
     cd src
     python train.py
   - Output: metrik (accuracy, classification_report) dan file `../models/sentiment_model.joblib` dibuat.

   - Catatan: Jika training error karena file tidak ditemukan, pastikan kamu menjalankan dari folder `src` dan path relatif data/reviews.csv benar.

7) Jalankan aplikasi web (Flask)
   - Di folder src yang sama:
     python app.py
   - Terminal akan menampilkan:
     "🚀 Server berjalan di http://127.0.0.1:5000"
   - Buka browser pada alamat tersebut.

8) Cara menguji di web UI
   - Ketik contoh:
     - "Chatbot ini sangat membantu dan cepat" → diharapkan: positive
     - "Chatbot tidak berguna sama sekali" → diharapkan: negative
   - Hasil tampil di halaman: processed text, prediksi, probabilitas, flag sarkasme; dashboard juga menampilkan grafik distribusi.

9) Endpoint API (alternatif)
   - POST /predict (form field `text`)
     - Contoh menggunakan curl:
       curl -X POST -d "text=chatbot ini bagus" http://127.0.0.1:5000/predict

   - (Opsional) Jika IndoBERT sudah dilatih dan diintegrasikan:
     - POST /predict_bert (form field `text`)

10) Jika mau menggunakan IndoBERT (opsional)
    - Jalankan training IndoBERT (butuh resource):
      cd src
      python bert_train_optional.py
    - Model disimpan ke src/bert_out/
    - Integrasikan ke app.py (lihat fungsi predict_bert yang sudah disiapkan)
    - Perhatikan: membutuhkan GPU untuk training cepat; di CPU butuh penyesuaian batch size dan epoch.

TROUBLESHOOTING UMUM

A. Error NLTK LookupError (contoh: punkt not found)
   - Jalankan:
     python -m nltk.downloader punkt stopwords punkt_tab

B. Prediksi "tidak berguna" jadi positif
   - Penyebab: kata negasi 'tidak' terhapus atau tidak di-handle; solusinya:
     - Jangan hapus kata 'tidak' dari stopwords.
     - Tambahkan negation handling di preprocess.py:
       def handle_negation(text):
           text = re.sub(r'\btidak\s+(\w+)', r'tidak_\1', text)
           text = re.sub(r'\bnggak\s+(\w+)', r'tidak_\1', text)
           return text
     - Panggil handle_negation() sebelum tokenisasi/stemming, lalu latih ulang model.

C. VSCode / linter menandai `datasets` missing
   - Pastikan virtualenv aktif dan interpreter VSCode mengarah ke venv. Jika tidak butuh IndoBERT, abaikan file bert_train_optional.py.

D. Jika server tidak mau jalan (port in use)
   - Tutup aplikasi lain yang memakai port 5000 atau jalankan flask pada port lain:
     python app.py  # atau edit app.run(port=5001)

E. Jika model tidak ada (models/sentiment_model.joblib)
   - Pastikan train.py dijalankan tanpa error; jika gagal, periksa output error di terminal.

CATATAN UNTUK DOSEN (tips cepat)
- Untuk demo cepat tanpa training: sediakan file models/sentiment_model.joblib di folder `models/` (sudah dilatih) agar dosen tidak perlu menjalankan train.py.
- Untuk uji kasus negasi/sarkasme: siapkan daftar contoh kalimat (5-10 kalimat) dan tampilkan perbandingan hasil NaiveBayes vs IndoBERT (jika IndoBERT tersedia).
- Jika ada kendala instalasi, mintakan akses terminal/TeamViewer atau minta file ZIP repo yang langsung saya siapkan (opsional).

CONTOH TEST (bisa copy paste ke UI atau curl)
1) Chatbot ini sangat membantu dan cepat
   - Expected: positive

2) Chatbot tidak berguna sama sekali
   - Expected: negative (jika negasi ditangani dengan benar atau IndoBERT digunakan)

3) Fiturnya sering error, sangat mengecewakan
   - Expected: negative

4) Lumayan tapi perlu banyak perbaikan
   - Expected: neutral

Jika butuh bantuan lebih lanjut: beri tahu saya contoh error beserta output terminal (copy–paste), saya bantu per baris.

Terima kasih — semoga lancar.
