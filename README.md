# 📰 AI News Classifier - Pengklasifikasi Berita Otomatis

Aplikasi web berbasis **Streamlit** yang menggunakan Machine Learning (**Decision Tree**) untuk mengklasifikasikan artikel berita ke dalam kategori tertentu secara otomatis. Proyek ini dikembangkan sebagai bagian dari tugas kelompok untuk mata kuliah Pengantar Pemrosesan Data dan Multimedia.

## 🚀 Fitur Utama

- **Input Berita Tunggal**: Masukkan judul dan deskripsi berita secara manual untuk mendapatkan prediksi kategori instan.
- **Unggah Batch (CSV/Excel)**: Unggah file dataset berita dalam format CSV atau Excel untuk diproses secara massal.
- **Visualisasi Hasil**: Menampilkan pratinjau data asli dan hasil prediksi dalam tabel yang interaktif.
- **Unduh Hasil**: Hasil klasifikasi batch dapat diunduh kembali dalam format CSV.
- **Model Machine Learning**: Menggunakan model *Decision Tree* yang sudah dilatih sebelumnya dengan ekstraksi fitur TF-IDF dan seleksi fitur Chi-Square.

## 🛠️ Teknologi yang Digunakan

- **Bahasa Pemrograman**: Python 3.x
- **Framework Web**: [Streamlit](https://streamlit.io/)
- **Data Science**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **NLP**: NLTK (untuk preprocessing teks)
- **Model Persistence**: Joblib

## 📁 Struktur Proyek

```text
📦 Tugas Kelompok-1
 ┣ 📂 modules
 ┃ ┣ 📜 model_loader.py    # Logika pemuatan model dan prediksi
 ┃ ┗ 📜 preprocessing.py   # Pipeline pembersihan teks (NLP)
 ┣ 📜 app.py               # File utama aplikasi Streamlit
 ┣ 📜 requirements.txt     # Daftar dependensi library
 ┣ 📜 tfidf_vectorizer.pkl # Model vectorizer yang sudah dilatih
 ┣ 📜 chi2_selector.pkl    # Model selector fitur yang sudah dilatih
 ┣ 📜 decision_tree_model.pkl # Model classifier Decision Tree
 ┣ 📜 news-classification-DT.ipynb # Notebook untuk training dan riset model
 ┗ 📜 README.md            # Dokumentasi proyek
```

## ⚙️ Instalasi & Cara Menjalankan

Ikuti langkah-langkah di bawah ini untuk menjalankan proyek ini di lingkungan lokal Anda:

### 1. Clone Repositori
```bash
git clone https://github.com/username/repository-name.git
cd repository-name
```

### 2. Buat Virtual Environment (Opsional tapi Disarankan)
```bash
python -m venv venv
# Aktifkan venv (Windows)
.\venv\Scripts\activate
# Aktifkan venv (Linux/Mac)
source venv/bin/activate
```

### 3. Instal Dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
```bash
streamlit run app.py
```

Setelah menjalankan perintah di atas, aplikasi akan terbuka secara otomatis di browser Anda (biasanya di `http://localhost:8501`).

## 📖 Cara Penggunaan

1. **Input Tunggal**:
   - Ketik judul berita di kolom yang tersedia.
   - Ketik atau tempel deskripsi berita.
   - Klik tombol **"Prediksi Kategori"**.

2. **Unggah Batch**:
   - Pindah ke tab **"Unggah Batch"**.
   - Unggah file CSV atau Excel Anda.
   - Pilih kolom mana yang merupakan judul dan mana yang merupakan deskripsi.
   - Klik **"Proses Klasifikasi Batch"**.
   - Setelah selesai, Anda dapat melihat pratinjau hasil dan mengunduhnya via tombol **"Unduh Hasil sebagai CSV"**.

