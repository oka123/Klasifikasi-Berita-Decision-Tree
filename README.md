# 📰 AI News Classifier - Pengklasifikasi Berita Otomatis

Aplikasi web berbasis **Streamlit** yang menggunakan Machine Learning (**Decision Tree**) untuk mengklasifikasikan artikel berita ke dalam kategori tertentu secara otomatis. Proyek ini dikembangkan sebagai bagian dari tugas kelompok pertama Kelas A Kelompok 4 pada mata kuliah Pengantar Pemrosesan Data dan Multimedia.

### 🔗 Link Terkait

- **Live Demo**: [klasifikasi-berita-dt.streamlit.app](https://klasifikasi-berita-dt.streamlit.app/)

## 🚀 Fitur Utama

- **Input Berita Tunggal**: Masukkan judul dan deskripsi berita secara manual untuk mendapatkan prediksi kategori instan.
- **Input Banyak Berita (CSV/Excel)**: Unggah file dataset berita dalam format CSV atau Excel untuk diproses secara massal.
- **Visualisasi Hasil**: Menampilkan pratinjau data asli dan hasil prediksi dalam tabel yang interaktif.
- **Unduh Hasil**: Hasil klasifikasi batch dapat diunduh kembali dalam format CSV.
- **Model Machine Learning**: Menggunakan model _Decision Tree_ yang sudah dilatih sebelumnya dengan ekstraksi fitur TF-IDF dan seleksi fitur Chi-Square.

## 🛠️ Teknologi yang Digunakan

- **Bahasa Pemrograman**: Python 3.x
- **Framework Web**: [Streamlit](https://streamlit.io/)
- **Data Science**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **NLP**: NLTK (untuk preprocessing teks)
- **Model Persistence**: Joblib

## 📁 Struktur Proyek

```text
 ┣ 📂 Dataset_AG_News
 ┃ ┣ 📜 test.csv              # Dataset uji
 ┃ ┗ 📜 train.csv             # Dataset latih
 ┣ 📂 modules
 ┃ ┣ 📜 __init__.py
 ┃ ┣ 📜 model_loader.py       # Logika pemuatan model dan prediksi
 ┃ ┗ 📜 preprocessing.py      # Pipeline pembersihan teks (NLP)
 ┣ 📜 Notebook.ipynb          # Jupyter Notebook untuk analisis data & training model
 ┣ 📜 app.py                  # File utama aplikasi Streamlit
 ┣ 📜 requirements.txt        # Daftar dependensi library
 ┣ 📜 tfidf_vectorizer.pkl    # Model vectorizer yang sudah dilatih
 ┣ 📜 ig_selector.pkl         # Model selector fitur yang sudah dilatih
 ┣ 📜 decision_tree_model.pkl # Model classifier Decision Tree
 ┗ 📜 README.md               # Dokumentasi proyek
```

## ⚙️ Instalasi & Cara Menjalankan

Ikuti langkah-langkah di bawah ini untuk menjalankan proyek ini di lingkungan lokal Anda:

### 1. Clone Repositori

```bash
git clone https://github.com/oka123/Klasifikasi-Berita-Decision-Tree.git
cd Klasifikasi-Berita-Decision-Tree
```

### 2. Instal Dependensi

```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi

```bash
streamlit run app.py
```

Setelah menjalankan perintah di atas, aplikasi akan terbuka secara otomatis di browser Anda (biasanya di `http://localhost:8501`).

## 📖 Cara Penggunaan

1. **Input Tunggal**:
   - Ketik judul berita di kolom yang tersedia.
   - Ketik atau tempel deskripsi berita.
   - Klik tombol **"Prediksi Kategori"**.

2. **Input Banyak Berita**:
   - Pindah ke tab **"Input Banyak Berita"**.
   - Unggah file CSV atau Excel Anda.
   - Pilih kolom mana yang merupakan judul dan mana yang merupakan deskripsi.
   - Klik **"Proses Klasifikasi Batch"**.
   - Setelah selesai, Anda dapat melihat pratinjau hasil dan mengunduhnya via tombol **"Unduh Hasil sebagai CSV"**.
