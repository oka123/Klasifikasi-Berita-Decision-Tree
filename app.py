import streamlit as st
import pandas as pd
from modules.model_loader import load_models, predict_news

# =================================================================
# PAGE CONFIGURATION
# =================================================================
st.set_page_config(
    page_title="AI Pengklasifikasi Berita",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================================================================
# RESOURCE INITIALIZATION
# =================================================================
vectorizer, chi2_selector, dt_model = load_models()

# =================================================================
# USER INTERFACE
# =================================================================
def main():
    # Sidebar
    with st.sidebar:
        st.title("📰 AI Pengklasifikasi Berita")
        st.markdown(
            """
            Selamat datang di Aplikasi Pengklasifikasi Berita! 
            
            Aplikasi ini menggunakan model Machine Learning (Decision Tree) untuk mengklasifikasikan berita ke dalam tiga kategori:
            - **Politik** 🏛️
            - **Olahraga** ⚽
            - **Teknologi** 💻
            
            **Cara penggunaan:**
            1. Gunakan tab **Input Berita Tunggal** untuk mengklasifikasikan satu artikel pada satu waktu.
            2. Gunakan tab **Unggah Batch** untuk mengklasifikasikan banyak artikel dari file CSV atau Excel.
            """
        )
        st.divider()
        st.caption("Ditenagai oleh Streamlit & Scikit-Learn")

    # Main Area
    st.title("Klasifikasi Berita Otomatis")
    st.markdown("Klasifikasikan artikel berita ke dalam kategorinya masing-masing dengan mudah menggunakan Kecerdasan Buatan.")
    st.divider()
    
    # Check if models are loaded before proceeding
    if not dt_model:
        st.warning("⚠️ Aplikasi saat ini tidak tersedia karena file model tidak ditemukan. Silakan hubungi administrator.")
        return

    # Tabs for different inputs
    tab1, tab2 = st.tabs(["✍️ Input Berita Tunggal", "📁 Unggah Batch (CSV/Excel)"])
    
    # --- TAB 1: SINGLE NEWS INPUT ---
    with tab1:
        st.subheader("Klasifikasikan satu artikel berita")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            news_title = st.text_input("Judul Berita", placeholder="Masukkan judul berita di sini...")
            news_desc = st.text_area("Deskripsi Berita", placeholder="Masukkan deskripsi berita atau teks lengkap di sini...", height=200)
            
            if st.button("Prediksi Kategori", type="primary"):
                if not news_title.strip() and not news_desc.strip():
                    st.warning("Harap masukkan judul atau deskripsi sebelum memprediksi.")
                else:
                    # Combine title and description
                    full_text = f"{news_title} {news_desc}"
                    
                    with st.spinner("Menganalisis teks..."):
                        prediction = predict_news(full_text, vectorizer, chi2_selector, dt_model)
                    
                    st.success("Klasifikasi Selesai!")
                    
                    # Display result creatively
                    st.markdown("### Hasil Prediksi:")
                    if str(prediction).lower() == 'politik' or str(prediction).lower() == 'politics':
                        st.info(f"🏛️ **{str(prediction).upper()}**")
                    elif str(prediction).lower() == 'olahraga' or str(prediction).lower() == 'sports':
                        st.success(f"⚽ **{str(prediction).upper()}**")
                    elif str(prediction).lower() == 'teknologi' or str(prediction).lower() == 'technology':
                        st.warning(f"💻 **{str(prediction).upper()}**")
                    else:
                        st.info(f"**{str(prediction).upper()}**")
                        
    # --- TAB 2: BATCH UPLOAD ---
    with tab2:
        st.subheader("Klasifikasikan banyak artikel berita dari sebuah file")
        st.markdown("Unggah file CSV atau Excel yang berisi data berita Anda.")
        
        uploaded_file = st.file_uploader("Pilih file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                # Read file based on extension
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.write("### Pratinjau Data")
                st.dataframe(df.head(), width='stretch')
                
                # Column selection
                st.markdown("#### Pilih kolom yang relevan")
                col_names = [str(c) for c in df.columns]
                
                # Try to guess default columns
                col_lower = [c.lower() for c in col_names]
                default_title_idx = col_lower.index('title') if 'title' in col_lower else 0
                default_desc_idx = col_lower.index('description') if 'description' in col_lower else (1 if len(col_names) > 1 else 0)
                
                c1, c2 = st.columns(2)
                with c1:
                    title_col = st.selectbox("Pilih kolom judul/title", ["None"] + col_names, index=default_title_idx + 1 if default_title_idx is not None else 0)
                with c2:
                    desc_col = st.selectbox("Pilih kolom deskripsi/description", col_names, index=default_desc_idx)
                
                if st.button("Proses Klasifikasi Batch", type="primary"):
                    with st.spinner("Memproses dokumen... Ini mungkin memakan waktu tergantung pada ukuran file."):
                        predictions = []
                        
                        # Process each row
                        for index, row in df.iterrows():
                            # Combine title and description safely
                            title_val = str(row[title_col]) if title_col != "None" and str(row[title_col]) != "nan" else ""
                            desc_val = str(row[desc_col]) if str(row[desc_col]) != "nan" else ""
                            
                            full_text = f"{title_val} {desc_val}".strip()
                            pred = predict_news(full_text, vectorizer, chi2_selector, dt_model)
                            predictions.append(pred)
                            
                        # Add predictions to dataframe
                        df['Predicted_Category'] = predictions
                        
                    st.success("Klasifikasi batch berhasil diselesaikan!")
                    
                    st.write("### Pratinjau Hasil")
                    st.dataframe(df.head(10), width='stretch')
                    
                    # Provide download button
                    # Convert df to csv string
                    csv = df.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="Unduh Hasil sebagai CSV",
                        data=csv,
                        file_name='classified_news_results.csv',
                        mime='text/csv',
                    )
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}")

if __name__ == "__main__":
    main()