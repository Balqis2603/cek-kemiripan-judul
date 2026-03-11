import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Deteksi Plagiarisme", layout="wide")

# --- CUSTOM CSS (Tampilan UI) ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        width: 100%;
        font-weight: bold;
        padding: 10px;
    }
    h1 { color: #2c3e50; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🛡️ Sistem Pengecekan Kemiripan Judul Skripsi")
st.markdown("<p style='text-align: center;'>Prodi Teknik Komputer - Universitas Sains Cut Nyak Dhien</p>", unsafe_allow_html=True)
st.divider()

# --- INPUT DATASET (FITUR UPLOAD FLEKSIBEL) ---
st.subheader("1. Persiapan Dataset")
uploaded_file = st.file_uploader("Upload Dataset Judul Skripsi (CSV)", type=["csv"])

if uploaded_file is not None:
    # Membaca data
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset '{uploaded_file.name}' berhasil diupload")
    
    # Menampilkan preview dataset dengan nomor urut mulai dari 1
    with st.expander("Lihat Daftar Judul dalam Dataset"):
        df_display = df.copy()
        df_display.index = range(1, len(df_display) + 1)
        st.dataframe(df_display, use_container_width=True)

    # --- INPUT JUDUL BARU ---
    st.subheader("2. Pengecekan Judul")
    judul_baru = st.text_area("Input Judul Skripsi Baru:", placeholder="Masukkan judul di sini...")

    if st.button("Cek Kemiripan"):
        if judul_baru.strip() != "":
            # Mengambil kolom judul dan pastikan tidak ada nilai kosong
            list_judul_lama = df['judul'].astype(str).tolist()
            
            # Gabungkan judul baru ke dalam list untuk proses TF-IDF
            semua_judul = list_judul_lama + [judul_baru]
            
            # --- PROSES TF-IDF ---
            # Menghitung bobot kata berdasarkan dataset yang diupload
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(semua_judul)
            
            # --- PROSES COSINE SIMILARITY ---
            # Membandingkan vektor judul baru dengan seluruh judul di dataset
            cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
            
            # Menyimpan hasil skor dalam persentase
            df['Skor Kemiripan'] = (cosine_sim[0] * 100).round(2)
            
            # Urutkan dari yang paling mirip
            hasil_df = df.sort_values(by='Skor Kemiripan', ascending=False).copy()
            
            # --- PERBAIKAN NOMOR URUT (Dimulai dari 1) ---
            # Mengatur ulang indeks agar di tabel dimulai dari angka 1
            hasil_df.index = range(1, len(hasil_df) + 1)
            
            # --- TAMPILAN HASIL ---
            st.subheader("Hasil Kemiripan")
            
            max_skor = hasil_df['Skor Kemiripan'].max()
            if max_skor > 70:
                st.error(f"🚨 Kemiripan Tinggi Terdeteksi! Skor tertinggi: {max_skor}%")
            elif max_skor > 40:
                st.warning(f"⚠️ Kemiripan Sedang. Skor tertinggi: {max_skor}%")
            else:
                st.success(f"✅ Kemiripan Rendah. Skor tertinggi: {max_skor}%")
            
            # Fungsi styling warna kolom skor
            def style_skor(val):
                color = 'red' if val > 70 else 'orange' if val > 40 else 'green'
                return f'background-color: {color}; color: white; font-weight: bold'

            # Menampilkan tabel hasil akhir
            st.dataframe(
                hasil_df[['judul', 'Tahun', 'Prodi', 'Skor Kemiripan']].style.applymap(style_skor, subset=['Skor Kemiripan']),
                use_container_width=True
            )
        else:
            st.info("Silakan masukkan judul baru terlebih dahulu.")
else:
    st.info("💡 Silakan upload file dataset (.csv) untuk memulai pengecekan.")

# --- FOOTER ---
st.divider()
st.caption("© 2026 Pendeteksian Plagiarisme Judul Skripsi - Teknik Komputer")
