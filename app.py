import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Deteksi Kemiripan Judul Skripsi", layout="centered")

st.markdown(
    "<h2 style='text-align: center;'>SISTEM PENGECEKAN KEMIRIPAN JUDUL SKRIPSI</h2>",
    unsafe_allow_html=True
)

st.write("Sistem ini menggunakan metode TF-IDF dan Cosine Similarity untuk menghitung tingkat kemiripan judul skripsi.")

# Load dataset
data = pd.read_csv("data_judul_urut_tahun.csv")

st.markdown("### Judul Skripsi Baru")
judul_input = st.text_input("", placeholder="Masukkan judul skripsi yang ingin dicek...")

if st.button("🔍 Cek Kemiripan"):

    if judul_input.strip() != "":

        daftar_judul = data["judul"].tolist()
        semua_judul = daftar_judul + [judul_input]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(semua_judul)

        similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        skor = similarity.flatten() * 100

        hasil = pd.DataFrame({
            "Judul Skripsi Lama": data["judul"],
            "Tahun": data["tahun"],
            "Prodi": data["prodi"],
            "Skor Kemiripan (%)": skor.round(2)
        })

        hasil = hasil.sort_values(
            by=["Skor Kemiripan (%)", "Tahun"],
            ascending=[False, True]
        )

        hasil = hasil.reset_index(drop=True)

        st.markdown("### Hasil Kemiripan")

        # Highlight skor tertinggi
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: #ffdddd' if v else '' for v in is_max]

        styled = hasil.style.apply(highlight_max, subset=["Skor Kemiripan (%)"])

        st.dataframe(styled, use_container_width=True)

    else:
        st.warning("Masukkan judul terlebih dahulu.")
