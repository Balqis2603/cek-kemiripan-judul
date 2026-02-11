import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Deteksi Kemiripan Judul Skripsi", layout="wide")

st.title("Sistem Pengecekan Kemiripan Judul Skripsi")
st.write("Sistem ini digunakan untuk mendeteksi tingkat kemiripan judul skripsi menggunakan metode TF-IDF dan Cosine Similarity.")

data = pd.read_csv("data_judul_urut_tahun.csv")

st.success("Dataset berhasil dimuat")

judul_input = st.text_input("Masukkan Judul Skripsi Baru")

if st.button("Cek Kemiripan"):
    if judul_input != "":
        daftar_judul = data["judul"].tolist()
        semua_judul = daftar_judul + [judul_input]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(semua_judul)

        similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        skor = similarity.flatten() * 100

        hasil = pd.DataFrame({
            "Judul Lama": data["judul"],
            "Tahun": data["tahun"],
            "Prodi": data["prodi"],
            "Skor Kemiripan (%)": skor.round(2)
        })

        hasil = hasil.sort_values(
            by=["Skor Kemiripan (%)", "Tahun"],
            ascending=[False, True]
        )

        st.subheader("Hasil Kemiripan")
        st.dataframe(hasil, use_container_width=True)
    else:
        st.warning("Masukkan judul terlebih dahulu")
