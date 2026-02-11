import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Deteksi Kemiripan Judul Skripsi", layout="wide")

st.title("Sistem Pengecekan Kemiripan Judul Skripsi")

uploaded_file = st.file_uploader("Upload Dataset Judul Skripsi (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Dataset berhasil diupload")

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

