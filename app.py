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

    judul_input = st.text_input("Judul Skripsi Baru")

    # Tombol di tengah
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        cek = st.button("Cek Kemiripan")

    if cek:
        if judul_input.strip() != "":

            daftar_judul = data["judul"].tolist()
            semua_judul = daftar_judul + [judul_input]

            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(semua_judul)

            similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
            skor = similarity.flatten() * 100  # masih dalam bentuk angka

            # Buat DataFrame dengan skor angka dulu
            hasil = pd.DataFrame({
                "Judul Skripsi Lama": data["judul"],
                "Tahun": data["tahun"],
                "Prodi": data["prodi"],
                "Skor Kemiripan": skor
            })

            # Urutkan dari skor tertinggi ke terendah
            hasil = hasil.sort_values(
                by="Skor Kemiripan",
                ascending=False
            ).reset_index(drop=True)

            # Tambahkan tanda %
            hasil["Skor Kemiripan"] = hasil["Skor Kemiripan"].round(2).astype(str) + "%"

            st.subheader("Hasil Kemiripan")

            # Rata tengah semua kolom
            styled = hasil.style.set_properties(**{'text-align': 'center'})
            styled = styled.set_table_styles(
                [dict(selector='th', props=[('text-align', 'center')])]
            )

            st.dataframe(styled, use_container_width=True, hide_index=True)

        else:
            st.warning("Masukkan judul terlebih dahulu.")

