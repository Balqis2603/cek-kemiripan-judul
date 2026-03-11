import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Sistem Deteksi Kemiripan Judul Skripsi", layout="wide")

st.title("🛡️ Sistem Pengecekan Kemiripan Judul Skripsi")
st.write("Program Studi Teknik Komputer")

uploaded_file = st.file_uploader("Upload Dataset Judul Skripsi (CSV)", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    st.success("Dataset berhasil diupload")

    # PREVIEW DATASET
    with st.expander("Lihat Daftar Judul Skripsi dalam Dataset"):
        df_display = data.copy()
        df_display.index = range(1, len(df_display) + 1)
        st.dataframe(df_display, use_container_width=True)

    st.subheader("Masukkan Judul Skripsi Baru")

    judul_input = st.text_input("Judul Skripsi")

    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        cek = st.button("🔍 Cek Kemiripan")

    if cek:

        if judul_input.strip() != "":

            daftar_judul = data["judul"].astype(str).tolist()
            semua_judul = daftar_judul + [judul_input]

            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(semua_judul)

            similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
            skor = similarity.flatten() * 100

            hasil = pd.DataFrame({
                "Judul Skripsi": data["judul"],
                "Tahun": data["tahun"],
                "Prodi": data["prodi"],
                "Skor Kemiripan": skor
            })

            hasil = hasil.sort_values(
                by="Skor Kemiripan",
                ascending=False
            ).reset_index(drop=True)

            skor_tertinggi = hasil["Skor Kemiripan"].max()

            st.subheader("Hasil Analisis Kemiripan")

            # STATUS KEMIRIPAN
            if skor_tertinggi > 70:
                st.error(f"🚨 Kemiripan Tinggi Terdeteksi ({skor_tertinggi:.2f}%)")
            elif skor_tertinggi > 40:
                st.warning(f"⚠️ Kemiripan Sedang ({skor_tertinggi:.2f}%)")
            else:
                st.success(f"✅ Kemiripan Rendah ({skor_tertinggi:.2f}%)")

            # fungsi warna skor
            def warna_skor(val):
                if val > 70:
                    color = "red"
                elif val > 40:
                    color = "orange"
                else:
                    color = "green"
                return f"background-color:{color}; color:white; font-weight:bold"

            styled = hasil.style.applymap(
                warna_skor,
                subset=["Skor Kemiripan"]
            ).format({
                "Skor Kemiripan": "{:.2f}%"
            })

            st.dataframe(styled, use_container_width=True)

            # TOP 5 JUDUL PALING MIRIP
            st.subheader("Top 5 Judul Paling Mirip")

            top5 = hasil.head(5).copy()
            top5["Skor Kemiripan"] = top5["Skor Kemiripan"].round(2).astype(str) + "%"

            top5.index = range(1, len(top5)+1)

            st.table(top5)

        else:
            st.warning("Masukkan judul terlebih dahulu")
