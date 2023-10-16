import streamlit as st
import pandas as pd
from crawling import crawl_pta
from preprocessing import preprocess_text
from vsm import vsm_term_frequency
import base64
from lda import lda_topic_modelling
from clustering import kmeans_clustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import requests
from bs4 import BeautifulSoup

def get_all_prodi_links():
    # URL halaman utama PTA Trunojoyo
    main_url = "https://pta.trunojoyo.ac.id/"
    
    # Lakukan permintaan ke halaman utama
    response = requests.get(main_url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Temukan semua link prodi pada halaman utama
    prodi_links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and "/c_search/byprod" in href:
            prodi_links.append((link.text, href))
    
    return prodi_links
# Menu 1: Data Crawling
def data_crawling():
    st.write("Data Crawling")

    # Dapatkan semua link prodi
    prodi_links = get_all_prodi_links()
    
    # Daftar pilihan prodi
    prodi_options = [prodi[0] for prodi in prodi_links]
    
    # Biarkan pengguna memilih prodi
    selected_prodi = st.selectbox("Pilih Prodi:", prodi_options)

    # Dapatkan URL prodi berdasarkan pilihan pengguna
    url_link = [prodi[1] for prodi in prodi_links if prodi[0] == selected_prodi]

    if url_link:
        id_prodi = selected_prodi

        # Tombol "Crawl" untuk memulai proses crawling
        if st.button("Crawl"):
            crawl_pta(url_link[0], id_prodi)  # Gunakan URL pertama (indeks 0) jika ada banyak URL
            st.success(f"Data Prodi '{selected_prodi}' berhasil di-crawl.")

            # Simpan hasil crawling ke dalam session state
            st.session_state.df_crawled = pd.read_csv(f'PTA_{id_prodi}.csv')
            st.write(st.session_state.df_crawled)

            # Tambahkan tombol untuk mengunduh DataFrame ke dalam format CSV
            csv_file = st.session_state.df_crawled.to_csv(index=False)
            b64 = base64.b64encode(csv_file.encode()).decode()  # Encode sebagai base64
            href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download Data CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

    else:
        st.warning("Pilih prodi terlebih dahulu.")

# Menu 2: Data Preprocessing
def data_preprocessing():
    st.write("Data Preprocessing")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    # Periksa apakah data hasil crawling sudah ada dalam session state
    if 'df_crawled' in st.session_state:
        df = st.session_state.df_crawled
    else:
        df = None

    if uploaded_file is not None:
        # Jika ada file yang diunggah, gunakan data dari file
        df = pd.read_csv(uploaded_file)

    if df is not None:
        st.write("Data sebelum preprocessing:")
        st.write(df)
        # Remove missing values
        df.dropna(how='any', inplace=True)
        # Reset index after dropping rows
        df.reset_index(drop=True, inplace=True)
        # Preprocess the data
        df['Abstrak'] = df['Abstrak'].apply(preprocess_text)
        
        st.write("Data setelah preprocessing:")
        st.write(df)

        # Simpan hasil preprocessing ke dalam session state
        st.session_state.df_preprocessed = df

        # Konversi kolom Abstrak ke dalam format list teks
        texts = df['Abstrak'].tolist()

        # Panggil fungsi VSM dengan input teks
        tf_df = vsm_term_frequency(texts)

        # Simpan hasil Term Frequency ke dalam session state
        st.session_state.tf_df = tf_df

        return df



def term_frequency_analysis():
    st.write("Ekstraksi Fitur")

    # Check if preprocessing has been done previously and stored in a variable (e.g., df_preprocessed)
    if 'df_preprocessed' in st.session_state:
        df = st.session_state.df_preprocessed

        # Konversi kolom Abstrak ke dalam format list teks
        texts = df['Abstrak'].tolist()
        
        # Menggunakan CountVectorizer untuk menghitung Term Frequency
        vectorizer = CountVectorizer()
        tf_matrix = vectorizer.fit_transform(texts)

        # Membentuk DataFrame Term Frequency
        tf_df = pd.DataFrame(tf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

        # Menambahkan kolom "Abstrak" dengan teks abstrak asli
        tf_df.insert(0, "Judul", texts)

        st.write("Term Frequency DataFrame:")
        st.write(tf_df)
        st.write("Bentuk (Shape) dari Term Frequency DataFrame:", tf_df.shape)

    else:
        st.warning("Harap lakukan preprocessing terlebih dahulu atau pilih menu 'Data Preprocessing'.")

def topic_modelling_lda():
    try:
        st.write("LDA Topic Modelling")

        # Check if preprocessing has been done previously and stored in a variable (e.g., df_preprocessed)
        if 'df_preprocessed' in st.session_state:
            df = st.session_state.df_preprocessed

            if not df.empty:
                # Konversi kolom Abstrak ke dalam format list teks
                texts = df['Abstrak'].tolist()

                # Check if VSM has been computed previously and stored in a variable (e.g., tf_df)
                if 'tf_df' in st.session_state:
                    tf_df = st.session_state.tf_df
                else:
                    # Gunakan hasil VSM sebagai input untuk LDA
                    vectorizer = CountVectorizer()
                    tf_matrix = vectorizer.fit_transform(texts)

                    # Membentuk DataFrame Term Frequency
                    tf_df = pd.DataFrame(tf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
                    # Simpan hasil VSM ke dalam session state
                    st.session_state.tf_df = tf_df

                num_topics = 3 # Slider untuk memilih jumlah topik

                U, VT_tabel, lda = lda_topic_modelling(tf_df, num_topics)

                # Tampilkan hasil LDA

                # Menambahkan kolom "Abstrak" dengan teks abstrak asli
                tf_df.insert(0, "Judul", texts)
                st.write("Term Frequency DataFrame:")
                st.write(tf_df)
                st.write(tf_df.shape)

                U.insert(0, "Judul", texts)
                st.write("Matriks Dokumen-Topik (U):")
                st.dataframe(U)
                st.session_state.U = U
                st.write(U.shape)

                st.write("Matriks Topik-Kata (VT):")
                st.dataframe(VT_tabel)
                st.write(VT_tabel.shape)
            else:
                st.warning("Data preprocessing kosong.")
        else:
            st.warning("Harap lakukan preprocessing terlebih dahulu atau pilih menu 'Data Preprocessing'.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")




def clustering_kmeans():
    st.write("Clustering with K-Means")

    # Cek apakah U sudah ada di session state
    if 'U' in st.session_state:
        U = st.session_state.U

        # Melakukan clustering K-Means
        num_clusters = st.slider("Jumlah Cluster K-Means", 2, 10, 2)  # Slider untuk memilih jumlah cluster
        clusters = kmeans_clustering(U, num_clusters)

        # Menambahkan hasil clustering ke DataFrame U
        U['Cluster'] = clusters

        # Tampilkan hasil akhir dalam tabel
        st.write("Hasil Akhir:")
        st.dataframe(U)

    else:
        st.warning("Anda perlu melakukan analisis LDA terlebih dahulu di menu 'LDA Topic Modelling'.")



# Main page with 4 menus
def main():
    st.title("Topic Modelling LDA")
    menu = ["Data Crawling", "Data Preprocessing", "Ekstraksi Fitur", "LDA", "Clustering with K-Means"]
    choice = st.sidebar.selectbox("Pilih menu", menu)

    if choice == "Data Crawling":
        data_crawling()
    elif choice == "Data Preprocessing":
        data_preprocessing()
    elif choice == "Ekstraksi Fitur":
        term_frequency_analysis()
    elif choice == "LDA":
        topic_modelling_lda()
    elif choice == "Clustering with K-Means":
        clustering_kmeans() 

if __name__ == "__main__":
    main()
