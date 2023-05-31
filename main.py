import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
nltk.download('popular')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from itertools import chain
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from streamlit_option_menu import option_menu
st.set_page_config(page_title="Informatika Pariwisata", page_icon='')

with st.container():
    with st.sidebar:
        choose = option_menu("Menu", ["Home", "Implementation"],
                             icons=['house', 'basket-fill'],
                             menu_icon="app-indicator", default_index=0,
                             styles={
            "container": {"padding": "5!important", "background-color": "10A19D"},
            "icon": {"color": "#fb6f92", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#c6e2e9"},
            "nav-link-selected": {"background-color": "#a7bed3"},
        }
        )

    if choose == "Home":
        
        st.markdown('<h1 style = "text-align: center;"> <b>Informatika Pariwisata</b> </h1>', unsafe_allow_html = True)
        st.markdown('')
        st.markdown("# Judul Project ")
        st.info("Analisis Sentimen Review Terhadap Cita Rasa Warung Amboina Bangkalan menggunakan metode Random Forest dan Term Frequency-Inverse Document Frequency")
        st.markdown("# Dataset ")
        st.info("Data yang digunakan pada laporan ini adalah data ulasan cita rasa Warung Amboina. Data yang diambil dari Warung Amboina tersebut sebanyak lebih kurang 500 data dengan data yang diambil dalam waktu terdekat.")
        st.markdown("# Metode Usulan ")
        st.info("Random Forest")
        
    elif choose == "Implementation":
        st.title("Informatika Pariwisata")
        st.write("Amallia Tiara Putri - 200411100025")
        st.write("Diah Kamalia - 200411100061")
        desc, dataset, preprocessing, classification, implementation = st.tabs(["Deskripsi Data", "Dataset", "Preprocessing", "Classification", "Implementation"])
        with desc:
            st.write("# About Dataset")
            
            st.write("## Content")
            st.write("""
            1.  Name :
                > Tabel Name berisi nama pengguna yang memberikan komentar di Warung Amboina.
            2.  Text :
                > Tabel Text berisi komentar yang diberikan oleh pengguna.
            3.  Label :
                > Tabel Label berisi label Positif dan Negatif dari cita rasa makanan di Warung Amboina.
            4. Review URL :
                > Tabel Review URL berisi link yang mengarahkan ke halaman ulasan yang ada di Google Maps pada Warung Amboina.
            5. Reviewer URL :
                > Tabel Reviewer URL berisi link yang mengarahkan ke profil pengguna yang menambahkan ulasan yang ada di Google Maps pada Warung Amboina.
            6. Stars :
                > Tabel Stars berisi bintang yang diberikan oleh pengguna saat mengulas Warung Amboina di Google Maps.
            7. Publish at :
                > Tabel Publish at berisi waktu pengguna menambahkan ulasan di Warung Amboina pada Google Maps.
                    """)

            st.write("## Repository Github")
            st.write(" Click the link below to access the source code")
            repo = "https://github.com/diahkamalia/informatikapariwisata"
            st.markdown(f'[ Link Repository Github ]({repo})')
        with dataset:
            st.write("""# Load Dataset""")
            df = pd.read_csv("https://raw.githubusercontent.com/diahkamalia/DataMining1/main/amboina.csv")
            df
            sumdata = len(df)
            st.success(f"#### Total Data : {sumdata}")
            st.write("## Dataset Explanation")
            st.info("#### Classes :")
            st.write("""
            1. Positif
            2. Negatif
            """)

            col1,col2 = st.columns(2)
            with col1:
                st.info("#### Data Type")
                df.dtypes
            with col2:
                st.info("#### Empty Data")
                st.write(df.isnull().sum())
                #===================================
             
                
                
        with preprocessing : 
            st.write("""# Preprocessing""")
            st.write("""
            > Preprocessing data adalah proses menyiapkan data mentah dan membuatnya cocok untuk model pembelajaran mesin. Ini adalah langkah pertama dan penting saat membuat model pembelajaran mesin. Saat membuat proyek pembelajaran mesin, kami tidak selalu menemukan data yang bersih dan terformat.
            """)
            st.info("## Cleaned Data")
            data = pd.read_csv('https://raw.githubusercontent.com/diahkamalia/DataMining1/main/cleanedtext.csv', index_col=0)
            data
            Sumdata = len(data)
            st.success(f"#### Total Cleaned Data : {Sumdata}")
            
            st.info("## TF - IDF (Term Frequency Inverse Document Frequency)")
            from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
            countvectorizer = CountVectorizer()
            tfidfvectorizer = TfidfVectorizer()
            tfidf = TfidfVectorizer()
            countwm = CountVectorizer()
            documents_list = data.values.reshape(-1,).tolist()
            count_wm = countwm.fit_transform(data['text_tokens'].apply(lambda x: np.str_(x)))
            train_data = tfidf.fit_transform(data['text_tokens'].apply(lambda x: np.str_(x)))
            count_array = count_wm.toarray()
            tf_idf_array = train_data.toarray()
            words_set = tfidf.get_feature_names_out()
            count_set = countwm.get_feature_names_out()
            df_tf_idf = pd.DataFrame(tf_idf_array, columns = words_set)
            df_tf_idf
            
            st.info("## Dimension Reduction using PCA")
            # Impor library yang dibutuhkan
            from sklearn.decomposition import PCA
            # Inisialisasi objek PCA dengan 4 komponen
            pca = PCA(n_components=4)
            # Melakukan fit transform pada data
            X_pca = pca.fit_transform(df_tf_idf)
            X_pca.shape
            
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder() 
            data['Label']= label_encoder.fit_transform(data['Label'])

            y = data['Label'].values
            # y = data_vec.label.values
            X_train, X_test, y_train, y_test = train_test_split(X_pca, y ,test_size = 0.7, random_state =1)


        with classification : 
            st.write("""# Classification""")
            st.info("## Random Forest")
            st.write(""" > Random Forest adalah algoritma machine learning yang menggabungkan keluaran dari beberapa decision tree untuk mencapai satu hasil. Random Forest bekerja dengan membangun beberapa decision tree dan menggabungkannya demi mendapatkan prediksi yang lebih stabil dan akurat. Forest atau ‘Hutan’ yang dibangun oleh Random Forest adalah kumpulan decision tree di mana biasanya dilatih dengan metode bagging. 
            """)
            
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
            from sklearn.metrics import classification_report
            import warnings 
            warnings. filterwarnings('ignore')
            rf    = RandomForestClassifier(n_estimators=100, max_depth=10, criterion='entropy')
            rf.fit(X_train, y_train)
            y_pred  =  rf.predict(X_test)
            rf_accuracy  = round(100*accuracy_score(y_test, y_pred),1)
            rf_eval = classification_report(y_test, y_pred,output_dict = True)
            rf_eval_df = pd.DataFrame(rf_eval).transpose()
            st.header("Accuracy Result")
            st.info(f"Akurasi cita rasa dari Warung Amboina menggunakan metode Random Forest adalah : **{rf_accuracy}%** ")
#
        with implementation:
            st.write("# Implementation")
            st.write("### Add Review :")

            # Lowercase
            def text_lowercase(text):
                return text.lower()
            # Remove number
            def remove_numbers(text):
                result = re.sub(r'\d+', '', text)
                return result
            # Remove punctuation
            def remove_punctuation(text):
                translator = str.maketrans('', '', string.punctuation)
                return text.translate(translator)
            # Remove whitespace
            def remove_whitespace(text):
                return  " ".join(text.split())
            # Tokenization
            # Stop Words Removal
            stop_words = set(chain(stopwords.words('indonesian'),stopwords.words('english')))
            # Define a function to remove stop words from a sentence 
            def remove_stop_words(text): 
              # Use a list comprehension to remove stop words 
              filtered_words = [word for word in text if word not in stop_words] 
              # Join the filtered words back into a sentence 
              return ' '.join(filtered_words)
            # Stemming
            def stemming(text):
                factory = StemmerFactory()
                stemmer = factory.create_stemmer()
                text = stemmer.stem(text)
                return text
    
            # Input teks
            input_text = st.text_input('Comment')

            # Jika teks tersedia
            if input_text:
#                 # Preprocessing teks input
                l0w=text_lowercase(input_text)
                l0w1=remove_numbers(l0w)
                l0w2=remove_punctuation(l0w1)
                l0w3=remove_whitespace(l0w2)
                l0w4=word_tokenize(l0w3)
                l0w5=remove_stop_words(l0w4)
                l0w6=stemming(l0w5)
                #TF - IDF
                from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
                countvectorizer = CountVectorizer()
                tfidfvectorizer = TfidfVectorizer()
                l0w6=[l0w6]
                count_wm = countvectorizer.fit_transform(l0w6)
                tfidf_wm = tfidfvectorizer.fit_transform(l0w6)
                count_tokens = countvectorizer.get_feature_names_out()
                tfidf_tokens = tfidfvectorizer.get_feature_names_out()
                df_countvect = pd.DataFrame(data = count_wm.toarray(),columns = count_tokens)
                df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
                FIRST_IDX = 0
                use_model = rf
                        predictresult = use_model.predict(input_norm)[FIRST_IDX]
                        if predictresult == 0:
                            st.info(f"Negatif.")
                        elif predictresult == 1:
                            st.success(f"Positif")

                # Menampilkan hasil analisis sentimen
                st.subheader('Hasil Analisis Sentimen')
                st.write('Teks Asli:')
                st.write(input_text)
                st.write('Teks Setelah Preprocessing:')
                st.write(l0w)
                st.write(l0w1)
                st.write(l0w2)
                st.write(l0w3)
                st.write(l0w4)
                st.write(l0w5)
                st.write(l0w6)
#                 st.write(df_couNT)
                df_countvect

