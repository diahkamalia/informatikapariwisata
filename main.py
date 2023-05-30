import streamlit as st
import pandas as pd
import numpy as np
from sklearn.utils.validation import joblib
import joblib
from PIL import Image
import io

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
            # Text Cleaning
#             def cleaning(text):
#                 # HTML Tag Removal
#                 text = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});').sub('', str(text))

#                 # Case folding
#                 text = text.lower()

#                 # Trim text
#                 text = text.strip()

#                 # Remove punctuations, karakter spesial, and spasi ganda
#                 text = re.compile('<.*?>').sub('', text)
#                 text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
#                 text = re.sub('\s+', ' ', text)

#                 # Number removal
#                 text = re.sub(r'\[[0-9]*\]', ' ', text)
#                 text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
#                 text = re.sub(r'\d', ' ', text)
#                 text = re.sub(r'\s+', ' ', text)

#                 # Mengubah text 'nan' dengan whitespace agar nantinya dapat dihapus
#                 text = re.sub('nan', '', text)

#                 return text
#             st.info("## Text Cleaning")
#             df['text'] = df['text'].apply(lambda x: cleaning(x))
            
#             st.info("## Tokenization")
#             df['text_tokens'] = df['text'].apply(lambda x: word_tokenize(x))
#             df[["text", "text_tokens"]].head()
            
#             st.info("## Stop Words Removal")
#             stop_words = set(chain(stopwords.words('indonesian'), stopwords.words('english')))
#             df['text_tokens'] = df['text_tokens'].apply(lambda x: [w for w in x if not w in stop_words])
#             df[["text", "text_tokens"]].head(20)
            
#             st.info("## Stemming")
#             tqdm.pandas()
#             factory = StemmerFactory()
#             stemmer = factory.create_stemmer()
#             df['text_tokens'] = df['text_tokens'].progress_apply(lambda x: stemmer.stem(' '.join(x)).split(' '))
#             df[["text", "text_tokens"]].head(20)
            
            st.info("## Cleaned Data")
            data = pd.read_csv('https://raw.githubusercontent.com/diahkamalia/DataMining1/main/cleanedtext.csv', index_col=0)
            data
            
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
            # data['label'].value_counts()
#             st.write("### Formula")
#             st.latex(r'''
#             X = \frac{X_i - X_{min}}{X_{max} - X_{min}}
#             ''')
#             st.warning("### Data Before Normalization")
#             st.dataframe(df)
#             label = df["Potability"]
#             st.info("## Drop Column")
#             st.write(" > Dropping 'Potability' Table")
#             X = df.drop(columns=["Potability"])
#             st.dataframe(X)
#             st.info(" ## Normalisasi ")
#             st.write(""" > ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity """)
#             st.write("#### Data After Normalization ")
#             label=df["Potability"]
#             data_for_minmax_scaler=pd.DataFrame(df, columns = ['ph', 'Hardness', 'Solids', 'Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity'])
#             data_for_minmax_scaler.to_numpy()
#             scaler = MinMaxScaler()
#             hasil_minmax=scaler.fit_transform(data_for_minmax_scaler)
#             hasil_minmax = pd.DataFrame(hasil_minmax,columns = ['ph', 'Hardness', 'Solids', 'Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity'])
#             st.dataframe(hasil_minmax)
#             X = hasil_minmax
#             X=X.join(label) 
#             st.success("## New Dataframe")
#             st.dataframe(X)
#             st.write("## Counting Data")
#             st.write("- Drop Potability column from dataframe")
#             st.write("- Split Data Training and Data Test")
#             st.warning(""" ### Spliting Data""")
#             st.write("""        
#             - Data Training & X_train 
#             - Data Test & X_test 
#             - Data Training (Class) & y_train
#             - Data Test (Class) & y_test
#             """)
            
#             X   = hasil_minmax.iloc[:,:9]
#             y  = df.iloc[:,-1]
#             # membagi data menjadi set train dan test (70:30)
#             X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)
#             st.success("Showing X")
#             st.write(X)
            
#             st.success("Showing Y")
#             st.write(y)

        with classification : 
            st.write("""# Classification""")
            st.info("## Random Forest")
            
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
            st.info(f"Water Potability accuracy of K-Nearest Neighbour model is : **{rf_accuracy}%** ")
#             knn,gaussian,decision= st.tabs(["K-Nearest Neighbor", "Gaussian Naive Bayes", "Decision Tree"])
#             with knn:
#                 st.write("## K-Nearest Neighbor")
#                 st.write( '<div style ="text-align: justify;" > Algortima K-Nearest Neighbor (KNN) adalah merupakan sebuah metode untuk melakukan klasifikasi terhadap obyek baru berdasarkan (K) tetangga terdekatnya. Metode pencarian jarak yang digunakan adalah Euclidean Distance yaitu perhitungan jarak terdekat. </div>', unsafe_allow_html = True)
#                 st.write("### Formula")
#                 st.latex(r'''
#                 d(p,q) = d(p,q) = \sqrt{(q_1 - p_1)^2 + (q_2 - p_2)^2 + . . . + (q_n - p_n)^2}
#                                 = \sqrt{\displaystyle\sum_{i=1}^n (q_i - p_i)^2}
#                 ''')
#                 st.write("### Step of the K-Nearest Neighbor Algorithm")
#                 st.write("""
#                 1. Menentukan parameter K (jumlah tetangga paling dekat)
#                 2. Menghitung kuadrat jarak euclidian (euclidean distance) masing-masing obyek terhadap data sampel yang diberikan
#                 3. Mengurutkan objek-objek tersebut ke dalam kelompok yang mempunyai jarak euclidean terkecil
#                 4. Mengumpulkan kategori Y (klasifikasi nearest neighbor)
#                 5. Dengan menggunakan kategori mayoritas,maka dapat diprediksikan nilai query instance yang telah dihitung
#                 """)
#                 # Inisialisasi K-NN
#                 k_range = range(1,26)
#                 scores = {}
#                 scores_list = []
#                 for k in k_range:
#                     knn = KNeighborsClassifier(n_neighbors=k)
#                     knn.fit(X_train, y_train)
#                     y_pred_knn = knn.predict(X_test)
#                     knn_accuracy = round(100 * accuracy_score(y_test, y_pred_knn), 2)
#                     knn_eval = classification_report(y_test, y_pred_knn,output_dict = True)
#                     knn_eval_df = pd.DataFrame(knn_eval).transpose()
#                 st.header("Accuracy Result")
#                 st.info(f"Water Potability accuracy of K-Nearest Neighbour model is : **{knn_accuracy}%** ")
#                 st.write("> K = 1 - 25")
            
#             with gaussian:
#                 st.write("## Gaussian Naive Bayes")
#                 st.write(' <div style ="text-align: justify;"> Naive Bayes adalah algoritma machine learning untuk masalah klasifikasi. Ini didasarkan pada teorema probabilitas Bayes. Hal ini digunakan untuk klasifikasi teks yang melibatkan set data pelatihan dimensi tinggi. Beberapa contohnya adalah penyaringan spam, analisis sentimental, dan klasifikasi artikel berita.</div>', unsafe_allow_html = True)
#                 st.write("### Formula")
#                 st.latex(r'''
#                 P(C_k | x) = \frac{P(C_k) P(x|C_k)}{P(x)}
#                 ''')
#                 # Inisialisasi Gaussian
#                 gaussian    = GaussianNB()
#                 gaussian.fit(X_train,y_train)
#                 y_pred_gaussian   =  gaussian.predict(X_test)
#                 gauss_accuracy  = round(100*accuracy_score(y_test, y_pred_gaussian),2)
#                 gaussian_eval = classification_report(y_test, y_pred_gaussian,output_dict = True)
#                 gaussian_eval_df = pd.DataFrame(gaussian_eval).transpose()
#                 st.header("Accuracy Result")
#                 st.info(f"Water Potability accuracy of Gaussian Naive Bayes model is : **{gauss_accuracy}%** ")
                
#             with decision:
#                 st.write("## Decision Tree")
#                 st.write('<div style ="text-align: justify;"> Decision tree merupakan alat pendukung keputusan dengan struktur seperti pohon yang memodelkan kemungkinan hasil, biaya sumber daya, utilitas, dan kemungkinan konsekuensi. Konsepnya adalah dengan cara menyajikan algoritma dengan pernyataan bersyarat yang meliputi cabang untuk mewakili langkah-langkah pengambilan keputusan yang dapat mengarah pada hasil yang menguntungkan.</div>' , unsafe_allow_html = True)
#                 st.write("### Define Decision Tree Roots")
#                 st.write("""
#                 > Akar akan diambil dari atribut yang terpilih, dengan cara menghitung nilai gain dari masing – masing atribut. Nilai gain yang paling tinggi akan menjadi akar pertama. Sebelum menghitung niali gain dari atribut, harus menghitung nilai entropy terlebih dahulu
#                 """)
#                 st.write("### Formula Entropy")
#                 st.latex(r'''
#                 Entropy\left(\LARGE{D_1}\right) = - \displaystyle\sum_{i=1}^m p_i log_2 p_i
#                 ''')
#                 st.write("### Formula Gain (D1)")
#                 st.latex(r'''
#                 Gain(E_{new}) = E_{initial} - E_{new}
#                 ''')
#                 # Inisialisasi Decision Tree
#                 decission3  = DecisionTreeClassifier(criterion="gini")
#                 decission3.fit(X_train,y_train)
#                 y_pred_decission3 = decission3.predict(X_test)
#                 d3_accuracy = round(100*accuracy_score(y_test, y_pred_decission3),2)
#                 d3_eval = classification_report(y_test, y_pred_decission3,output_dict = True)
#                 d3_eval_df = pd.DataFrame(d3_eval).transpose()
#                 st.header("Accuracy Result")
#                 st.info(f"Water Potability accuracy of Decision Tree model is : **{d3_accuracy}%** ")
                
        with implementation:
            st.write("# Implementation")
#             st.write("### Input Data :")
#             ph = st.number_input("pH",min_value=0.0000, max_value=14.0000)
#             Hardness = st.number_input("Hardness",min_value=47.4320, max_value=323.1240)
#             Solids = st.number_input("Solids",min_value=320.9426, max_value=61227.1960)
#             Chloramines = st.number_input("Chloramines",min_value=0.3520, max_value=13.1270)
#             Sulfate = st.number_input("Sulfate",min_value=0.0000, max_value=481.0306)
#             Conductivity = st.number_input("Conductivity",min_value=181.4838, max_value=753.3426)
#             Organic_carbon = st.number_input("Organic Carbon",min_value=2.2000, max_value=28.3000)
#             Trihalomethanes = st.number_input("Trihalomethanes",min_value=0.0000, max_value=124.0000)
#             Turbidity = st.number_input("Turbidity",min_value=1.4500, max_value=6.7390)
#             result = st.button("Submit")
#             best,each = st.tabs(["Best Modelling", "Every Modelling"])
#             with best:
#                 st.write("# Classification Result")
#                 if knn_accuracy > gauss_accuracy and knn_accuracy > d3_accuracy:
#                     use_model = knn
#                     model = "K-Nearest Neighbor"
#                 elif gauss_accuracy > knn_accuracy and gauss_accuracy > d3_accuracy:
#                     use_model = gaussian
#                     model = " Gaussian Naive Bayes"
#                 else:
#                     use_model = decission3
#                     model = "Decission Tree"
#                 input = [[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]]
#                 input_norm = scaler.transform(input)
#                 FIRST_IDX = 0
#                 if result:
#                     use_model = knn
#                     predictresult = use_model.predict(input_norm)[FIRST_IDX]
#                     if predictresult == 0:
#                         st.info(f"I'm Sorry, the water you tested is **{predictresult}** which means **Not Potable**  based on {model} model.")
#                     elif predictresult == 1:
#                         st.success(f"Good news, the water you tested is {predictresult} which means **Potable** B based on {model} model.")
#             with each:
#                 kaen,naif,pohh= st.tabs(["K-Nearest Neighbour", "Naive Bayes Gaussian", "Decision Tree"])
#                 with kaen:
#                     input = [[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]]
#                     input_norm = scaler.transform(input)
#                     FIRST_IDX = 0
#                     if result:
#                         use_model = knn
#                         predictresult = use_model.predict(input_norm)[FIRST_IDX]
#                         if predictresult == 0:
#                             st.info(f"I'm Sorry, the water you tested is **{predictresult}** which means **Not Potable**  based on K-Nearest Neighbor model.")
#                         elif predictresult == 1:
#                             st.success(f"Good news, the water you tested is {predictresult} which means **Potable** based on K-Nearest Neighbor model.")
#                 with naif:
#                     input = [[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]]
#                     input_norm = scaler.transform(input)
#                     FIRST_IDX = 0
#                     if result:
#                         use_model = gaussian
#                         predictresult = use_model.predict(input_norm)[FIRST_IDX]
#                         if predictresult == 0:
#                             st.info(f"I'm Sorry, the water you tested is **{predictresult}** which means **Not Potable**  based on Gaussian Naive Bayes model.")
#                         elif predictresult == 1:
#                             st.success(f"Good news, the water you tested is {predictresult} which means **Potable** based on Gaussian Naive Bayes model.")
#                 with pohh:
#                     input = [[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]]
#                     input_norm = scaler.transform(input)
#                     FIRST_IDX = 0
#                     if result:
#                         use_model = decission3
#                         predictresult = use_model.predict(input_norm)[FIRST_IDX]
#                         if predictresult == 0:
#                             st.info(f"I'm Sorry, the water you tested is **{predictresult}** which means **Not Potable**  based on Decision Tree model.")
#                         elif predictresult == 1:
#                             st.success(f"Good news, the water you tested is {predictresult} which means **Potable** based on Decision Tree model.")
