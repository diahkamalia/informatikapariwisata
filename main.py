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
        choose = option_menu("Informatika Pariwisata", ["Home", "Implementation"],
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
        st.markdown("## Judul Project ")
        st.info("Analisis Sentimen Review Terhadap Cita Rasa Warung Amboina Bangkalan menggunakan metode Random Forest dan Term Frequency-Inverse Document Frequency")

    elif choose == "Project":
        st.title("Informatika Pariwisata")
        st.write("Amallia Tiara Putri- 200411100025")
        st.write("Diah Kamalia - 200411100061")
        desc, dataset = st.tabs(["Deskripsi Data", "Dataset"])
        with desc:
            st.write("## About Dataset")
            
            st.write("## Content")
            st.write("""
            1.  Name :
                > PH is an important parameter in evaluating the acid–base balance of water. It is also the indicator of acidic or alkaline condition of water status. WHO has recommended maximum permissible limit of pH from 6.5 to 8.5. The current investigation ranges were 6.52–6.83 which are in the range of WHO standards.
            2.  Text :
                > Hardness is mainly caused by calcium and magnesium salts. These salts are dissolved from geologic deposits through which water travels. The length of time water is in contact with hardness producing material helps determine how much hardness there is in raw water. Hardness was originally defined as the capacity of water to precipitate soap caused by Calcium and Magnesium.
            3.  Label :
                > Water has the ability to dissolve a wide range of inorganic and some organic minerals or salts such as potassium, calcium, sodium, bicarbonates, chlorides, magnesium, sulfates etc. These minerals produced un-wanted taste and diluted color in appearance of water. This is the important parameter for the use of water. The water with high TDS value indicates that water is highly mineralized. Desirable limit for TDS is 500 mg/l and maximum limit is 1000 mg/l which prescribed for drinking purpose.
            4. Review URL :
                > Chlorine and chloramine are the major disinfectants used in public water systems. Chloramines are most commonly formed when ammonia is added to chlorine to treat drinking water. Chlorine levels up to 4 milligrams per liter (mg/L or 4 parts per million (ppm)) are considered safe in drinking water.
            5. Reviewer URL :
                > Sulfates are naturally occurring substances that are found in minerals, soil, and rocks. They are present in ambient air, groundwater, plants, and food. The principal commercial use of sulfate is in the chemical industry. Sulfate concentration in seawater is about 2,700 milligrams per liter (mg/L). It ranges from 3 to 30 mg/L in most freshwater supplies, although much higher concentrations (1000 mg/L) are found in some geographic locations.
            6. Stars :
                > Pure water is not a good conductor of electric current rather’s a good insulator. Increase in ions concentration enhances the electrical conductivity of water. Generally, the amount of dissolved solids in water determines the electrical conductivity. Electrical conductivity (EC) actually measures the ionic process of a solution that enables it to transmit current. According to WHO standards, EC value should not exceeded 400 μS/cm.
            7. Publish at
                > Total Organic Carbon (TOC) in source waters comes from decaying natural organic matter (NOM) as well as synthetic sources. TOC is a measure of the total amount of carbon in organic compounds in pure water. According to US EPA < 2 mg/L as TOC in treated / drinking water, and < 4 mg/Lit in source water which is use for treatment.
                    """)

            st.write("## Repository Github")
            st.write(" Click the link below to access the source code")
            repo = "https://github.com/diahkamalia/waterquality"
            st.markdown(f'[ Link Repository Github ]({repo})')
        with dataset:
            st.write("""# Load Dataset""")
            df = pd.read_csv("https://raw.githubusercontent.com/diahkamalia/DataMining1/main/amboina.csv")
            df
#             sumdata = len(df)
#             st.success(f"#### Total Data : {sumdata}")
#             st.dataframe(df)
#             st.write("## Dataset Explanation")
#             st.info("#### Classes :")
#             st.write('0 : Positif')
#             st.write('1 : Negatif')
#             col1,col2 = st.columns(2)
#             with col1:
#                 st.info("#### Data Type")
#                 df.dtypes
#             with col2:
#                 st.info("#### Amount of Data")
#                 df.count
                #===================================
             
                
                
#         with preprocessing : 
#             st.write("""# Preprocessing""")
#             st.write("""
#             > Preprocessing data adalah proses menyiapkan data mentah dan membuatnya cocok untuk model pembelajaran mesin. Ini adalah langkah pertama dan penting saat membuat model pembelajaran mesin. Saat membuat proyek pembelajaran mesin, kami tidak selalu menemukan data yang bersih dan terformat.
#             """)
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
#         with classification : 
#             st.write("""# Classification""")
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
                
#         with implementation:
#             st.write("# Implementation")
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
