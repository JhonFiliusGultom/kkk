import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


st.title("KECERDASAN KOMPUTASIONAL C")
st.write("##### Kelompok ")
st.write("##### 1. Nuriya amelia Febrianti 210411100019 ")


# Tampilan Aturan Navbarnya 
masukkandata, preprocessing, modeling = st.tabs(["Upload Data", "Preprocessing", "Modeling"])

df = pd.read_csv('https://raw.githubusercontent.com/NuriyahFebrianti/data/main/DATASET.csv')

# Masukkan data
with masukkandata:
    uploaded_file = st.file_uploader("Upload file CSV")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.header("Dataset")
        st.dataframe(df)

# Preprocessing
with preprocessing:
    st.subheader("Normalisasi Data")
    df = df.drop(columns=["Nomor"])
    X = df.drop(columns=["Penyakit"])
    y = df["Penyakit"].values
    df_min = X.min()
    df_max = X.max()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    feature_names = X.columns.copy()
    scaled_features = pd.DataFrame(scaled, columns=feature_names)

    st.subheader("Hasil Normalisasi Data")
    st.write(scaled_features)

    st.subheader("Target Label")
    labels = pd.get_dummies(df.Penyakit)
    st.write(labels)

# Modeling
with modeling:
    training, test, training_label, test_label = train_test_split(scaled_features, y, test_size=0.1, random_state=900)
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilih Model untuk menghitung akurasi:")
        linear_reg = st.checkbox('Regresi Linear')
        submitted = st.form_submit_button("Submit")

        # Regresi Linear
        if linear_reg:
            linear_model = LinearRegression()
            linear_model.fit(training, training_label)
            linear_pred = linear_model.predict(test)
            mse = mean_squared_error(test_label, linear_pred)
            rmse = np.sqrt(mse)
            linear_accuracy = 1 - rmse/np.mean(test_label)

            if submitted:
                st.write("Model Regresi Linear accuracy score: {0:0.2f}".format(linear_accuracy))

        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi': [linear_accuracy],
                'Model': ['Regresi Linear'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
