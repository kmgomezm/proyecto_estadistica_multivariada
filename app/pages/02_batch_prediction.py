import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("📤 Predicción masiva")

uploaded_file = st.file_uploader("Sube un CSV", type=["csv"])

MAX_ROWS = 500

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Preview:", df.head())

    if len(df) > MAX_ROWS:
        st.error(f"Máximo permitido: {MAX_ROWS} filas")
    else:
        try:
            preds = np.expm1(model.predict(df))
            df["PredictedPrice"] = preds

            st.success("Predicción completada")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar resultados", csv, "predicciones.csv")

        except Exception as e:
            st.error(e)