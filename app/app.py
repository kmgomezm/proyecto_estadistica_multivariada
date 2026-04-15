import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏠 Predicción de Precio de Vivienda")
st.write("Modelo basado en regresión avanzada (Kaggle Ames Housing)")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("artifacts/final_model.pkl")
    return model

model = load_model()

# =========================
# INPUTS (SIMPLIFICADO)
# =========================
st.sidebar.header("Características de la vivienda")

def user_input():
    data = {
        "GrLivArea": st.sidebar.number_input("Área habitable (GrLivArea)", 500, 5000, 1500),
        "TotalBsmtSF": st.sidebar.number_input("Área sótano", 0, 3000, 800),
        "GarageCars": st.sidebar.number_input("Capacidad garaje", 0, 5, 2),
        "OverallQual": st.sidebar.slider("Calidad general", 1, 10, 5),
        "YearBuilt": st.sidebar.number_input("Año construcción", 1900, 2025, 2000),
        "FullBath": st.sidebar.number_input("Baños completos", 0, 5, 2),
        "TotRmsAbvGrd": st.sidebar.number_input("Habitaciones", 1, 15, 6),
        "Neighborhood": st.sidebar.selectbox(
            "Barrio",
            ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel"]
        ),
        "HouseStyle": st.sidebar.selectbox(
            "Tipo de casa",
            ["1Story", "2Story", "1.5Fin"]
        ),
        "ExterQual": st.sidebar.selectbox(
            "Calidad exterior",
            ["TA", "Gd", "Ex"]
        )
    }
    
    return pd.DataFrame([data])

input_df = user_input()

st.subheader("📋 Datos de entrada")
st.write(input_df)

# =========================
# PREDICCIÓN
# =========================
if st.button("Predecir precio"):
    
    try:
        prediction_log = model.predict(input_df)
        
        # revertir log
        prediction = np.expm1(prediction_log)
        
        st.success(f"💰 Precio estimado: ${prediction[0]:,.2f}")
    
    except Exception as e:
        st.error("Error en la predicción")
        st.write(e)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.write("Proyecto de Regresión - Estadística Multivariada")