import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏠 Predicción de Precio de Vivienda")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("artifacts/final_model.pkl")

model = load_model()

# =========================
# LOAD TRAIN COLUMNS
# =========================
@st.cache_data
def load_columns():
    df = pd.read_csv("data/clean/X_train.csv")
    return df.columns.tolist()

all_columns = load_columns()

# =========================
# INPUTS
# =========================
st.sidebar.header("Características principales")

user_input = {
    "GrLivArea": st.sidebar.number_input("Área habitable", 500, 5000, 1500),
    "TotalBsmtSF": st.sidebar.number_input("Área sótano", 0, 3000, 800),
    "GarageCars": st.sidebar.number_input("Garaje (autos)", 0, 5, 2),
    "OverallQual": st.sidebar.slider("Calidad general", 1, 10, 5),
    "YearBuilt": st.sidebar.number_input("Año construcción", 1900, 2025, 2000),
    "FullBath": st.sidebar.number_input("Baños", 0, 5, 2),
    "TotRmsAbvGrd": st.sidebar.number_input("Habitaciones", 1, 15, 6),
    "Neighborhood": st.sidebar.selectbox(
        "Barrio",
        ["CollgCr", "NAmes", "OldTown", "Edwards", "Somerst"]
    ),
    "HouseStyle": st.sidebar.selectbox(
        "Tipo vivienda",
        ["1Story", "2Story", "1.5Fin"]
    ),
    "ExterQual": st.sidebar.selectbox(
        "Calidad exterior",
        ["TA", "Gd", "Ex"]
    )
}

# =========================
# CONSTRUIR INPUT COMPLETO
# =========================
input_df = pd.DataFrame([user_input])

# completar columnas faltantes
for col in all_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# ordenar columnas
input_df = input_df[all_columns]

st.subheader("📋 Input final")
st.write(input_df)

# =========================
# PREDICCIÓN
# =========================
if st.button("Predecir precio"):
    
    try:
        pred_log = model.predict(input_df)
        pred = np.expm1(pred_log)
        
        st.metric("💰 Precio estimado", f"${pred[0]:,.0f}")
    
    except Exception as e:
        st.error("Error en predicción")
        st.write(e)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.write("Proyecto de Regresión - Kaggle House Prices")