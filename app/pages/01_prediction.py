import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(PROJECT_ROOT, "artifacts", "final_model.pkl"))

model = load_model()

st.title("🤖 Predicción de vivienda")

st.header("Características")

# =========================
# NUMÉRICAS
# =========================
GrLivArea = st.number_input(
    "GrLivArea (Área habitable en pies cuadrados)",
    300, 5000, 1500
)

TotalBsmtSF = st.number_input(
    "TotalBsmtSF (Área total del sótano)",
    0, 3000, 800
)

GarageCars = st.number_input(
    "GarageCars (Capacidad del garaje)",
    0, 5, 2
)

YearBuilt = st.number_input(
    "YearBuilt (Año de construcción)",
    1900, 2025, 2000
)

# =========================
# ORDINALES
# =========================
OverallQual = st.slider(
    "OverallQual (Calidad general: 1=Muy pobre, 10=Excelente)",
    1, 10, 5
)

ExterQual = st.selectbox(
    "ExterQual (Calidad exterior)",
    ["Po", "Fa", "TA", "Gd", "Ex"]
)

# =========================
# CATEGÓRICAS
# =========================
Neighborhood = st.selectbox(
    "Neighborhood (Barrio)",
    ["CollgCr", "NAmes", "OldTown", "Edwards", "Somerst"]
)

HouseStyle = st.selectbox(
    "HouseStyle (Tipo de vivienda)",
    ["1Story", "2Story", "1.5Fin"]
)