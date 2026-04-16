import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_model, load_columns, load_combined_results

st.set_page_config(page_title="House Price Predictor", layout="wide")

# =========================
# LOAD
# =========================
@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def get_columns():
    return load_columns()

model = get_model()
all_columns = get_columns()

# =========================
# VARIABLES SELECCIONADAS
# =========================
selected_features = [
    "GrLivArea","TotalBsmtSF","GarageArea","GarageCars","1stFlrSF",
    "YearBuilt","YearRemodAdd","FullBath","TotRmsAbvGrd","LotArea",
    "OverallQual","KitchenQual","ExterQual","BsmtQual",
    "Neighborhood","HouseStyle","BldgType","GarageType","CentralAir","SaleCondition"
]

# =========================
# DEFAULTS (CRÍTICO)
# =========================
@st.cache_data
def get_defaults():
    df = pd.read_csv(os.path.join("data","clean","X_train.csv"))
    defaults = {}

    for col in df.columns:
        if df[col].dtype in ["int64","float64"]:
            defaults[col] = df[col].median()
        else:
            defaults[col] = df[col].mode()[0]

    return defaults

defaults = get_defaults()

# =========================
# UI
# =========================
st.title("🏠 Predicción de precio de vivienda")

tab1, tab2 = st.tabs(["Predicción", "Métricas"])

# =========================
# TAB 1
# =========================
with tab1:

    st.subheader("Características principales")

    input_data = {}

    col1, col2, col3 = st.columns(3)

    def num_input(label, default, help_text):
        return st.number_input(label, value=float(default), help=help_text)

    def cat_input(label, options, default, help_text):
        return st.selectbox(label, options, index=options.index(default) if default in options else 0, help=help_text)

    # =========================
    # NUMÉRICAS
    # =========================
    with col1:
        input_data["GrLivArea"] = num_input("GrLivArea", defaults["GrLivArea"], "Área habitable (sq ft)")
        input_data["TotalBsmtSF"] = num_input("TotalBsmtSF", defaults["TotalBsmtSF"], "Área sótano")
        input_data["GarageArea"] = num_input("GarageArea", defaults["GarageArea"], "Área garaje")
        input_data["GarageCars"] = num_input("GarageCars", defaults["GarageCars"], "Capacidad garaje")

    with col2:
        input_data["1stFlrSF"] = num_input("1stFlrSF", defaults["1stFlrSF"], "Área primer piso")
        input_data["YearBuilt"] = num_input("YearBuilt", defaults["YearBuilt"], "Año construcción")
        input_data["YearRemodAdd"] = num_input("YearRemodAdd", defaults["YearRemodAdd"], "Año remodelación")

    with col3:
        input_data["FullBath"] = num_input("FullBath", defaults["FullBath"], "Baños completos")
        input_data["TotRmsAbvGrd"] = num_input("TotRmsAbvGrd", defaults["TotRmsAbvGrd"], "Habitaciones")
        input_data["LotArea"] = num_input("LotArea", defaults["LotArea"], "Área lote")

    # =========================
    # ORDINALES
    # =========================
    quality_levels = ["Po","Fa","TA","Gd","Ex"]

    input_data["OverallQual"] = st.slider("OverallQual", 1, 10, int(defaults["OverallQual"]), help="Calidad general (1-10)")
    input_data["KitchenQual"] = cat_input("KitchenQual", quality_levels, defaults["KitchenQual"], "Calidad cocina")
    input_data["ExterQual"] = cat_input("ExterQual", quality_levels, defaults["ExterQual"], "Calidad exterior")
    input_data["BsmtQual"] = cat_input("BsmtQual", quality_levels, defaults["BsmtQual"], "Calidad sótano")

    # =========================
    # CATEGÓRICAS
    # =========================
    df = pd.read_csv(os.path.join("data","clean","X_train.csv"))

    for col in ["Neighborhood","HouseStyle","BldgType","GarageType","SaleCondition"]:
        input_data[col] = st.selectbox(
            col,
            sorted(df[col].dropna().unique()),
            help=f"Categoría de {col}"
        )

    input_data["CentralAir"] = st.selectbox("CentralAir", ["Y","N"], help="Aire acondicionado")

    # =========================
    # COMPLETAR VARIABLES
    # =========================
    final_input = defaults.copy()
    final_input.update(input_data)

    df_input = pd.DataFrame([final_input])

    if st.button("Predecir"):
        pred = model.predict(df_input)[0]
        st.success(f"💰 Precio estimado: ${pred:,.0f}")

# =========================
# TAB 2
# =========================
with tab2:

    st.subheader("Comparación de modelos")

    results = load_combined_results()

    results = results.sort_values("rmse_test")

    st.dataframe(results)

    best = results.iloc[0]
    st.success(f"Mejor modelo: {best['model']} | RMSE test: {best['rmse_test']:.2f}")