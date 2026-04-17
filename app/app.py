import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# =========================
# PATH FIX
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.utils import load_model, load_combined_results

st.set_page_config(page_title="House Price Predictor", layout="wide")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "data", "clean", "X_train.csv"))

df = load_data()

# =========================
# MODEL
# =========================
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# =========================
# VARIABLES (SHAP)
# =========================
selected_features = [
    "Neighborhood","MSZoning","HouseStyle","SaleCondition","SaleType","Condition1",
    "OverallQual","OverallCond","Functional","GarageQual","Foundation",
    "GrLivArea","TotalBsmtSF","1stFlrSF","2ndFlrSF","GarageArea","BsmtFinSF1",
    "YearBuilt","YearRemodAdd",
    "Fireplaces","TotRmsAbvGrd"
]

# =========================
# DEFAULTS POR VECINDARIO
# =========================
def compute_defaults(neighborhood):

    df_sub = df[df["Neighborhood"] == neighborhood]

    # fallback si pocos datos
    if len(df_sub) < 10:
        df_sub = df

    defaults = {}

    for col in df.columns:

        # =========================
        # NUMÉRICAS
        # =========================
        if df[col].dtype in ["int64", "float64"]:
            val = df_sub[col].median()

            if pd.isna(val):
                val = df[col].median()

            defaults[col] = val

        # =========================
        # CATEGÓRICAS
        # =========================
        else:
            mode_series = df_sub[col].mode()

            if len(mode_series) > 0:
                val = mode_series.iloc[0]
            else:
                # fallback global
                global_mode = df[col].mode()
                val = global_mode.iloc[0] if len(global_mode) > 0 else "NA"

            defaults[col] = val

    return defaults

# =========================
# UI
# =========================
st.title("🏠 Predicción de precio de vivienda")

tab1, tab2 = st.tabs(["Predicción", "Métricas"])

# =========================
# TAB 1
# =========================
with tab1:

    st.subheader("Características de la vivienda")

    # -------------------------
    # Neighborhood primero
    # -------------------------
    neighborhoods = sorted(df["Neighborhood"].dropna().unique())

    selected_neigh = st.selectbox(
        "Neighborhood",
        neighborhoods,
        help="Ubicación de la vivienda"
    )

    defaults = compute_defaults(selected_neigh)

    input_data = {}

    # =========================
    # FUNCIONES INPUT
    # =========================
    def num_input(col):
        min_val = df[col].min()
        max_val = df[col].max()
        default = defaults[col]

        return st.number_input(
            col,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default),
            help=f"Rango típico: {min_val:.0f} - {max_val:.0f}"
        )

    def cat_input(col):
        options = sorted(df[col].dropna().unique())
        default = defaults[col]

        idx = options.index(default) if default in options else 0

        return st.selectbox(
            col,
            options,
            index=idx
        )

    # =========================
    # INPUTS
    # =========================
    col1, col2, col3 = st.columns(3)

    with col1:
        input_data["GrLivArea"] = num_input("GrLivArea")
        input_data["TotalBsmtSF"] = num_input("TotalBsmtSF")
        input_data["1stFlrSF"] = num_input("1stFlrSF")
        input_data["2ndFlrSF"] = num_input("2ndFlrSF")

    with col2:
        input_data["GarageArea"] = num_input("GarageArea")
        input_data["BsmtFinSF1"] = num_input("BsmtFinSF1")
        input_data["YearBuilt"] = num_input("YearBuilt")
        input_data["YearRemodAdd"] = num_input("YearRemodAdd")

    with col3:
        input_data["Fireplaces"] = num_input("Fireplaces")
        input_data["TotRmsAbvGrd"] = num_input("TotRmsAbvGrd")

    # =========================
    # ORDINALES
    # =========================
    input_data["OverallQual"] = st.slider(
        "OverallQual", 1, 10, int(defaults["OverallQual"])
    )

    input_data["OverallCond"] = st.slider(
        "OverallCond", 1, 10, int(defaults["OverallCond"])
    )

    # =========================
    # CATEGÓRICAS
    # =========================
    for col in [
        "MSZoning","HouseStyle","SaleCondition",
        "SaleType","Condition1","Functional",
        "GarageQual","Foundation"
    ]:
        input_data[col] = cat_input(col)

    input_data["Neighborhood"] = selected_neigh

    # =========================
    # COMPLETAR VECTOR
    # =========================
    final_input = defaults.copy()
    final_input.update(input_data)

    df_input = pd.DataFrame([final_input])

    st.info("Las variables no visibles se completan automáticamente según el vecindario.")

    # =========================
    # PREDICCIÓN
    # =========================
    if st.button("Predecir precio"):

        try:
            pred = model.predict(df_input)[0]
            price = np.expm1(pred)
            st.success(f"💰 Precio estimado: ${price:,.0f}")

        except Exception as e:
            st.error(f"Error: {e}")

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