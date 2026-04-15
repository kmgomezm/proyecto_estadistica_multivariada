import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# =========================
# PATH FIX (import src en deploy)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PROJECT_ROOT)

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏠 Predicción de Precio de Vivienda")
st.caption("Comparación de modelos + predicción con el mejor modelo")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(PROJECT_ROOT, "artifacts", "final_model.pkl"))

model = load_model()

# =========================
# LOAD RESULTS (RMSE)
# =========================
@st.cache_data
def load_results():
    path = os.path.join(PROJECT_ROOT, "data", "results", "test_results.csv")
    return pd.read_csv(path)

results_df = load_results().sort_values("rmse_test").reset_index(drop=True)

# =========================
# LOAD TRAIN COLUMNS
# =========================
@st.cache_data
def load_columns():
    path = os.path.join(PROJECT_ROOT, "data", "clean", "X_train.csv")
    return pd.read_csv(path).columns.tolist()

all_columns = load_columns()

# =========================
# SIDEBAR INPUTS
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
# BUILD FULL INPUT
# =========================
input_df = pd.DataFrame([user_input])

for col in all_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[all_columns]

# =========================
# PREDICTION
# =========================
st.subheader("🔮 Predicción")

if st.button("Predecir precio"):
    try:
        pred_log = model.predict(input_df)
        pred = np.expm1(pred_log)

        st.metric("💰 Precio estimado", f"${pred[0]:,.0f}")
    except Exception as e:
        st.error("Error en la predicción")
        st.write(e)

# =========================
# MODEL COMPARISON
# =========================
st.markdown("---")
st.subheader("📊 RMSE en conjunto de prueba")

st.dataframe(results_df)

# =========================
# BEST MODEL
# =========================
best_model = results_df.iloc[0]

st.success(f"""
🏆 Mejor modelo: {best_model['model']}

RMSE: {best_model['rmse_test']:.0f}
MAE: {best_model['mae_test']:.0f}
R²: {best_model['r2_test']:.3f}
""")

# =========================
# PLOT
# =========================
st.subheader("📉 Comparación RMSE")

st.bar_chart(
    results_df.set_index("model")["rmse_test"]
)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.write("Proyecto de Regresión - House Prices (Kaggle)")