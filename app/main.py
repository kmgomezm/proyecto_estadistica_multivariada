import streamlit as st

st.set_page_config(page_title="House Price App", layout="wide")

st.title("🏠 House Price Prediction App")

st.markdown("""
Aplicación de Machine Learning para predicción de precios de vivienda.

### 🔹 Funcionalidades:
- Predicción individual
- Predicción masiva (CSV)
- Análisis exploratorio (EDA)
- Interpretabilidad (SHAP)
- Comparación de modelos

### 🔹 Dataset:
Kaggle House Prices (79 variables, ~1460 observaciones)
""")