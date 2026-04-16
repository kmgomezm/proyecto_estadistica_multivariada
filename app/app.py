import streamlit as st

st.set_page_config(page_title="House Price App", layout="wide")

st.title("🏠 House Price Prediction App")
st.info(
    "Esta app usa navegación multipágina. Usa el menú lateral de Streamlit para entrar a: "
    "Test individual, Test por CSV, EDA, SHAP y Comparación de modelos."
)
