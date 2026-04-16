import streamlit as st

st.set_page_config(page_title="House Price App", layout="wide")

st.title("🏠 House Price Prediction App")

st.markdown(
    """
Aplicación de Machine Learning para predicción de precios de vivienda.

### 🔹 Páginas disponibles
- **Test del modelo (individual)**: formulario completo con todas las variables y significado.
- **Test por CSV**: carga masiva de viviendas para predecir en lote.
- **EDA**: análisis exploratorio rápido del dataset.
- **SHAP values**: interpretabilidad global y local del modelo.
- **Comparación de modelos**: métricas train (CV) vs test.

### 🔹 Dataset
Kaggle House Prices (Ames, Iowa).
"""
)
