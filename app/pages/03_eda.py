import os

import pandas as pd
import streamlit as st

from shared import PROJECT_ROOT

st.title("📈 EDA rápido del dataset")
st.caption("Análisis exploratorio básico para entender calidad y distribución de los datos.")

train_path = os.path.join(PROJECT_ROOT, "data", "raw", "train.csv")
train_df = pd.read_csv(train_path)

st.subheader("Tamaño y tipos")
col1, col2, col3 = st.columns(3)
col1.metric("Filas", f"{train_df.shape[0]:,}")
col2.metric("Columnas", f"{train_df.shape[1]:,}")
col3.metric("Variables numéricas", str(train_df.select_dtypes(include="number").shape[1]))

st.subheader("Muestra de datos")
st.dataframe(train_df.head(20))

st.subheader("Valores faltantes")
missing = train_df.isna().sum().sort_values(ascending=False)
missing = missing[missing > 0]
if missing.empty:
    st.success("No hay valores faltantes.")
else:
    missing_df = missing.reset_index()
    missing_df.columns = ["variable", "n_missing"]
    missing_df["pct_missing"] = (missing_df["n_missing"] / len(train_df) * 100).round(2)
    st.dataframe(missing_df)
    st.bar_chart(missing_df.set_index("variable")["pct_missing"])

st.subheader("Distribución del objetivo (SalePrice)")
if "SalePrice" in train_df.columns:
    st.write(train_df["SalePrice"].describe())
    st.bar_chart(train_df["SalePrice"])

st.subheader("Top correlaciones con SalePrice")
if "SalePrice" in train_df.columns:
    num_df = train_df.select_dtypes(include="number")
    corrs = num_df.corr(numeric_only=True)["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False).head(15)
    st.dataframe(corrs.reset_index().rename(columns={"index": "variable", "SalePrice": "|corr|"}))
    st.bar_chart(corrs)
