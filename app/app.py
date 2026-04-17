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

@st.cache_data
def load_full_data():
    X = pd.read_csv(os.path.join(BASE_DIR, "data", "clean", "X_train.csv"))
    y = pd.read_csv(os.path.join(BASE_DIR, "data", "clean", "y_train.csv"))

    df_full = X.copy()
    df_full["SalePrice"] = y.values

    return df_full

df_full = load_full_data()

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

feature_descriptions = {
    "Neighborhood": "Vecindario donde se ubica la vivienda",
    "MSZoning": "Clasificación de zonificación (residencial, comercial, etc.)",
    "HouseStyle": "Tipo de estructura (1 piso, 2 pisos, etc.)",
    "SaleCondition": "Condición de la venta (normal, foreclosure, etc.)",
    "SaleType": "Tipo de transacción (efectivo, crédito, etc.)",
    "Condition1": "Proximidad a condiciones externas (vías, parques, etc.)",

    "OverallQual": "Calidad general de materiales y acabados (1–10)",
    "OverallCond": "Condición general de la vivienda (1–10)",
    "Functional": "Nivel de funcionalidad de la vivienda",
    "GarageQual": "Calidad del garaje",
    "Foundation": "Tipo de cimentación",

    "GrLivArea": "Área habitable sobre el nivel del suelo (pies cuadrados)",
    "TotalBsmtSF": "Área total del sótano",
    "1stFlrSF": "Área del primer piso",
    "2ndFlrSF": "Área del segundo piso",
    "GarageArea": "Área del garaje",
    "BsmtFinSF1": "Área terminada del sótano",

    "YearBuilt": "Año de construcción",
    "YearRemodAdd": "Año de remodelación",

    "Fireplaces": "Número de chimeneas",
    "TotRmsAbvGrd": "Número total de habitaciones (sin incluir baños)"
}

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

tab1, tab2, tab3 = st.tabs(["Predicción", "Métricas", "Análisis"])

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

        desc = feature_descriptions.get(col, "")

        help_text = f"""
        {desc}

        Rango típico: {min_val:.0f} - {max_val:.0f}
        """

        return st.number_input(
            col,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default),
            help=help_text
        )

    def cat_input(col):
        options = sorted(df[col].dropna().unique())
        default = defaults[col]

        desc = feature_descriptions.get(col, "")

        help_text = f"""
        {desc}

        Valores posibles: {', '.join(options[:5])}...
        """

        idx = options.index(default) if default in options else 0

        return st.selectbox(
            col,
            options,
            index=idx,
            help=help_text
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

    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    st.subheader("Comparación de modelos")

    # =========================
    # CARGAR RESULTADOS
    # =========================
    results = load_combined_results()

    # =========================
    # NORMALIZAR MÉTRICAS TRAIN (LOG SCALE)
    # =========================
    results["rmse_train"] = results["rmse"].fillna(results["rmse_mean"])
    results["mae_train"] = results["mae"].fillna(results["mae_mean"])
    results["r2_train"]  = results["r2"].fillna(results["r2_mean"])

    # =========================
    # TABLA FINAL
    # =========================
    results_clean = results[[
        "model",
        "rmse_train", "mae_train", "r2_train",
        "rmse_test", "mae_test", "r2_test"
    ]].copy()

    results_clean = results_clean.round(3)
    results_clean = results_clean.sort_values("rmse_test")

    # =========================
    # TABLA COMPARATIVA
    # =========================
    st.markdown("### Comparación entre modelos")

    st.dataframe(results_clean)

    if not results_clean.empty:
        best = results_clean.iloc[0]
        st.success(
            f"Mejor modelo: {best['model']} | RMSE test: {best['rmse_test']:.2f}"
        )

    st.caption(
        "Nota: Las métricas de entrenamiento en la tabla están en escala logarítmica "
        "(validación cruzada), mientras que las métricas de test están en escala original."
    )

# =========================
# TAB 3
# =========================

with tab3:

    st.subheader("Análisis exploratorio del dataset")

    # =========================
    # FILTRO
    # =========================
    neighborhoods = ["Todos"] + sorted(df_full["Neighborhood"].dropna().unique())

    selected_neigh = st.selectbox(
        "Filtrar por vecindario",
        neighborhoods
    )

    if selected_neigh != "Todos":
        df_plot = df_full[df_full["Neighborhood"] == selected_neigh]
    else:
        df_plot = df_full.copy()

    # =========================
    # KPIs
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("Precio promedio", f"${df_plot['SalePrice'].mean():,.0f}")
    col2.metric("Mediana", f"${df_plot['SalePrice'].median():,.0f}")
    col3.metric("Observaciones", len(df_plot))

    # =========================
    # DISTRIBUCIÓN
    # =========================
    st.subheader("Distribución del precio")

    st.bar_chart(df_plot["SalePrice"].value_counts().sort_index())

    # =========================
    # PRECIO VS ÁREA
    # =========================
    st.subheader("Precio vs Área habitable")

    scatter_df = df_plot[["GrLivArea", "SalePrice"]].dropna()

    st.scatter_chart(scatter_df, x="GrLivArea", y="SalePrice")

    # =========================
    # PRECIO VS CALIDAD
    # =========================
    st.subheader("Precio por calidad (OverallQual)")

    qual_df = df_plot.groupby("OverallQual")["SalePrice"].mean()

    st.line_chart(qual_df)

    # =========================
    # PRECIO POR VECINDARIO
    # =========================
    if selected_neigh == "Todos":

        st.subheader("Precio promedio por vecindario")

        neigh_df = (
            df_full.groupby("Neighborhood")["SalePrice"]
            .mean()
            .sort_values(ascending=False)
        )

        st.bar_chart(neigh_df)