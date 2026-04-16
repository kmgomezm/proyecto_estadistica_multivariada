import numpy as np
import pandas as pd
import streamlit as st

from shared import build_feature_metadata, coerce_and_align_features, load_model

st.title("🧪 Test del modelo (predicción individual)")
st.caption("Esta página es solo para probar el modelo con una vivienda nueva.")

model = load_model()
metadata = build_feature_metadata()

st.info("Todas las variables del modelo aparecen debajo. Cada una incluye su significado entre paréntesis.")

with st.form("single_prediction_form"):
    user_input = {}

    # =========================
    # NUMÉRICAS
    # =========================
    st.subheader("Variables numéricas")
    numeric_cols = [c for c, m in metadata.items() if m["type"] == "numeric"]

    for col in numeric_cols:
        m = metadata[col]
        label = f"{col} ({m['description']})"

        # manejar None / errores
        min_val = float(m.get("min", 0))
        max_val = float(m.get("max", 1e6))
        default = float(m.get("default", min_val))

        # step dinámico seguro
        try:
            step = 1.0 if float(max_val).is_integer() else 0.01
        except:
            step = 1.0

        user_input[col] = st.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=default,
            step=step,
        )

    # =========================
    # CATEGÓRICAS
    # =========================
    st.subheader("Variables categóricas")
    cat_cols = [c for c, m in metadata.items() if m["type"] == "categorical"]

    for col in cat_cols:
        m = metadata[col]
        label = f"{col} ({m['description']})"

        options = m.get("options", [])
        default = m.get("default", options[0] if options else None)

        if default in options:
            idx = options.index(default)
        else:
            idx = 0

        st.caption(f"Opciones de {col}: {', '.join(options)}")

        user_input[col] = st.selectbox(label, options, index=idx)

    submitted = st.form_submit_button("Predecir precio")

# =========================
# PREDICCIÓN
# =========================
if submitted:
    try:
        input_df = pd.DataFrame([user_input])
        aligned = coerce_and_align_features(input_df)

        pred = model.predict(aligned)

        # asegurar formato
        pred = np.array(pred).flatten()[0]

        # SOLO si usaste log en entrenamiento
        pred = np.expm1(pred)

        st.success("Predicción completada")
        st.metric("💰 Precio estimado", f"${pred:,.0f}")

    except Exception as e:
        st.error("No se pudo generar la predicción.")
        st.exception(e)