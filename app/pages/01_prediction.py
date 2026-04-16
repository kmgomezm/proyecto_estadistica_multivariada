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

    st.subheader("Variables numéricas")
    numeric_cols = [c for c, m in metadata.items() if m["type"] == "numeric"]
    for col in numeric_cols:
        m = metadata[col]
        label = f"{col} ({m['description']})"
        step = 1.0 if float(m["max"]).is_integer() else 0.01
        user_input[col] = st.number_input(
            label,
            min_value=float(m["min"]),
            max_value=float(m["max"]),
            value=float(m["default"]),
            step=step,
        )

    st.subheader("Variables categóricas")
    cat_cols = [c for c, m in metadata.items() if m["type"] == "categorical"]
    for col in cat_cols:
        m = metadata[col]
        label = f"{col} ({m['description']})"
        st.caption(f"Opciones de {col}: {', '.join(m['options'])}")
        user_input[col] = st.selectbox(label, m["options"], index=0)

    submitted = st.form_submit_button("Predecir precio")

if submitted:
    try:
        input_df = pd.DataFrame([user_input])
        aligned = coerce_and_align_features(input_df)
        pred_log = model.predict(aligned)
        pred = np.expm1(pred_log)

        st.success("Predicción completada")
        st.metric("💰 Precio estimado", f"${pred[0]:,.0f}")
    except Exception as e:
        st.error("No se pudo generar la predicción.")
        st.exception(e)
