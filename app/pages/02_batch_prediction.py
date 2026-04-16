import numpy as np
import pandas as pd
import streamlit as st

from shared import MAX_BATCH_ROWS, coerce_and_align_features, load_model, load_training_data

st.title("📤 Test del modelo por CSV")
st.caption("Sube un archivo CSV para estimar precios de múltiples casas en lote.")

model = load_model()
expected_cols = load_training_data().columns.tolist()

st.markdown(f"**Máximo permitido:** `{MAX_BATCH_ROWS}` casas (filas) por archivo.")

with st.expander("Ver columnas esperadas por el modelo"):
    st.write(expected_cols)

uploaded_file = st.file_uploader("Sube tu CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Vista previa del archivo cargado", df.head())

    if len(df) > MAX_BATCH_ROWS:
        st.error(f"El archivo tiene {len(df)} filas. El máximo permitido es {MAX_BATCH_ROWS}.")
    else:
        missing_cols = [c for c in expected_cols if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in expected_cols]

        if missing_cols:
            st.warning(
                "Faltan columnas en el CSV. Se completarán automáticamente con valores por defecto: "
                + ", ".join(missing_cols)
            )
        if extra_cols:
            st.info("Columnas extra detectadas y descartadas: " + ", ".join(extra_cols))

        try:
            aligned = coerce_and_align_features(df)
            preds = np.expm1(model.predict(aligned))

            result_df = df.copy()
            result_df["PredictedPrice"] = preds

            st.success("Predicción masiva completada.")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar resultados",
                data=csv,
                file_name="predicciones_lote.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error("No se pudo procesar el CSV.")
            st.exception(e)
