import numpy as np
import pandas as pd
import streamlit as st

from shared import coerce_and_align_features, load_model, load_training_data

st.title("🧠 SHAP values (interpretabilidad)")

st.markdown(
    "**¿Sirve SHAP si el mejor modelo es lineal?** Sí. "
    "En modelos lineales, SHAP asigna contribuciones aditivas por variable de forma muy natural."
)

model = load_model()
X_train = load_training_data()
X_sample = X_train.sample(min(200, len(X_train)), random_state=42)

try:
    import shap

    st.success("Librería SHAP disponible. Mostrando explicaciones globales y locales.")

    predictor = lambda x: model.predict(coerce_and_align_features(pd.DataFrame(x, columns=X_train.columns)))

    explainer = shap.Explainer(predictor, X_sample, feature_names=X_train.columns)
    explanation = explainer(X_sample)

    mean_abs_shap = np.abs(explanation.values).mean(axis=0)
    shap_df = pd.DataFrame(
        {"feature": X_train.columns, "mean_abs_shap": mean_abs_shap}
    ).sort_values("mean_abs_shap", ascending=False)

    st.subheader("Importancia global por SHAP (media absoluta)")
    st.dataframe(shap_df.head(20))
    st.bar_chart(shap_df.head(20).set_index("feature")["mean_abs_shap"])

    st.subheader("Contribución local (una observación)")
    row_idx = st.slider("Selecciona la observación", 0, len(X_sample) - 1, 0)
    local_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "shap_value": explanation.values[row_idx],
            "feature_value": X_sample.iloc[row_idx].values,
        }
    ).sort_values("shap_value", key=lambda s: s.abs(), ascending=False)

    st.dataframe(local_df.head(20))

except Exception as e:
    st.warning(
        "No fue posible calcular SHAP en este entorno. "
        "Si quieres habilitarlo, instala `shap` en requirements."
    )
    st.exception(e)
