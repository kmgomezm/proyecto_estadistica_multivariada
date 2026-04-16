import os
import glob

import pandas as pd
import streamlit as st

from shared import PROJECT_ROOT

st.title("🏁 Comparación de modelos (train vs test)")
st.caption("Comparativa del desempeño en entrenamiento (CV) y prueba final.")

metrics_path = os.path.join(PROJECT_ROOT, "artifacts", "metrics")
metric_files = glob.glob(os.path.join(metrics_path, "*_metrics.csv"))

if not metric_files:
    st.error("No se encontraron métricas de entrenamiento en artifacts/metrics.")
else:
    train_dfs = []
    for file in metric_files:
        df = pd.read_csv(file)
        model_name = os.path.basename(file).replace("_metrics.csv", "")
        df["model"] = model_name
        train_dfs.append(df)

    train_df = pd.concat(train_dfs, ignore_index=True)

    test_path = os.path.join(PROJECT_ROOT, "data", "results", "test_results.csv")
    test_df = pd.read_csv(test_path)

    merged = train_df.merge(test_df, on="model", how="inner")
    merged = merged.sort_values("rmse_test").reset_index(drop=True)

    st.subheader("Tabla de métricas")
    st.dataframe(merged)

    st.subheader("RMSE: train (CV) vs test")
    rmse_compare = merged[["model", "rmse_mean", "rmse_test"]].set_index("model")
    st.bar_chart(rmse_compare)

    st.subheader("MAE: train (CV) vs test")
    mae_compare = merged[["model", "mae_mean", "mae_test"]].set_index("model")
    st.bar_chart(mae_compare)

    st.subheader("R²: train (CV) vs test")
    r2_compare = merged[["model", "r2_mean", "r2_test"]].set_index("model")
    st.bar_chart(r2_compare)

    best = merged.iloc[0]
    st.success(
        f"Mejor modelo por RMSE de test: {best['model']} | "
        f"RMSE test={best['rmse_test']:.2f}, RMSE train={best['rmse_mean']:.2f}"
    )
