# src/utils.py
import pandas as pd
import joblib
import os
import glob

# =========================
# ROOT 
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# =========================
# MODEL
# =========================
def load_model():
    path = os.path.join(PROJECT_ROOT, "artifacts", "final_model.pkl")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo no encontrado en {path}")
    
    return joblib.load(path)

# =========================
# RESULTS
# =========================
def load_results():
    path = os.path.join(PROJECT_ROOT, "data", "results", "test_results.csv")
    
    if not os.path.exists(path):
        raise FileNotFoundError("test_results.csv no encontrado")
    
    return pd.read_csv(path)

# =========================
# TRAIN DATA
# =========================
def load_train():
    return pd.read_csv(os.path.join(PROJECT_ROOT, "data", "clean", "train.csv"))

def load_columns():
    return pd.read_csv(
        os.path.join(PROJECT_ROOT, "data", "clean", "X_train.csv")
    ).columns.tolist()

# =========================
# COMBINED METRICS
# =========================
def load_combined_results():
    
    metrics_path = os.path.join(PROJECT_ROOT, "artifacts", "metrics")
    files = glob.glob(os.path.join(metrics_path, "*_metrics.csv"))

    if len(files) == 0:
        raise ValueError("No se encontraron archivos de métricas en artifacts/metrics")

    train_dfs = []

    for file in files:
        df = pd.read_csv(file)

        model_name = os.path.basename(file).replace("_metrics.csv", "")
        df["model"] = model_name

        train_dfs.append(df)

    train_df = pd.concat(train_dfs, ignore_index=True)

    # =========================
    # TEST
    # =========================
    test_df = load_results()

    # Validación
    if "model" not in test_df.columns:
        raise ValueError("test_results.csv no tiene columna 'model'")

    # =========================
    # MERGE
    # =========================
    final_df = train_df.merge(test_df, on="model", how="inner")

    if final_df.empty:
        raise ValueError("El merge quedó vacío → revisa nombres de modelos")

    return final_df