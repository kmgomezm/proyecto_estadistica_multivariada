# src/utils.py
import pandas as pd
import joblib
import os
import glob


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_model():
    return joblib.load(os.path.join(PROJECT_ROOT, "artifacts", "final_model.pkl"))

def load_results():
    path = os.path.join(PROJECT_ROOT, "data", "results", "test_results.csv")
    return pd.read_csv(path)

def load_full_results():
    path = os.path.join(PROJECT_ROOT, "data", "results", "full_results.csv")
    return pd.read_csv(path)

def load_train():
    return pd.read_csv(os.path.join(PROJECT_ROOT, "data", "clean", "train.csv"))

def load_columns():
    return pd.read_csv(
        os.path.join(PROJECT_ROOT, "data", "clean", "X_train.csv")
    ).columns.tolist()

def load_combined_results():
    # =========================
    # LOAD TRAIN (CV)
    # =========================
    metrics_path = os.path.join(PROJECT_ROOT, "artifacts", "metrics")
    files = glob.glob(os.path.join(metrics_path, "*_metrics.csv"))

    train_dfs = []

    for file in files:
        df = pd.read_csv(file)

        model_name = os.path.basename(file).replace("_metrics.csv", "")
        df["model"] = model_name

        train_dfs.append(df)

    train_df = pd.concat(train_dfs, ignore_index=True)

    # =========================
    # LOAD TEST
    # =========================
    test_df = load_results()

    # =========================
    # MERGE
    # =========================
    final_df = train_df.merge(test_df, on="model", how="inner")

    return final_df