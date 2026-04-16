import os
import joblib
import pandas as pd
import streamlit as st
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DESCRIPTION_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "data_description.txt")
MAX_BATCH_ROWS = 1000

# Streamlit Cloud can execute from /app, so ensure the repo root is importable
# before unpickling models that reference modules like `src.*`.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(PROJECT_ROOT, "artifacts", "final_model.pkl"))


@st.cache_data
def load_training_data():
    return pd.read_csv(os.path.join(PROJECT_ROOT, "data", "clean", "X_train.csv"))


@st.cache_data
def parse_data_descriptions(path: str = DATA_DESCRIPTION_PATH):
    descriptions = {}
    options = {}

    if not os.path.exists(path):
        return descriptions, options

    current_var = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if not stripped:
                continue

            if ":" in stripped and not line.startswith(" "):
                key, desc = stripped.split(":", 1)
                current_var = key.strip()
                descriptions[current_var] = desc.strip()
                options.setdefault(current_var, [])
                continue

            if current_var and line.startswith(" ") and "\t" in stripped:
                parts = [p.strip() for p in stripped.split("\t") if p.strip()]
                if len(parts) >= 1:
                    opt_value = parts[0]
                    if opt_value not in options[current_var]:
                        options[current_var].append(opt_value)

    return descriptions, options


@st.cache_data
def build_feature_metadata():
    x_train = load_training_data()
    descriptions, desc_options = parse_data_descriptions()

    metadata = {}
    for col in x_train.columns:
        if x_train[col].dtype == "object":
            train_options = sorted(x_train[col].dropna().astype(str).unique().tolist())
            options = train_options if train_options else desc_options.get(col, [])
            default_value = options[0] if options else ""
            metadata[col] = {
                "type": "categorical",
                "description": descriptions.get(col, "Sin descripción en data_description"),
                "options": options,
                "default": default_value,
            }
        else:
            col_min = float(x_train[col].min())
            col_max = float(x_train[col].max())
            median = float(x_train[col].median())
            metadata[col] = {
                "type": "numeric",
                "description": descriptions.get(col, "Sin descripción en data_description"),
                "min": col_min,
                "max": col_max,
                "default": median,
            }

    return metadata


def coerce_and_align_features(df: pd.DataFrame):
    x_train = load_training_data()
    meta = build_feature_metadata()
    expected_columns = x_train.columns.tolist()

    aligned = df.copy()

    for col in expected_columns:
        if col not in aligned.columns:
            if meta[col]["type"] == "numeric":
                aligned[col] = meta[col]["default"]
            else:
                aligned[col] = meta[col]["default"]

    aligned = aligned[expected_columns]

    for col in expected_columns:
        if meta[col]["type"] == "numeric":
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
            aligned[col] = aligned[col].fillna(meta[col]["default"])
        else:
            aligned[col] = aligned[col].astype(str)
            valid_options = set(meta[col]["options"])
            default = meta[col]["default"]
            aligned[col] = aligned[col].apply(lambda x: x if x in valid_options else default)

    return aligned
