import os
import sys
import joblib
import pandas as pd
import streamlit as st
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DESCRIPTION_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "data_description.txt")
MAX_BATCH_ROWS = 1000

def ensure_src_package_available():
    """Make sure modules under src.* are importable before unpickling models."""
    candidates = [
        PROJECT_ROOT,
        os.getcwd(),
        os.path.dirname(os.getcwd()),
    ]

    for base_path in candidates:
        if not base_path:
            continue

        src_path = os.path.join(base_path, "src")
        if os.path.isdir(src_path) and base_path not in sys.path:
            sys.path.insert(0, base_path)

    # Force import now so joblib/pickle can resolve src.* references.
    import src.preprocessing  # noqa: F401


@st.cache_resource
def load_model():
    ensure_src_package_available()
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
        if not pd.api.types.is_numeric_dtype(x_train[col]):
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
            numeric_col = pd.to_numeric(x_train[col], errors="coerce")
            col_min = float(numeric_col.min())
            col_max = float(numeric_col.max())
            median = float(numeric_col.median())
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
