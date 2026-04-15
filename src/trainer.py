# src/trainer.py

import numpy as np
import pandas as pd
import joblib
import os
from contextlib import nullcontext

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn


# =========================
# MÉTRICAS
# =========================
def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "rmse": rmse,
        "mse": mse,
        "mae": mae,
        "r2": r2
    }


# =========================
# TRAIN + CV (SIN GRID)
# =========================
def train_cv(model_name, model, pipeline_builder, X, y, cv=5):
    
    pipe = Pipeline([
        ("features", pipeline_builder(X)),
        ("model", model)
    ])
    
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2"
    }
    
    results = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    
    metrics = {
        "rmse_mean": -np.mean(results["test_rmse"]),
        "mse_mean": -np.mean(results["test_mse"]),
        "mae_mean": -np.mean(results["test_mae"]),
        "r2_mean": np.mean(results["test_r2"])
    }
    
    return pipe, metrics


# =========================
# TRAIN + GRIDSEARCH
# =========================
def train_grid(model_name, model, param_grid, pipeline_builder, X, y, cv=5):
    
    pipe = Pipeline([
        ("features", pipeline_builder(X)),
        ("model", model)
    ])
    
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X, y)
    
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    
    # evaluación final sobre train (consistente con CV)
    y_pred = best_model.predict(X)
    metrics = compute_metrics(y, y_pred)
    
    return best_model, best_params, metrics


# =========================
# SAVE RESULTS
# =========================
def save_model(model, model_name, path="../artifacts/models"):
    
    os.makedirs(path, exist_ok=True)
    
    file_path = os.path.join(path, f"{model_name}.pkl")
    joblib.dump(model, file_path)
    
    return file_path


def save_metrics(metrics, model_name, path="../artifacts/metrics"):
    
    os.makedirs(path, exist_ok=True)
    
    df = pd.DataFrame([metrics])
    file_path = os.path.join(path, f"{model_name}_metrics.csv")
    
    df.to_csv(file_path, index=False)
    
    return file_path


# =========================
# MASTER TRAINER
# =========================
def run_training(
    model_name,
    model,
    pipeline_builder,
    X,
    y,
    param_grid=None,
    cv=5,
    use_mlflow=True
):
    
    with mlflow.start_run(run_name=model_name) if use_mlflow else nullcontext():
        
        if param_grid is not None:
            best_model, best_params, metrics = train_grid(
                model_name, model, param_grid, pipeline_builder, X, y, cv
            )
        else:
            best_model, metrics = train_cv(
                model_name, model, pipeline_builder, X, y, cv
            )
            best_params = {}
        
        # LOG MLFLOW
        if use_mlflow:
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            
            for k, v in best_params.items():
                mlflow.log_param(k, v)
            
            mlflow.sklearn.log_model(best_model, name="model")
        
        # SAVE LOCAL
        model_path = save_model(best_model, model_name)
        metrics_path = save_metrics(metrics, model_name)
        
        return {
            "model_name": model_name,
            "metrics": metrics,
            "params": best_params,
            "model_path": model_path,
            "metrics_path": metrics_path
        }