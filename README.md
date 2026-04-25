# Proyecto Estadística Multivariada — Predicción de Precios de Casas

## Descripción

Proyecto de estadística multivariada que entrena múltiples modelos de aprendizaje supervisado para predecir precios de viviendas (`SalePrice`). Utiliza técnicas de preprocesamiento, validación cruzada, tuning de hiperparámetros y tracking de experimentos con MLflow.

---

## Dataset

- **Fuente:** `datos/raw/train.csv` (1,460 registros) y `test.csv` (1,459 registros)
- **Variable objetivo:** `SalePrice` (precio de venta en dólares)
- **Variables:** 80 características incluyendo área, calidad, año, materiales, condiciones, etc.
- **Procesamiento:** Datos transformados a escala logarítmica `log(SalePrice)`

---

## Estructura del Proyecto

```
├── data/
│   ├── raw/              → Datos originales (train.csv, test.csv)
│   ├── processed/        → Datos procesados (.npy, .csv)
│   ├── clean/            → Datos limpios y transformados
│   └── results/          → Resultados de predicciones
├── notebooks/            → Análisis y experimentos
│   ├── 01_EDA_*.ipynb           → Análisis exploratorio
│   ├── 02_preprocessing.ipynb   → Preprocesamiento de datos
│   ├── 03_modeling_baseline.ipynb → Modelo baseline
│   ├── 03_modeling.ipynb        → Entrenamiento de modelos
│   └── 04_shap_values.ipynb     → Interpretabilidad
├── src/                  → Código reutilizable
│   ├── preprocessing.py  → Pipeline de preprocesamiento
│   ├── trainer.py        → Funciones de entrenamiento
│   ├── utils.py          → Funciones auxiliares
│   └── __init__.py
├── app/                  → Aplicación web
│   ├── app.py
│   └── requirements.txt
├── artifacts/            → Modelos y métricas entrenados
│   ├── models/           → Modelos guardados (.pkl)
│   └── metrics/          → Métricas de evaluación (.csv)
└── mlruns/               → Tracking de experimentos (MLflow)
```

---

## Modelos Entrenados

**Familia 1 — Modelos Lineales Regularizados**
- Linear Regression
- Ridge Regression
- Lasso Regression

**Familia 2 — Modelos Basados en Árboles y Boosting**
- Decision Tree (CART)
- Random Forest
- XGBoost
- LightGBM
- CatBoost

**Familia 3 — Redes Neuronales**
- Multilayer Perceptron (MLP)

---

## Métricas de Evaluación

| Métrica | Descripción |
|---|---|
| **RMSE (log)** | Error cuadrático medio en escala logarítmica |
| **RMSE ($)** | Error cuadrático medio en dólares |
| **MAE** | Error absoluto medio |
| **MAPE** | Error absoluto porcentual medio |
| **R² Score** | Coeficiente de determinación |

---

## Flujo de Trabajo

1. **EDA:** Análisis exploratorio de datos y estadísticas descriptivas
2. **Preprocesamiento:** Manejo de valores faltantes, codificación, escalado
3. **Baseline:** Modelo de referencia (media) para comparación
4. **Modelado:** Entrenamiento con validación cruzada (5-fold)
5. **Tuning:** Búsqueda de hiperparámetros óptimos
6. **Interpretabilidad:** Análisis SHAP para importancia de características

---

## Herramientas y Librerías

| Librería | Uso |
|---|---|
| `scikit-learn` | Modelos, pipeline, validación |
| `XGBoost`, `LightGBM`, `CatBoost` | Modelos de boosting |
| `TensorFlow / Keras` | Redes neuronales |
| `MLflow` | Tracking de experimentos |
| `pandas`, `numpy` | Manipulación de datos |
| `matplotlib`, `seaborn` | Visualización |

---

## Uso Básico

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar notebooks en orden
#    01_EDA → 02_preprocessing → 03_modeling → 04_shap_values

# 3. Consultar historial de experimentos con MLflow
mlflow ui
```

Los modelos entrenados se guardan automáticamente en `artifacts/models/`.

---

## Resultados Esperados

Los mejores modelos logran:

- **RMSE ($):** ~$18,000–$25,000 USD
- **R² Score:** ~0.90–0.94
- **Características más importantes:** área habitable, calidad general, ubicación

---

## Autor

Proyecto de estadística multivariada para análisis de regresión con múltiples técnicas de machine learning.
