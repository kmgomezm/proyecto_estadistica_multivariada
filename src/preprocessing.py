import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import OrdinalEncoder


# Tratamiento de variables con datos faltantes y mapeo de categorías
class ManualImputer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # =========================
        # QUAL ORDINALS 
        # =========================
        qual_cols = [
            "ExterQual","ExterCond","BsmtQual","BsmtCond",
            "HeatingQC","KitchenQual","FireplaceQu",
            "GarageQual","GarageCond","PoolQC"
        ]
        
        for col in qual_cols:
            if col in X.columns:
                X[col] = X[col].fillna("NA")
        
        # =========================
        # NONE CATEGORIES (verdadero "no existe")
        # =========================
        none_cols = [
            "MiscFeature","Alley", "MasVnrType"
        ]
        
        for col in none_cols:
            if col in X.columns:
                X[col] = X[col].fillna("None")
        
        # =========================
        # BASEMENT 
        # =========================
        bsmt_cols = [
            "BsmtExposure","BsmtFinType1","BsmtFinType2"
        ]
        
        for col in bsmt_cols:
            if col in X.columns:
                X[col] = X[col].fillna("NA")
        
            
        # GARAGE
        # =========================
        X["HasGarage"] = X["GarageType"].notnull().astype(int)
        
        X["GarageYrBlt"] = X["GarageYrBlt"].fillna(X["YearBuilt"])
    
        # GarageFinish
        X["GarageFinish"] = X["GarageFinish"].fillna("NA")
        
        # =========================
        # MasVnrArea
        # =========================
        X["MasVnrArea"] = X["MasVnrArea"].fillna(0)
        
        # =========================
        # Electrical (1 missing)
        # =========================
        X["Electrical"] = X["Electrical"].fillna(X["Electrical"].mode()[0])
        
        # =========================
        # LotFrontage (por Neighborhood)
        # =========================
        if "Neighborhood" in X.columns:
            X["LotFrontage"] = X.groupby("Neighborhood")["LotFrontage"]\
                .transform(lambda x: x.fillna(x.median()))
        
        return X
    
class CustomFeatures(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X["TotalSF"] = (
            X["TotalBsmtSF"] +
            X["1stFlrSF"] +
            X["2ndFlrSF"]
        )
        
        return X
    


# ORDINAL CONFIGURATION
ordinal_cols = [
    # CALIDAD
    "ExterQual","ExterCond","BsmtQual","BsmtCond",
    "HeatingQC","KitchenQual","FireplaceQu",
    "GarageQual","GarageCond","PoolQC",
    
    # BASEMENT
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    
    # GARAGE
    "GarageFinish",
    
    # FUNCIONALIDAD
    "Functional",
    
    # EXTERIOR / LOTE
    "PavedDrive",
    "LandSlope",
    "LotShape"
]

ordinal_categories_map = {

    # CALIDAD
    "ExterQual": ["NA","Po","Fa","TA","Gd","Ex"],
    "ExterCond": ["NA","Po","Fa","TA","Gd","Ex"],
    "BsmtQual": ["NA","Po","Fa","TA","Gd","Ex"],
    "BsmtCond": ["NA","Po","Fa","TA","Gd","Ex"],
    "HeatingQC": ["NA","Po","Fa","TA","Gd","Ex"],
    "KitchenQual": ["NA","Po","Fa","TA","Gd","Ex"],
    "FireplaceQu": ["NA","Po","Fa","TA","Gd","Ex"],
    "GarageQual": ["NA","Po","Fa","TA","Gd","Ex"],
    "GarageCond": ["NA","Po","Fa","TA","Gd","Ex"],
    "PoolQC": ["NA","Fa","TA","Gd","Ex"],

    # BASEMENT
    "BsmtExposure": ["NA","No","Mn","Av","Gd"],
    "BsmtFinType1": ["NA","Unf","LwQ","Rec","BLQ","ALQ","GLQ"],
    "BsmtFinType2": ["NA","Unf","LwQ","Rec","BLQ","ALQ","GLQ"],

    # GARAGE
    "GarageFinish": ["NA","Unf","RFn","Fin"],

    # FUNCIONALIDAD
    "Functional": ["Sal","Sev","Maj2","Maj1","Mod","Min2","Min1","Typ"],

    # EXTERIOR
    "PavedDrive": ["N","P","Y"],
    "LandSlope": ["Gtl","Mod","Sev"],
    "LotShape": ["IR3","IR2","IR1","Reg"]
}


def build_preprocessor(X):
    
    # =========================
    # COLUMN SPLIT
    # =========================
    num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    
    ordinal_cols_used = [c for c in ordinal_cols if c in X.columns]
    nominal_cols = [c for c in cat_cols if c not in ordinal_cols_used]
    
    # =========================
    # NUMERICAL
    # =========================
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # =========================
    # NOMINAL
    # =========================
    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    # =========================
    # ORDINAL
    # =========================
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=[ordinal_categories_map[col] for col in ordinal_cols_used],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])
    
    return ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("nom", nominal_pipeline, nominal_cols),
        ("ord", ordinal_pipeline, ordinal_cols_used)
    ])

def build_pipeline(X):
    
    manual = ManualImputer()
    features = CustomFeatures()
    
    X_temp = manual.fit_transform(X)
    X_temp = features.fit_transform(X_temp)
    
    preprocessor = build_preprocessor(X_temp)
    
    pipeline = Pipeline([
        ("manual_rules", ManualImputer()),
        ("feature_engineering", CustomFeatures()),
        ("preprocessing", preprocessor)
    ])
    
    return pipeline