"""
Entrenamiento de LightGBM con tracking en MLflow.
El servidor MLflow corre en Docker (localhost:5000).
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from src.features.build_features import (
    FEATURE_COLS,
    TEST_END,
    TEST_START,
    TRAIN_END,
)
from src.models.evaluate import compute_metrics, metrics_by_category, walk_forward_evaluate

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "olist_demand_forecasting"
TARGET_COL = "demand"

DEFAULT_LGB_PARAMS = {
    "objective":          "regression",
    "metric":             "rmse",
    "n_estimators":       400,
    "learning_rate":      0.03,
    "num_leaves":         15,
    "min_child_samples":  3,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "reg_alpha":          0.1,
    "reg_lambda":         0.5,
    "random_state":       42,
    "verbose":            -1,
}


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[dict] = None,
) -> lgb.LGBMRegressor:
    """Entrena un LGBMRegressor y retorna el modelo ajustado."""
    p = params or DEFAULT_LGB_PARAMS
    model = lgb.LGBMRegressor(**p)
    model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(period=-1)])
    return model


def _model_fn(params: dict):
    """Closure que retorna una funcion compatible con walk_forward_evaluate."""
    def fn(X_train, y_train):
        return train_model(X_train, y_train, params)
    return fn


def run_experiment(
    df_model: pd.DataFrame,
    feature_cols: list = FEATURE_COLS,
    lgb_params: Optional[dict] = None,
    run_name: str = "lgbm_walk_forward",
    models_dir: Optional[Path] = None,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> dict:
    """
    Ejecuta el experimento completo y registra todo en MLflow:
      - Hiperparametros del modelo
      - Metricas globales: MAPE, RMSE, MAE (modelo y naive)
      - Metricas por categoria
      - Feature importance
      - Modelo serializado como artefacto

    Args:
        df_model:     Tabla de features lista para modelado
        feature_cols: Columnas de features a usar
        lgb_params:   Hiperparametros LightGBM (usa DEFAULT si es None)
        run_name:     Nombre del run en MLflow
        models_dir:   Directorio donde guardar el modelo .pkl

    Returns:
        Diccionario con metricas globales y df_results del walk-forward
    """
    params = lgb_params or DEFAULT_LGB_PARAMS
    if models_dir is None:
        models_dir = Path(__file__).resolve().parents[2] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params({
            "train_end":  train_end,
            "test_start": test_start,
            "test_end":   test_end,
        })

        # --- Walk-forward evaluation ---
        df_results = walk_forward_evaluate(
            df_model=df_model,
            feature_cols=feature_cols,
            target_col=TARGET_COL,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            model_fn=_model_fn(params),
        )

        # --- Metricas globales ---
        metrics = compute_metrics(
            df_results["actual"].values,
            df_results["predicted"].values,
        )
        naive_metrics = compute_metrics(
            df_results["actual"].values,
            df_results["predicted_naive"].values,
        )

        # --- Modelo final entrenado en todo el train set ---
        train = df_model[df_model["year_month"] <= train_end]
        model_final = train_model(train[feature_cols], train[TARGET_COL], params)

        # --- Feature importance ---
        feat_imp = pd.DataFrame({
            "feature":    feature_cols,
            "importance": model_final.feature_importances_,
        }).sort_values("importance", ascending=False)

        # --- Log en MLflow ---
        mlflow.log_params(params)
        mlflow.log_params({"n_features": len(feature_cols), "n_categories": df_model["product_category_name"].nunique()})

        mlflow.log_metrics({
            "mape":       metrics["mape"],
            "rmse":       metrics["rmse"],
            "mae":        metrics["mae"],
            "mape_naive": naive_metrics["mape"],
            "rmse_naive": naive_metrics["rmse"],
            "mae_naive":  naive_metrics["mae"],
            "mape_improvement_pp": round(naive_metrics["mape"] - metrics["mape"], 4),
        })

        # Log metricas por categoria como artefacto CSV
        cat_metrics = metrics_by_category(df_results)
        with tempfile.TemporaryDirectory() as tmp:
            cat_path = Path(tmp) / "metrics_by_category.csv"
            cat_metrics.to_csv(cat_path, index=False)
            mlflow.log_artifact(str(cat_path))

            imp_path = Path(tmp) / "feature_importance.csv"
            feat_imp.to_csv(imp_path, index=False)
            mlflow.log_artifact(str(imp_path))

            pred_path = Path(tmp) / "wf_predictions.csv"
            df_results.to_csv(pred_path, index=False)
            mlflow.log_artifact(str(pred_path))

        # Modelo serializado
        model_path = models_dir / "lgbm_demand_v2.pkl"
        joblib.dump(model_final, model_path)
        mlflow.log_artifact(str(model_path))

    return {
        "metrics":     metrics,
        "naive":       naive_metrics,
        "df_results":  df_results,
        "cat_metrics": cat_metrics,
        "feat_imp":    feat_imp,
        "model":       model_final,
    }
