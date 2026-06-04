"""
Funciones de evaluacion de modelos de forecasting.
Metricas: MAPE, RMSE, MAE.
Metodologia: walk-forward validation (sin data leakage).
"""

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Ignora filas donde actual == 0."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Retorna MAPE, RMSE y MAE en un diccionario."""
    return {
        "mape": round(mape(actual, predicted), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
        "mae":  round(float(mean_absolute_error(actual, predicted)), 4),
    }


def walk_forward_evaluate(
    df_model: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    train_end: str,
    test_start: str,
    test_end: str,
    model_fn: Callable,
) -> pd.DataFrame:
    """
    Evaluacion walk-forward: para cada mes de test entrena con todo
    lo disponible hasta el mes anterior y predice el mes actual.

    Simula el uso real del modelo en produccion — cada mes que llegan
    datos nuevos se reentrana y se emite una prediccion.

    Args:
        df_model:     Tabla de features (salida de build_features.get_feature_table)
        feature_cols: Lista de columnas de features
        target_col:   Nombre de la columna target
        train_end:    Ultimo mes de entrenamiento (inclusive), ej: '2018-02-28'
        test_start:   Primer mes de test, ej: '2018-03-01'
        test_end:     Ultimo mes de test, ej: '2018-08-31'
        model_fn:     Funcion que recibe (X_train, y_train) y retorna un modelo

    Returns:
        DataFrame con columnas: year_month, product_category_name,
        actual, predicted, predicted_naive
    """
    test_months = sorted(
        df_model.loc[
            (df_model["year_month"] >= test_start) &
            (df_model["year_month"] <= test_end),
            "year_month",
        ].unique()
    )

    results = []
    for month in test_months:
        train_wf = df_model[df_model["year_month"] < month]
        pred_rows = df_model[df_model["year_month"] == month]

        if len(train_wf) == 0 or len(pred_rows) == 0:
            continue

        model = model_fn(
            train_wf[feature_cols],
            train_wf[target_col],
        )
        preds = np.maximum(model.predict(pred_rows[feature_cols]), 0)

        for i, (_, row) in enumerate(pred_rows.iterrows()):
            results.append({
                "year_month":             month,
                "product_category_name":  row["product_category_name"],
                "actual":                 row[target_col],
                "predicted":              round(preds[i], 1),
                "predicted_naive":        row["lag_1"],
            })

    return pd.DataFrame(results)


def metrics_by_category(df_results: pd.DataFrame) -> pd.DataFrame:
    """Calcula MAPE del modelo y del Naive por cada categoria."""
    rows = []
    for cat, group in df_results.groupby("product_category_name"):
        m_model = mape(group["actual"].values, group["predicted"].values)
        m_naive = mape(group["actual"].values, group["predicted_naive"].values)
        rows.append({
            "category": cat,
            "mape_model": round(m_model, 2),
            "mape_naive": round(m_naive, 2),
            "improvement_pp": round(m_naive - m_model, 2),
        })
    return pd.DataFrame(rows).sort_values("mape_model").reset_index(drop=True)
