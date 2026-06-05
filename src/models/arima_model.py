"""
Entrenamiento y evaluación de ARIMA por categoría.
Réplica corregida del notebook 03_train_arima_sarima_olist.ipynb como módulo importable.

Mejoras respecto al notebook original:
  - PeriodIndex mensual correcto — elimina los warnings de statsmodels
  - Backtest con reentrenamiento en train+val (evita filtrar información del futuro)
  - Predicciones clippeadas a 0 (demanda no puede ser negativa)
  - SettingWithCopyWarning eliminado con .loc
  - Integración con MLflow
"""

import os
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.models.evaluate import compute_metrics

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = "olist_demand_forecasting"

TRAIN_END = "2017-12-01"
VAL_END   = "2018-05-01"

DEFAULT_ARIMA_ORDER = (1, 1, 1)


def _to_series(demand: pd.Series, year_month: pd.Series) -> pd.Series:
    """Construye la serie con PeriodIndex mensual que statsmodels reconoce."""
    idx = pd.PeriodIndex(year_month.dt.to_period("M"))
    return pd.Series(demand.values, index=idx, dtype=float, name="demand")


def _fit_arima(series: pd.Series, order: tuple) -> object:
    """Ajusta ARIMA silenciando warnings de convergencia."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(series, order=order).fit()


def _safe_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """compute_metrics con filtro previo de NaN."""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() == 0:
        return {"mape": np.nan, "rmse": np.nan, "mae": np.nan}
    return compute_metrics(actual[mask], predicted[mask])


def run_arima_per_category(
    df: pd.DataFrame,
    train_end: str        = TRAIN_END,
    val_end: str          = VAL_END,
    order: tuple          = DEFAULT_ARIMA_ORDER,
    min_train_months: int = 6,
    verbose: bool         = True,
) -> pd.DataFrame:
    """
    Entrena ARIMA(order) + baseline naive_lag_1 para cada categoría.
    Evalúa en validation y backtest usando demand_next_month como target.

    Splits (réplica exacta del notebook 03):
      train      : year_month <= train_end
      validation : train_end < year_month <= val_end
      backtest   : year_month > val_end  — modelo re-entrenado en train+val

    Args:
        df:               Base mensual con columnas year_month, product_category_name,
                          demand, demand_next_month
        train_end:        Último mes del set de entrenamiento (inclusive)
        val_end:          Último mes de validación (inclusive)
        order:            Orden ARIMA (p, d, q)
        min_train_months: Mínimo de meses en train para ajustar ARIMA

    Returns:
        DataFrame con columnas: dataset, model, category, mae, rmse, mape
    """
    results = []

    for cat, df_cat in df.groupby("product_category_name"):
        df_cat = df_cat.sort_values("year_month").copy()

        train    = df_cat[df_cat["year_month"] <= train_end]
        val      = df_cat[(df_cat["year_month"] > train_end) & (df_cat["year_month"] <= val_end)]
        backtest = df_cat[df_cat["year_month"] > val_end]

        if verbose:
            print(f"  {cat}: train={len(train)}, val={len(val)}, backtest={len(backtest)}")

        # --- Naive baseline: demand actual predice demand_next_month ---
        for split_name, split_df in [("validation", val), ("backtest", backtest)]:
            if len(split_df) == 0:
                continue
            m = _safe_metrics(
                split_df["demand_next_month"].values,
                split_df["demand"].values,
            )
            results.append({
                "dataset": split_name, "model": "naive_lag_1", "category": cat, **m,
            })

        # --- ARIMA ---
        if len(train) < min_train_months:
            continue

        try:
            train_series = _to_series(train["demand"], train["year_month"])
            model_val    = _fit_arima(train_series, order)

            if len(val) > 0:
                val_pred = np.maximum(model_val.forecast(steps=len(val)), 0)
                m = _safe_metrics(val["demand_next_month"].values, val_pred)
                results.append({
                    "dataset": "validation", "model": "ARIMA", "category": cat, **m,
                })

            if len(backtest) > 0:
                # Re-entrenar en train+val para no filtrar información del futuro
                tv_demand    = pd.concat([train["demand"],     val["demand"]],     ignore_index=True)
                tv_ym        = pd.concat([train["year_month"], val["year_month"]], ignore_index=True)
                tv_series    = _to_series(tv_demand, tv_ym)
                model_bt     = _fit_arima(tv_series, order)
                bt_pred      = np.maximum(model_bt.forecast(steps=len(backtest)), 0)
                m = _safe_metrics(backtest["demand_next_month"].values, bt_pred)
                results.append({
                    "dataset": "backtest", "model": "ARIMA", "category": cat, **m,
                })

        except Exception as exc:
            if verbose:
                print(f"    ⚠ ARIMA falló para {cat}: {exc}")

    return pd.DataFrame(results)


def run_arima_experiment(
    df: pd.DataFrame,
    train_end: str        = TRAIN_END,
    val_end: str          = VAL_END,
    arima_order: tuple    = DEFAULT_ARIMA_ORDER,
    run_name: str         = "arima_per_category",
) -> dict:
    """
    Ejecuta el experimento ARIMA completo y registra en MLflow:
      - Parámetros: arima_order, train_end, val_end, n_categories
      - Métricas MAPE/RMSE en validation y backtest (ARIMA vs Naive)
      - Artefacto: CSV de resultados por categoría
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_type":   "ARIMA",
            "arima_order":  str(arima_order),
            "train_end":    train_end,
            "val_end":      val_end,
            "n_categories": df["product_category_name"].nunique(),
        })

        results_df = run_arima_per_category(df, train_end, val_end, arima_order)

        for split in ("validation", "backtest"):
            for mdl in ("ARIMA", "naive_lag_1"):
                sub = results_df[
                    (results_df["dataset"] == split) & (results_df["model"] == mdl)
                ]
                if sub.empty:
                    continue
                prefix = f"{split}_{mdl.lower()}"
                mlflow.log_metrics({
                    f"{prefix}_mape": round(float(sub["mape"].mean(skipna=True)), 4),
                    f"{prefix}_rmse": round(float(sub["rmse"].mean(skipna=True)), 4),
                    f"{prefix}_mae":  round(float(sub["mae"].mean(skipna=True)),  4),
                })

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arima_results_by_category.csv"
            results_df.to_csv(path, index=False)
            mlflow.log_artifact(str(path))

    return {"results": results_df}
