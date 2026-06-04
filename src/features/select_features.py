"""
Seleccion de features y categorias para el modelo de demand forecasting.

Tres tecnicas aplicadas en orden:
  1. Filtro de categorias por volatilidad (CV sobre el train set)
  2. Filtro de correlacion entre features (elimina redundancia)
  3. Filtro de importancia LightGBM (elimina features con aporte marginal)

Adicional: SHAP values para explicabilidad del modelo final.
"""

from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from src.features.build_features import TRAIN_END
from src.models.evaluate import mape


# ─────────────────────────────────────────────
# 1. FILTRO DE CATEGORIAS POR VOLATILIDAD
# ─────────────────────────────────────────────

def compute_category_stats(df_model: pd.DataFrame, train_end: str = TRAIN_END) -> pd.DataFrame:
    """
    Calcula CV y MAPE Naive por categoria usando solo el train set.
    Se usa para decidir que categorias incluir en el modelo.

    Args:
        df_model: Tabla de features completa
        train_end: Ultimo mes del train set

    Returns:
        DataFrame con category, mean_demand, cv_pct, mape_naive
    """
    train = df_model[df_model["year_month"] <= train_end].copy()

    rows = []
    for cat, group in train.groupby("product_category_name"):
        demand = group["demand"].values
        naive_pred = group["lag_1"].dropna().values
        actual_aligned = group.loc[group["lag_1"].notna(), "demand"].values

        cv = (demand.std() / demand.mean() * 100) if demand.mean() > 0 else 999
        naive_mape = mape(actual_aligned, naive_pred) if len(actual_aligned) > 0 else 999

        rows.append({
            "category":    cat,
            "n_months":    len(group),
            "mean_demand": round(demand.mean(), 1),
            "cv_pct":      round(cv, 1),
            "mape_naive":  round(naive_mape, 1),
        })

    return pd.DataFrame(rows).sort_values("cv_pct").reset_index(drop=True)


def filter_categories(
    df_model: pd.DataFrame,
    min_mean_demand: float = 50.0,
    min_months: int = 10,
    train_end: str = TRAIN_END,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list]:
    """
    Filtra categorias con volumen insuficiente para modelar de forma confiable.

    El CV no es un buen criterio para datos con tendencia creciente porque la
    tendencia infla artificialmente la dispersion. En su lugar se usan:
      - Demanda media >= min_mean_demand en el train set (volumen suficiente)
      - Al menos min_months meses con datos en el train set

    Args:
        df_model:         Tabla de features completa
        min_mean_demand:  Demanda media minima por mes para incluir la categoria
        min_months:       Minimo de meses requeridos en el train set
        train_end:        Ultimo mes del train set

    Returns:
        (df_filtrado, categorias_incluidas)
    """
    stats = compute_category_stats(df_model, train_end)

    included = stats[
        (stats["mean_demand"] >= min_mean_demand) &
        (stats["n_months"] >= min_months)
    ]["category"].tolist()

    excluded = stats[~stats["category"].isin(included)]["category"].tolist()

    if verbose:
        print(f"  Criterios: demanda media >= {min_mean_demand} ordenes/mes, >= {min_months} meses")
        print(f"  Categorias incluidas : {len(included)}")
        print(f"  Categorias excluidas : {len(excluded)}")
        if excluded:
            print(f"  Excluidas (muestra)  : {', '.join(excluded[:6])}{'...' if len(excluded) > 6 else ''}")
        if verbose and included:
            top = stats[stats["category"].isin(included)][["category", "mean_demand", "cv_pct", "mape_naive"]].head(5)
            print(f"\n  Top 5 categorias incluidas:")
            print(top.to_string(index=False))

    df_filtered = df_model[df_model["product_category_name"].isin(included)].copy()
    return df_filtered, included


# ─────────────────────────────────────────────
# 2. FILTRO DE CORRELACION
# ─────────────────────────────────────────────

def filter_correlated_features(
    df: pd.DataFrame,
    feature_cols: list,
    threshold: float = 0.95,
    verbose: bool = True,
) -> list:
    """
    Elimina features con correlacion de Pearson > threshold con otra feature
    de mayor importancia. Reduce redundancia entre lags y rolling averages.

    Estrategia: para cada par con correlacion > threshold, elimina la feature
    que aparece despues en la lista (asume que feature_cols esta ordenada
    por importancia descendente).

    Args:
        df:           DataFrame con las features
        feature_cols: Lista de features ordenada por importancia (mayor primero)
        threshold:    Umbral de correlacion para eliminar

    Returns:
        Lista de features seleccionadas (sin redundancias)
    """
    corr_matrix = df[feature_cols].corr().abs()
    selected = list(feature_cols)
    to_remove = set()

    for i, f1 in enumerate(feature_cols):
        if f1 in to_remove:
            continue
        for f2 in feature_cols[i + 1:]:
            if f2 in to_remove:
                continue
            if corr_matrix.loc[f1, f2] > threshold:
                to_remove.add(f2)

    selected = [f for f in feature_cols if f not in to_remove]

    if verbose:
        removed = [f for f in feature_cols if f in to_remove]
        print(f"  Features removidas por correlacion > {threshold}: {removed if removed else 'ninguna'}")
        print(f"  Features seleccionadas: {selected}")

    return selected


# ─────────────────────────────────────────────
# 3. FILTRO DE IMPORTANCIA
# ─────────────────────────────────────────────

def filter_by_importance(
    feat_imp: pd.DataFrame,
    feature_cols: list,
    min_pct: float = 3.0,
    verbose: bool = True,
) -> list:
    """
    Elimina features cuya importancia es menor a min_pct del total acumulado.

    Args:
        feat_imp:     DataFrame con columnas 'feature' e 'importance'
        feature_cols: Features actualmente en uso
        min_pct:      Porcentaje minimo de importancia relativa

    Returns:
        Lista de features seleccionadas
    """
    imp = feat_imp[feat_imp["feature"].isin(feature_cols)].copy()
    total = imp["importance"].sum()
    imp["pct"] = imp["importance"] / total * 100

    selected = imp[imp["pct"] >= min_pct]["feature"].tolist()
    removed  = imp[imp["pct"] <  min_pct]["feature"].tolist()

    if verbose:
        print(f"  Features removidas por importancia < {min_pct}% del total: "
              f"{removed if removed else 'ninguna'}")
        print(f"  Features seleccionadas: {selected}")

    return selected


# ─────────────────────────────────────────────
# 4. SHAP — EXPLICABILIDAD
# ─────────────────────────────────────────────

def compute_shap(
    model: lgb.LGBMRegressor,
    X: pd.DataFrame,
    feature_cols: list,
    max_samples: int = 200,
) -> pd.DataFrame:
    """
    Calcula SHAP values para el modelo entrenado.
    Retorna la importancia media absoluta de SHAP por feature.

    Args:
        model:        Modelo LightGBM entrenado
        X:            DataFrame de features (puede ser el train set)
        feature_cols: Lista de features usadas
        max_samples:  Limite de filas para calcular SHAP (velocidad)

    Returns:
        DataFrame con columnas feature, shap_importance, ordenado desc
    """
    X_sample = X[feature_cols].iloc[:max_samples]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap_imp = pd.DataFrame({
        "feature":          feature_cols,
        "shap_importance":  np.abs(shap_values).mean(axis=0),
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)

    return shap_imp


# ─────────────────────────────────────────────
# PIPELINE COMPLETO DE SELECCION
# ─────────────────────────────────────────────

def select_features_pipeline(
    df_model: pd.DataFrame,
    feature_cols: list,
    feat_imp: pd.DataFrame,
    model: lgb.LGBMRegressor,
    min_mean_demand: float = 50.0,
    min_months: int = 9,
    corr_threshold: float = 0.95,
    min_importance_pct: float = 3.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list, pd.DataFrame]:
    """
    Aplica los tres filtros en secuencia y retorna el dataset y features finales.

    Orden:
      1. Filtro de categorias por volumen minimo
      2. Filtro de correlacion (usando orden de importancia como criterio)
      3. Filtro de importancia minima

    Args:
        df_model:            Dataset completo
        feature_cols:        Features actuales
        feat_imp:            DataFrame con importancia de features
        model:               Modelo LightGBM (para SHAP)
        min_mean_demand:     Demanda media minima para incluir una categoria
        corr_threshold:      Umbral de correlacion para eliminar features redundantes
        min_importance_pct:  Importancia minima como % del total

    Returns:
        (df_filtrado, features_seleccionadas, shap_importance_df)
    """
    print("\n--- Filtro 1: Categorias por volumen ---")
    df_filtered, _ = filter_categories(df_model, min_mean_demand=min_mean_demand, min_months=min_months, verbose=verbose)

    # Ordena feature_cols por importancia descendente antes del filtro de correlacion
    feat_order = (
        feat_imp[feat_imp["feature"].isin(feature_cols)]
        .sort_values("importance", ascending=False)["feature"]
        .tolist()
    )

    print("\n--- Filtro 2: Correlacion entre features ---")
    features_after_corr = filter_correlated_features(
        df_filtered, feat_order, threshold=corr_threshold, verbose=verbose
    )

    print("\n--- Filtro 3: Importancia minima ---")
    features_final = filter_by_importance(
        feat_imp, features_after_corr, min_pct=min_importance_pct, verbose=verbose
    )

    print("\n--- SHAP: Importancia explicable ---")
    # Reentrenar con las features finales seleccionadas antes de calcular SHAP
    # (el modelo base fue entrenado con todas las features originales)
    train = df_filtered[df_filtered["year_month"] <= TRAIN_END]
    from src.models.train import train_model
    model_selected = train_model(train[features_final], train["demand"])
    shap_imp = compute_shap(model_selected, train, features_final)

    return df_filtered, features_final, shap_imp
