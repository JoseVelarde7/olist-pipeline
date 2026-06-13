"""
Entrenamiento ARIMA + Naive con optimización Optuna por categoría.
Versión PRO integrada al pipeline Airflow.

Extraído de 03_train_arima_sarima_olist_optuna.ipynb del equipo.
Toda la lógica del notebook se respeta al 100%; los únicos cambios son:
  - Se elimina la lectura de CSV hardcodeada (CANDIDATE_INPUTS / find_input_path).
  - TRAIN_END, VAL_END y OUTPUT_DIR son configurables desde run_optuna_experiment().
  - Se agrega integración con MLflow.
  - Se usa el backend 'Agg' de matplotlib (sin display en Docker).
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

try:
    import optuna
except ImportError as e:
    raise ImportError(
        "Este módulo requiere 'optuna'. Instálala con: pip install optuna"
    ) from e

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================
TRAIN_END = pd.Timestamp("2017-12-01")
VAL_END   = pd.Timestamp("2018-05-01")

N_TRIALS_NAIVE = 20
N_TRIALS_ARIMA = 40
OPTUNA_SEED    = 42

ARIMA_P_MIN, ARIMA_P_MAX = 0, 2
ARIMA_D_MIN, ARIMA_D_MAX = 0, 1
ARIMA_Q_MIN, ARIMA_Q_MAX = 0, 2
MIN_ARIMA_HISTORY         = 8

NAIVE_CANDIDATES = [
    "naive_lag_1",
    "naive_ma_2",
    "naive_wma_3",
    "naive_ma_3",
    "naive_lag_2",
]

ARIMA_MAX_REASONABLE_MAE  = 10000
ARIMA_MAX_ABS_PRED        = 1_000_000
ARIMA_TOP_MODELS_FOR_PLOT = 12
STABILITY_PENALTY_WEIGHT  = 0.10
CONSISTENCY_PENALTY_WEIGHT = 0

USE_LOG_TRANSFORM  = False
MIN_POSITIVE_CLIP  = 0.0

OUTPUT_DIR = Path("outputs_naive_arima_optuna_pro")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = "olist_demand_forecasting"

# ============================================================
# 2. UTILIDADES GENERALES
# ============================================================
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    return {
        "mae":  float(mae),
        "rmse": float(rmse),
        "mape": float(mape) if not pd.isna(mape) else np.nan,
    }


def is_valid_metric_dict(metrics):
    if metrics is None:
        return False
    values = [metrics.get("mae"), metrics.get("rmse"), metrics.get("mape")]
    for v in values:
        if pd.isna(v):
            continue
        if not np.isfinite(v):
            return False
    if metrics["mae"] > ARIMA_MAX_REASONABLE_MAE:
        return False
    return True


def are_valid_predictions(preds):
    preds = np.array(preds, dtype=float)
    if len(preds) == 0:
        return False
    if not np.isfinite(preds).all():
        return False
    if np.max(np.abs(preds)) > ARIMA_MAX_ABS_PRED:
        return False
    return True


def clean_arima_search_df(search_df):
    if search_df.empty:
        return search_df
    clean = search_df.copy()
    clean = clean[np.isfinite(clean["mae"])]
    clean = clean[np.isfinite(clean["rmse"])]
    clean = clean[clean["mae"] <= ARIMA_MAX_REASONABLE_MAE]
    return clean.reset_index(drop=True)


def safe_improvement(base_mae, candidate_mae):
    if pd.isna(base_mae) or base_mae == 0:
        return np.nan, np.nan
    delta = base_mae - candidate_mae
    pct   = (delta / base_mae) * 100
    return float(delta), float(pct)


def summarize_metrics(df_metrics, group_cols):
    if df_metrics.empty:
        return pd.DataFrame()
    return (
        df_metrics
        .groupby(group_cols, dropna=False)[["mae", "rmse", "mape"]]
        .mean()
        .reset_index()
        .sort_values(group_cols)
    )


def stability_penalty(preds):
    preds = np.array(preds, dtype=float)
    if len(preds) <= 1:
        return 0.0
    return float(np.std(preds))


def consistency_penalty(mae_val, mae_back):
    if pd.isna(mae_val) or pd.isna(mae_back):
        return 0.0
    return float(abs(mae_val - mae_back))


def selection_score(mae_val, mae_back, w_val=0.6, w_back=0.3, w_consistency=CONSISTENCY_PENALTY_WEIGHT):
    if pd.isna(mae_val) and pd.isna(mae_back):
        return np.inf
    if pd.isna(mae_back):
        return mae_val
    if pd.isna(mae_val):
        return mae_back
    return float(w_val * mae_val + w_back * mae_back + w_consistency * abs(mae_val - mae_back))


def transform_series_for_model(series, use_log=USE_LOG_TRANSFORM):
    series  = pd.Series(series).astype(float)
    if not use_log:
        return series
    clipped = series.clip(lower=MIN_POSITIVE_CLIP)
    return np.log1p(clipped)


def inverse_transform_predictions(preds, use_log=USE_LOG_TRANSFORM):
    preds = np.array(preds, dtype=float)
    if not use_log:
        return preds
    return np.expm1(preds)


# ============================================================
# 3. PREPARACIÓN DE DATOS Y CANDIDATOS NAIVE
# ============================================================
def prepare_category_frame(df_cat):
    df_cat = df_cat.copy().sort_values("year_month").reset_index(drop=True)

    if "demand_next_month" not in df_cat.columns:
        df_cat["demand_next_month"] = df_cat["demand"].shift(-1)

    df_cat["naive_lag_1"]      = df_cat["demand"]
    df_cat["naive_lag_2"]      = df_cat["demand"].shift(1)
    df_cat["naive_lag_3"]      = df_cat["demand"].shift(2)
    df_cat["naive_ma_2"]       = df_cat["demand"].rolling(2).mean()
    df_cat["naive_ma_3"]       = df_cat["demand"].rolling(3).mean()
    df_cat["naive_ma_6"]       = df_cat["demand"].rolling(6).mean()

    def weighted_ma(values):
        weights = np.arange(1, len(values) + 1)
        return np.average(values, weights=weights)

    df_cat["naive_wma_3"]      = df_cat["demand"].rolling(3).apply(weighted_ma, raw=True)
    df_cat["naive_seasonal_12"] = df_cat["demand"].shift(11)
    return df_cat


def split_category(df_cat):
    train    = df_cat[df_cat["year_month"] <= TRAIN_END].copy()
    val      = df_cat[(df_cat["year_month"] > TRAIN_END) & (df_cat["year_month"] <= VAL_END)].copy()
    backtest = df_cat[df_cat["year_month"] > VAL_END].copy()
    return train, val, backtest


def evaluate_naive_subset(df_subset, dataset_name, category, candidate_list=None):
    candidate_list = candidate_list or NAIVE_CANDIDATES
    results      = []
    pred_records = []

    for col in candidate_list:
        if col not in df_subset.columns:
            continue
        valid = df_subset[["year_month", "demand_next_month", col]].dropna().copy()
        if valid.empty:
            continue

        metrics = compute_metrics(valid["demand_next_month"], valid[col])
        results.append({
            "dataset":      dataset_name,
            "model_family": "naive",
            "model":        col,
            "category":     category,
            **metrics,
        })

        valid["dataset"]      = dataset_name
        valid["model_family"] = "naive"
        valid["model"]        = col
        valid["category"]     = category
        valid = valid.rename(columns={col: "prediction", "demand_next_month": "actual"})
        pred_records.append(
            valid[["year_month", "dataset", "model_family", "model", "category", "actual", "prediction"]]
        )

    results_df = pd.DataFrame(results)
    pred_df    = pd.concat(pred_records, ignore_index=True) if pred_records else pd.DataFrame()
    return results_df, pred_df


def build_naive_contribution_table(naive_metrics_cat):
    if naive_metrics_cat.empty:
        return pd.DataFrame()

    base = (
        naive_metrics_cat[naive_metrics_cat["model"] == "naive_lag_1"][["dataset", "category", "mae"]]
        .rename(columns={"mae": "base_mae"})
    )
    contrib = naive_metrics_cat.merge(base, on=["dataset", "category"], how="left")

    deltas = contrib.apply(lambda row: safe_improvement(row["base_mae"], row["mae"]), axis=1)
    contrib["delta_mae_vs_baseline"]         = [x[0] for x in deltas]
    contrib["improvement_pct_vs_baseline"]   = [x[1] for x in deltas]
    contrib["positive_improvement"]          = contrib["delta_mae_vs_baseline"].clip(lower=0)
    total_pos = contrib.groupby(["dataset", "category"], dropna=False)["positive_improvement"].transform("sum")
    contrib["proxy_importance_share_pct"]    = np.where(
        total_pos > 0, contrib["positive_improvement"] / total_pos * 100, 0.0
    )
    return contrib.sort_values(["dataset", "category", "mae"]).reset_index(drop=True)


def optimize_naive_with_optuna(val_df, category):
    val_candidates = evaluate_naive_subset(val_df, "validation", category, candidate_list=NAIVE_CANDIDATES)[0]
    if val_candidates.empty:
        return None, pd.DataFrame(), pd.DataFrame()

    metrics_map = val_candidates.set_index("model").to_dict(orient="index")

    def objective(trial):
        rule = trial.suggest_categorical("naive_rule", list(metrics_map.keys()))
        return metrics_map[rule]["mae"]

    sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED)
    study   = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=min(N_TRIALS_NAIVE, len(metrics_map)), show_progress_bar=False)

    trials_df = study.trials_dataframe()
    if not trials_df.empty:
        trials_df["category"]      = category
        trials_df["model_family"]  = "naive"
        trials_df["selected_rule"] = trials_df.get("params_naive_rule", np.nan)
        trials_df = trials_df.rename(columns={"value": "mae"}).sort_values("mae").reset_index(drop=True)

    best_rule = study.best_params["naive_rule"]
    return best_rule, val_candidates, trials_df


# ============================================================
# 4. ARIMA: FORECAST, OPTUNA E INTERPRETABILIDAD PROXY
# ============================================================
def rolling_arima_forecast(train_series, test_target_series, order, use_log=USE_LOG_TRANSFORM):
    history      = transform_series_for_model(
        pd.Series(train_series).dropna().astype(float), use_log=use_log
    ).tolist()
    y_test_original = pd.Series(test_target_series).dropna().astype(float).values

    if len(history) < MIN_ARIMA_HISTORY or len(y_test_original) == 0:
        return np.array([])

    preds_original = []

    for actual_original in y_test_original:
        try:
            model  = ARIMA(history, order=order, enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit()
            pred_transformed = float(fitted.forecast(steps=1)[0])
            pred_original    = float(inverse_transform_predictions([pred_transformed], use_log=use_log)[0])
            preds_original.append(pred_original)

            actual_transformed = float(
                transform_series_for_model([actual_original], use_log=use_log).iloc[0]
            )
            history.append(actual_transformed)
        except Exception:
            return np.array([])

    return np.array(preds_original)


def evaluate_arima_order(train_series, test_target_series, order, use_log=USE_LOG_TRANSFORM):
    preds  = rolling_arima_forecast(train_series, test_target_series, order, use_log=use_log)
    y_true = pd.Series(test_target_series).dropna().astype(float).values

    if len(preds) == 0 or len(preds) != len(y_true):
        return None, None
    if not are_valid_predictions(preds):
        return None, None

    metrics = compute_metrics(y_true, preds)
    if not is_valid_metric_dict(metrics):
        return None, None

    return metrics, preds


def optimize_arima_with_optuna(train_series, val_target_series, category):
    trial_rows = []

    def objective(trial):
        p     = trial.suggest_int("p", ARIMA_P_MIN, ARIMA_P_MAX)
        d     = trial.suggest_int("d", ARIMA_D_MIN, ARIMA_D_MAX)
        q     = trial.suggest_int("q", ARIMA_Q_MIN, ARIMA_Q_MAX)
        order = (p, d, q)

        metrics, preds = evaluate_arima_order(train_series, val_target_series, order, use_log=USE_LOG_TRANSFORM)
        if metrics is None:
            value = ARIMA_MAX_REASONABLE_MAE + 1
            trial.set_user_attr("rmse",               np.nan)
            trial.set_user_attr("mape",               np.nan)
            trial.set_user_attr("stability_penalty",  np.nan)
            trial.set_user_attr("valid_model",        False)
            trial_rows.append({
                "category": category, "order": str(order),
                "p": p, "d": d, "q": q,
                "mae": value, "rmse": np.nan, "mape": np.nan,
                "stability_penalty": np.nan, "objective_value": value, "valid_model": False,
            })
            return value

        penalty         = stability_penalty(preds)
        objective_value = metrics["mae"] + STABILITY_PENALTY_WEIGHT * penalty

        trial.set_user_attr("rmse",              metrics["rmse"])
        trial.set_user_attr("mape",              metrics["mape"])
        trial.set_user_attr("stability_penalty", penalty)
        trial.set_user_attr("valid_model",       True)

        trial_rows.append({
            "category": category, "order": str(order),
            "p": p, "d": d, "q": q,
            **metrics,
            "stability_penalty": penalty,
            "objective_value":   objective_value,
            "valid_model":       True,
        })
        return objective_value

    sampler = optuna.samplers.TPESampler(seed=OPTUNA_SEED)
    study   = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS_ARIMA, show_progress_bar=False)

    search_df = pd.DataFrame(trial_rows)
    if search_df.empty:
        return None, None, pd.DataFrame(), pd.DataFrame()

    search_df = search_df[search_df["valid_model"] == True].copy()
    search_df = clean_arima_search_df(search_df)
    search_df = search_df.sort_values(["objective_value", "mae", "rmse"]).reset_index(drop=True)

    trials_df = study.trials_dataframe()
    if not trials_df.empty:
        trials_df["category"]     = category
        trials_df["model_family"] = "arima"
        trials_df = trials_df.rename(columns={"value": "objective_value"}).sort_values("objective_value").reset_index(drop=True)

    if search_df.empty:
        return None, None, pd.DataFrame(), trials_df

    best_row     = search_df.iloc[0]
    best_order   = (int(best_row["p"]), int(best_row["d"]), int(best_row["q"]))
    best_metrics, best_preds = evaluate_arima_order(
        train_series, val_target_series, best_order, use_log=USE_LOG_TRANSFORM
    )

    return best_order, best_preds, search_df, trials_df


def fit_final_arima_model(series_all_history, order, use_log=USE_LOG_TRANSFORM):
    transformed = transform_series_for_model(series_all_history.astype(float), use_log=use_log)
    model  = ARIMA(transformed.values, order=order, enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit()
    return fitted


def extract_arima_parameter_importance(fitted_model, category, dataset_scope, order):
    values = list(np.array(fitted_model.params, dtype=float))
    if hasattr(fitted_model, "param_names"):
        names = list(fitted_model.param_names)
    elif hasattr(fitted_model, "model") and hasattr(fitted_model.model, "param_names"):
        names = list(fitted_model.model.param_names)
    else:
        names = []

    if len(names) != len(values):
        p, d, q = order
        names   = []
        if len(values) >= (p + q + 2):
            names.append("const")
        for i in range(p):
            names.append(f"ar.L{i+1}")
        for i in range(q):
            names.append(f"ma.L{i+1}")
        while len(names) < len(values) - 1:
            names.append(f"extra_param_{len(names)}")
        if len(names) < len(values):
            names.append("sigma2")
        if len(names) != len(values):
            names = [f"param_{i}" for i in range(len(values))]

    rows       = []
    abs_values = np.abs(values)
    abs_sum    = abs_values.sum() if len(abs_values) > 0 else 0
    for name, value, abs_value in zip(names, values, abs_values):
        name_str       = str(name).lower()
        component_type = "other"
        if name_str.startswith("ar.") or name_str.startswith("ar.l"):
            component_type = "autoregressive"
        elif name_str.startswith("ma.") or name_str.startswith("ma.l"):
            component_type = "moving_average"
        elif "sigma" in name_str:
            component_type = "variance"
        elif "const" in name_str or "intercept" in name_str:
            component_type = "constant"

        share = (abs_value / abs_sum * 100) if abs_sum > 0 else np.nan
        rows.append({
            "category":                  category,
            "dataset_scope":             dataset_scope,
            "order":                     str(order),
            "term":                      str(name),
            "component_type":            component_type,
            "coefficient":               float(value),
            "abs_coefficient":           float(abs_value),
            "proxy_importance_share_pct": float(share) if not pd.isna(share) else np.nan,
        })
    return pd.DataFrame(rows)


def build_arima_order_effect_table(search_df):
    if search_df.empty:
        return pd.DataFrame()
    clean_df = clean_arima_search_df(search_df)
    if clean_df.empty:
        return pd.DataFrame()

    summary_parts = []
    for param in ["p", "d", "q"]:
        temp = (
            clean_df.groupby(["category", param], dropna=False)[["mae", "rmse", "mape", "objective_value"]]
            .mean()
            .reset_index()
            .rename(columns={param: "value"})
        )
        temp["hyperparameter"] = param
        summary_parts.append(
            temp[["category", "hyperparameter", "value", "mae", "rmse", "mape", "objective_value"]]
        )
    return pd.concat(summary_parts, ignore_index=True)


# ============================================================
# 5. VISUALIZACIONES
# ============================================================
def plot_model_comparison(base_summary, optimized_summary):
    if base_summary.empty and optimized_summary.empty:
        return

    opt = optimized_summary.copy()
    if not opt.empty:
        opt = opt[np.isfinite(opt["mae"])]
        opt = opt[opt["mae"] <= ARIMA_MAX_REASONABLE_MAE]

    combined = pd.concat([base_summary, opt], ignore_index=True)

    for dataset_name in combined["dataset"].dropna().unique():
        plot_df = combined[combined["dataset"] == dataset_name].copy()
        if plot_df.empty:
            continue
        plot_df = plot_df.sort_values("mae").head(ARIMA_TOP_MODELS_FOR_PLOT)
        labels  = plot_df["model_family"] + " - " + plot_df["model"]

        plt.figure(figsize=(12, 5))
        plt.bar(labels, plot_df["mae"])
        plt.title(f"Comparación de MAE por modelo - {dataset_name} (filtrado)")
        plt.xlabel("Modelo")
        plt.ylabel("MAE")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"comparacion_mae_{dataset_name}.png", dpi=150)
        plt.close()


def plot_selected_real_vs_pred(selected_predictions, dataset_name):
    plot_df = selected_predictions[selected_predictions["dataset"] == dataset_name].copy()
    if plot_df.empty:
        return
    agg = plot_df.groupby("year_month")[["actual", "prediction"]].sum().reset_index()

    plt.figure(figsize=(12, 5))
    plt.plot(agg["year_month"], agg["actual"],     marker="o", label="Serie real")
    plt.plot(agg["year_month"], agg["prediction"], marker="o", label="Serie estimada")
    plt.title(f"Serie real vs serie estimada - modelos seleccionados ({dataset_name})")
    plt.xlabel("Fecha")
    plt.ylabel("Demanda agregada")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"real_vs_estimado_{dataset_name}.png", dpi=150)
    plt.close()


def plot_naive_proxy_importance(naive_contrib):
    if naive_contrib.empty:
        return
    agg = (
        naive_contrib.groupby("model", dropna=False)["proxy_importance_share_pct"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    plt.figure(figsize=(10, 5))
    plt.bar(agg["model"], agg["proxy_importance_share_pct"])
    plt.title("Importancia proxy de variables naive (promedio)")
    plt.xlabel("Variable / regla naive")
    plt.ylabel("Aporte proxy al desempeño (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "naive_proxy_importance.png", dpi=150)
    plt.close()


def plot_arima_proxy_importance(arima_param_importance):
    if arima_param_importance.empty:
        return
    agg = (
        arima_param_importance.groupby("term", dropna=False)["proxy_importance_share_pct"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    plt.figure(figsize=(10, 5))
    plt.bar(agg["term"], agg["proxy_importance_share_pct"])
    plt.title("Importancia proxy de términos ARIMA (promedio)")
    plt.xlabel("Término del modelo")
    plt.ylabel("Aporte proxy (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "arima_proxy_importance.png", dpi=150)
    plt.close()


def plot_arima_hyperparameter_effect(arima_order_effect):
    if arima_order_effect.empty:
        return
    clean = arima_order_effect.copy()
    clean = clean[np.isfinite(clean["mae"])]
    clean = clean[clean["mae"] <= ARIMA_MAX_REASONABLE_MAE]
    for hp in clean["hyperparameter"].unique():
        temp = (
            clean[clean["hyperparameter"] == hp]
            .groupby("value", dropna=False)["mae"]
            .mean()
            .reset_index()
            .sort_values("value")
        )
        plt.figure(figsize=(8, 4))
        plt.plot(temp["value"], temp["mae"], marker="o")
        plt.title(f"Efecto promedio de {hp} sobre el MAE en ARIMA (filtrado)")
        plt.xlabel(hp)
        plt.ylabel("MAE promedio")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"arima_efecto_{hp}_mae.png", dpi=150)
        plt.close()


# ============================================================
# 6. PIPELINE PRINCIPAL POR CATEGORÍA
# ============================================================
def run_category_pipeline(df_cat):
    category = (
        df_cat["product_category_name"].iloc[0]
        if "product_category_name" in df_cat.columns
        else "overall"
    )
    df_cat = prepare_category_frame(df_cat)
    train, val, backtest = split_category(df_cat)

    result = {
        "category":              category,
        "naive_metrics":         pd.DataFrame(),
        "naive_preds":           pd.DataFrame(),
        "naive_contrib":         pd.DataFrame(),
        "naive_trials":          pd.DataFrame(),
        "arima_base_metrics":    pd.DataFrame(),
        "arima_optuna_trials":   pd.DataFrame(),
        "arima_search":          pd.DataFrame(),
        "arima_best_metrics":    pd.DataFrame(),
        "arima_preds":           pd.DataFrame(),
        "arima_param_importance": pd.DataFrame(),
        "final_selection":       pd.DataFrame(),
        "final_models":          {},
    }

    # ---------- NAIVE ----------
    naive_val_metrics,  naive_val_preds  = evaluate_naive_subset(val,      "validation", category, candidate_list=NAIVE_CANDIDATES)
    naive_back_metrics, naive_back_preds = evaluate_naive_subset(backtest,  "backtest",  category, candidate_list=NAIVE_CANDIDATES)
    naive_metrics = pd.concat([naive_val_metrics, naive_back_metrics], ignore_index=True)
    naive_preds   = (
        pd.concat([naive_val_preds, naive_back_preds], ignore_index=True)
        if (not naive_val_preds.empty or not naive_back_preds.empty)
        else pd.DataFrame()
    )
    naive_contrib    = build_naive_contribution_table(naive_metrics)
    best_naive_model, _, naive_trials = optimize_naive_with_optuna(val, category)

    result["naive_metrics"] = naive_metrics
    result["naive_preds"]   = naive_preds
    result["naive_contrib"] = naive_contrib
    result["naive_trials"]  = naive_trials if naive_trials is not None else pd.DataFrame()

    # ---------- ARIMA BASE ----------
    arima_base_rows = []
    arima_preds_rows = []

    base_val_metrics, base_val_preds = evaluate_arima_order(
        train["demand"], val["demand_next_month"], (1, 1, 1), use_log=USE_LOG_TRANSFORM
    )
    if base_val_metrics is not None:
        arima_base_rows.append({
            "dataset": "validation", "model_family": "arima",
            "model": "ARIMA(1,1,1)", "category": category, **base_val_metrics,
        })
        temp = val[["year_month", "demand_next_month"]].dropna().copy()
        temp["dataset"] = "validation"; temp["model_family"] = "arima"
        temp["model"] = "ARIMA(1,1,1)"; temp["category"] = category
        temp["prediction"] = base_val_preds
        temp = temp.rename(columns={"demand_next_month": "actual"})
        arima_preds_rows.append(
            temp[["year_month", "dataset", "model_family", "model", "category", "actual", "prediction"]]
        )

    base_back_metrics, base_back_preds = evaluate_arima_order(
        pd.concat([train["demand"], val["demand"]]),
        backtest["demand_next_month"], (1, 1, 1), use_log=USE_LOG_TRANSFORM,
    )
    if base_back_metrics is not None:
        arima_base_rows.append({
            "dataset": "backtest", "model_family": "arima",
            "model": "ARIMA(1,1,1)", "category": category, **base_back_metrics,
        })
        temp = backtest[["year_month", "demand_next_month"]].dropna().copy()
        temp["dataset"] = "backtest"; temp["model_family"] = "arima"
        temp["model"] = "ARIMA(1,1,1)"; temp["category"] = category
        temp["prediction"] = base_back_preds
        temp = temp.rename(columns={"demand_next_month": "actual"})
        arima_preds_rows.append(
            temp[["year_month", "dataset", "model_family", "model", "category", "actual", "prediction"]]
        )

    result["arima_base_metrics"] = pd.DataFrame(arima_base_rows)

    # ---------- ARIMA OPTUNA PRO ----------
    best_order, best_val_preds, arima_search, arima_optuna_trials = optimize_arima_with_optuna(
        train["demand"], val["demand_next_month"], category
    )
    result["arima_optuna_trials"] = arima_optuna_trials if arima_optuna_trials is not None else pd.DataFrame()
    result["arima_search"]        = arima_search

    best_arima_rows        = []
    arima_param_importance = pd.DataFrame()

    if best_order is not None and not arima_search.empty:
        best_val_metrics = arima_search.iloc[0][["mae", "rmse", "mape", "stability_penalty", "objective_value"]].to_dict()
        best_arima_rows.append({
            "dataset": "validation", "model_family": "arima",
            "model": f"ARIMA{best_order}", "category": category, **best_val_metrics,
        })

        temp_val = val[["year_month", "demand_next_month"]].dropna().copy()
        temp_val["dataset"] = "validation"; temp_val["model_family"] = "arima"
        temp_val["model"] = f"ARIMA{best_order}"; temp_val["category"] = category
        temp_val["prediction"] = best_val_preds
        temp_val = temp_val.rename(columns={"demand_next_month": "actual"})
        arima_preds_rows.append(
            temp_val[["year_month", "dataset", "model_family", "model", "category", "actual", "prediction"]]
        )

        back_metrics, back_preds = evaluate_arima_order(
            pd.concat([train["demand"], val["demand"]]),
            backtest["demand_next_month"], best_order, use_log=USE_LOG_TRANSFORM,
        )
        if back_metrics is not None:
            best_arima_rows.append({
                "dataset": "backtest", "model_family": "arima",
                "model": f"ARIMA{best_order}", "category": category,
                **back_metrics,
                "stability_penalty": stability_penalty(back_preds),
                "objective_value":   back_metrics["mae"] + STABILITY_PENALTY_WEIGHT * stability_penalty(back_preds),
            })
            temp_back = backtest[["year_month", "demand_next_month"]].dropna().copy()
            temp_back["dataset"] = "backtest"; temp_back["model_family"] = "arima"
            temp_back["model"] = f"ARIMA{best_order}"; temp_back["category"] = category
            temp_back["prediction"] = back_preds
            temp_back = temp_back.rename(columns={"demand_next_month": "actual"})
            arima_preds_rows.append(
                temp_back[["year_month", "dataset", "model_family", "model", "category", "actual", "prediction"]]
            )

        full_history = df_cat["demand"].dropna()
        try:
            fitted_final           = fit_final_arima_model(full_history, best_order, use_log=USE_LOG_TRANSFORM)
            arima_param_importance = extract_arima_parameter_importance(
                fitted_final, category=category, dataset_scope="full_history", order=best_order
            )
            result["final_models"][category] = {
                "family":            "arima_candidate",
                "order":             best_order,
                "fitted_model":      fitted_final,
                "use_log_transform": USE_LOG_TRANSFORM,
            }
        except Exception:
            pass

    result["arima_best_metrics"]     = pd.DataFrame(best_arima_rows)
    result["arima_preds"]            = pd.concat(arima_preds_rows, ignore_index=True) if arima_preds_rows else pd.DataFrame()
    result["arima_param_importance"] = arima_param_importance

    # ---------- SELECCIÓN FINAL POR CATEGORÍA ----------
    final_rows  = []
    final_models = result["final_models"]

    naive_val_best_row  = None
    naive_back_best_row = None
    arima_val_best_row  = None
    arima_back_best_row = None

    if best_naive_model is not None and not naive_metrics.empty:
        tmp = naive_metrics[(naive_metrics["dataset"] == "validation") & (naive_metrics["model"] == best_naive_model)]
        if not tmp.empty:
            naive_val_best_row = tmp.iloc[0].to_dict()
        tmp = naive_metrics[(naive_metrics["dataset"] == "backtest") & (naive_metrics["model"] == best_naive_model)]
        if not tmp.empty:
            naive_back_best_row = tmp.iloc[0].to_dict()

    if not result["arima_best_metrics"].empty:
        tmp = result["arima_best_metrics"][result["arima_best_metrics"]["dataset"] == "validation"]
        if not tmp.empty:
            arima_val_best_row = tmp.iloc[0].to_dict()
        tmp = result["arima_best_metrics"][result["arima_best_metrics"]["dataset"] == "backtest"]
        if not tmp.empty:
            arima_back_best_row = tmp.iloc[0].to_dict()

    naive_score = selection_score(
        naive_val_best_row["mae"]  if naive_val_best_row  else np.nan,
        naive_back_best_row["mae"] if naive_back_best_row else np.nan,
    )
    arima_score = selection_score(
        arima_val_best_row["mae"]  if arima_val_best_row  else np.nan,
        arima_back_best_row["mae"] if arima_back_best_row else np.nan,
    )

    selected_source = None
    selected_model  = None

    if naive_score <= arima_score:
        selected_source = "naive"
        selected_model  = best_naive_model
    elif np.isfinite(arima_score):
        selected_source = "arima"
        selected_model  = arima_val_best_row["model"] if arima_val_best_row else None

    if selected_source == "naive" and selected_model is not None:
        model_rows = naive_metrics[naive_metrics["model"] == selected_model].copy()
        if not model_rows.empty:
            model_rows["selected"]            = True
            model_rows["selection_score"]     = naive_score
            model_rows["consistency_penalty"] = consistency_penalty(
                naive_val_best_row["mae"]  if naive_val_best_row  else np.nan,
                naive_back_best_row["mae"] if naive_back_best_row else np.nan,
            )
            final_rows.append(model_rows)
        final_models[category] = {
            "family":          "naive",
            "rule":            selected_model,
            "description":     "Modelo naive seleccionado con Optuna + consistency score",
            "selection_score": naive_score,
        }

    elif selected_source == "arima" and selected_model is not None:
        model_rows = result["arima_best_metrics"][result["arima_best_metrics"]["model"] == selected_model].copy()
        if not model_rows.empty:
            model_rows["selected"]            = True
            model_rows["selection_score"]     = arima_score
            model_rows["consistency_penalty"] = consistency_penalty(
                arima_val_best_row["mae"]  if arima_val_best_row  else np.nan,
                arima_back_best_row["mae"] if arima_back_best_row else np.nan,
            )
            final_rows.append(model_rows)
        if category in final_models:
            final_models[category]["selected"]        = True
            final_models[category]["selection_score"] = arima_score
        else:
            final_models[category] = {
                "family":          "arima",
                "order":           selected_model.replace("ARIMA", ""),
                "description":     "Modelo ARIMA seleccionado con Optuna + consistency score",
                "selection_score": arima_score,
            }

    result["final_selection"] = pd.concat(final_rows, ignore_index=True) if final_rows else pd.DataFrame()
    result["final_models"]    = final_models
    return result


# ============================================================
# 7. ENTRY POINT PARA EL DAG DE AIRFLOW
# ============================================================
def run_optuna_experiment(
    df: pd.DataFrame,
    output_dir,
    models_dir,
    train_end_str: str  = None,
    val_end_str:   str  = None,
    categories:    list = None,
) -> dict:
    """
    Punto de entrada para el DAG de Airflow.

    Acepta el DataFrame mensual directamente (sin leer CSV).
    Reproduce exactamente la lógica del notebook main() del equipo.

    Args:
        df:            Base mensual con columnas year_month, product_category_name,
                       demand, demand_next_month
        output_dir:    Directorio donde se guardan CSVs, gráficos y reporte markdown
        models_dir:    Directorio donde se guarda el .pkl del modelo final
        train_end_str: Override del split de entrenamiento (formato "YYYY-MM-DD")
        val_end_str:   Override del split de validación   (formato "YYYY-MM-DD")

    Returns:
        Dict con métricas y rutas de artefactos, compatible con save_report del DAG.
    """
    global TRAIN_END, VAL_END, OUTPUT_DIR

    if train_end_str:
        TRAIN_END = pd.Timestamp(train_end_str)
    if val_end_str:
        VAL_END = pd.Timestamp(val_end_str)

    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = df.copy()
    df["year_month"] = pd.to_datetime(df["year_month"])

    if "product_category_name" not in df.columns:
        df["product_category_name"] = "overall"

    df = df.sort_values(["product_category_name", "year_month"]).reset_index(drop=True)

    if categories is not None:
        df = df[df["product_category_name"].isin(categories)].copy()

    categories = list(df["product_category_name"].dropna().unique())

    print(f"Dataset shape: {df.shape}")
    print(f"Categorías a procesar: {len(categories)}")
    print(f"USE_LOG_TRANSFORM = {USE_LOG_TRANSFORM}")
    print(f"Splits — TRAIN_END={TRAIN_END.date()} | VAL_END={VAL_END.date()}")

    # Acumuladores (idénticos al notebook main())
    naive_metrics_all       = []
    naive_preds_all         = []
    naive_contrib_all       = []
    naive_trials_all        = []
    arima_base_all          = []
    arima_trials_all        = []
    arima_search_all        = []
    arima_best_all          = []
    arima_preds_all         = []
    arima_param_importance_all = []
    final_selection_all     = []
    final_models_artifact   = {}

    for idx, cat in enumerate(categories, start=1):
        print(f"Procesando categoría {idx}/{len(categories)}: {cat}")
        df_cat = df[df["product_category_name"] == cat].copy()
        result = run_category_pipeline(df_cat)

        if not result["naive_metrics"].empty:         naive_metrics_all.append(result["naive_metrics"])
        if not result["naive_preds"].empty:           naive_preds_all.append(result["naive_preds"])
        if not result["naive_contrib"].empty:         naive_contrib_all.append(result["naive_contrib"])
        if not result["naive_trials"].empty:          naive_trials_all.append(result["naive_trials"])
        if not result["arima_base_metrics"].empty:    arima_base_all.append(result["arima_base_metrics"])
        if not result["arima_optuna_trials"].empty:   arima_trials_all.append(result["arima_optuna_trials"])
        if not result["arima_search"].empty:          arima_search_all.append(result["arima_search"])
        if not result["arima_best_metrics"].empty:    arima_best_all.append(result["arima_best_metrics"])
        if not result["arima_preds"].empty:           arima_preds_all.append(result["arima_preds"])
        if not result["arima_param_importance"].empty: arima_param_importance_all.append(result["arima_param_importance"])
        if not result["final_selection"].empty:       final_selection_all.append(result["final_selection"])
        final_models_artifact.update(result["final_models"])

    naive_metrics_df        = pd.concat(naive_metrics_all,       ignore_index=True) if naive_metrics_all       else pd.DataFrame()
    naive_preds_df          = pd.concat(naive_preds_all,         ignore_index=True) if naive_preds_all         else pd.DataFrame()
    naive_contrib_df        = pd.concat(naive_contrib_all,       ignore_index=True) if naive_contrib_all       else pd.DataFrame()
    naive_trials_df         = pd.concat(naive_trials_all,        ignore_index=True) if naive_trials_all        else pd.DataFrame()
    arima_base_df           = pd.concat(arima_base_all,          ignore_index=True) if arima_base_all          else pd.DataFrame()
    arima_trials_df         = pd.concat(arima_trials_all,        ignore_index=True) if arima_trials_all        else pd.DataFrame()
    arima_search_df         = pd.concat(arima_search_all,        ignore_index=True) if arima_search_all        else pd.DataFrame()
    arima_best_df           = pd.concat(arima_best_all,          ignore_index=True) if arima_best_all          else pd.DataFrame()
    arima_preds_df          = pd.concat(arima_preds_all,         ignore_index=True) if arima_preds_all         else pd.DataFrame()
    arima_param_importance_df = pd.concat(arima_param_importance_all, ignore_index=True) if arima_param_importance_all else pd.DataFrame()
    final_selection_df      = pd.concat(final_selection_all,     ignore_index=True) if final_selection_all     else pd.DataFrame()

    arima_search_df      = clean_arima_search_df(arima_search_df)
    arima_order_effect_df = build_arima_order_effect_table(arima_search_df)

    base_models_df = pd.concat([
        naive_metrics_df[naive_metrics_df["model"] == "naive_lag_1"] if not naive_metrics_df.empty else pd.DataFrame(),
        arima_base_df,
    ], ignore_index=True) if (not naive_metrics_df.empty or not arima_base_df.empty) else pd.DataFrame()

    arima_optimized_for_export = pd.DataFrame()
    if not arima_search_df.empty:
        arima_optimized_for_export = arima_search_df.assign(
            dataset="validation",
            model_family="arima",
            model=arima_search_df.apply(
                lambda row: f"ARIMA({int(row['p'])}, {int(row['d'])}, {int(row['q'])})", axis=1
            ),
        )[["dataset", "model_family", "model", "category", "mae", "rmse", "mape", "objective_value", "stability_penalty"]]

    optimized_models_df = pd.concat([
        naive_metrics_df[naive_metrics_df["model"] != "naive_lag_1"] if not naive_metrics_df.empty else pd.DataFrame(),
        arima_optimized_for_export,
    ], ignore_index=True) if (not naive_metrics_df.empty or not arima_optimized_for_export.empty) else pd.DataFrame()

    base_summary_df      = summarize_metrics(base_models_df,      ["dataset", "model_family", "model"])
    optimized_summary_df = summarize_metrics(optimized_models_df, ["dataset", "model_family", "model"])
    selected_summary_df  = (
        summarize_metrics(final_selection_df, ["dataset", "model_family", "model"])
        if not final_selection_df.empty else pd.DataFrame()
    )

    selected_pred_frames = []
    if not final_selection_df.empty:
        for _, row in final_selection_df[["category", "dataset", "model_family", "model"]].drop_duplicates().iterrows():
            if row["model_family"] == "naive":
                temp = naive_preds_df[
                    (naive_preds_df["category"] == row["category"]) &
                    (naive_preds_df["dataset"]  == row["dataset"])  &
                    (naive_preds_df["model"]    == row["model"])
                ].copy()
            else:
                temp = arima_preds_df[
                    (arima_preds_df["category"] == row["category"]) &
                    (arima_preds_df["dataset"]  == row["dataset"])  &
                    (arima_preds_df["model"]    == row["model"])
                ].copy()
            if not temp.empty:
                selected_pred_frames.append(temp)
    selected_predictions_df = (
        pd.concat(selected_pred_frames, ignore_index=True) if selected_pred_frames else pd.DataFrame()
    )

    # --- Exportar CSVs ---
    export_map = {
        "01_base_models_metrics.csv":               base_models_df,
        "02_optimized_models_metrics.csv":           optimized_models_df,
        "03_naive_proxy_importance.csv":             naive_contrib_df,
        "04_arima_optuna_search_validation.csv":     arima_search_df,
        "05_arima_hyperparameter_effect.csv":        arima_order_effect_df,
        "06_arima_parameter_proxy_importance.csv":   arima_param_importance_df,
        "07_final_selection_by_category.csv":        final_selection_df,
        "08_selected_predictions.csv":               selected_predictions_df,
        "09_summary_base_models.csv":                base_summary_df,
        "10_summary_optimized_models.csv":           optimized_summary_df,
        "11_summary_selected_models.csv":            selected_summary_df,
        "12_optuna_naive_trials.csv":                naive_trials_df,
        "13_optuna_arima_trials.csv":                arima_trials_df,
    }

    for filename, dataframe in export_map.items():
        if dataframe is not None and not dataframe.empty:
            dataframe.to_csv(OUTPUT_DIR / filename, index=False)
            print(f"Exportado: {OUTPUT_DIR / filename}")
        else:
            print(f"Sin datos para: {filename}")

    # --- Gráficos ---
    plot_model_comparison(base_summary_df, optimized_summary_df)
    for dataset_name in ["validation", "backtest"]:
        plot_selected_real_vs_pred(selected_predictions_df, dataset_name)
    plot_naive_proxy_importance(naive_contrib_df)
    plot_arima_proxy_importance(arima_param_importance_df)
    plot_arima_hyperparameter_effect(arima_order_effect_df)
    print("Gráficos exportados en:", OUTPUT_DIR)

    # --- Reporte markdown ---
    lines = []
    lines.append("# Reporte comparativo PRO con Optuna: Naive vs ARIMA\n")
    lines.append("## 1. Enfoque de optimización\n")
    lines.append(
        "Se utilizó Optuna para automatizar la optimización de ambos enfoques. En Naive, Optuna eligió "
        "la mejor regla entre rezagos y medias móviles. En ARIMA, Optuna buscó la mejor combinación de "
        f"p, d y q dentro de un espacio restringido y con penalización por inestabilidad, usando como "
        "función objetivo una combinación de MAE y volatilidad de predicciones.\n"
    )
    lines.append("## 2. Mejoras metodológicas incorporadas\n")
    lines.append(
        f"El pipeline incorpora: corrección del backtest ARIMA con historia real, transformación "
        f"logarítmica opcional (USE_LOG_TRANSFORM={USE_LOG_TRANSFORM}), filtrado de configuraciones "
        "explosivas, penalización por inestabilidad y un score final que combina validation, backtest y consistencia.\n"
    )
    lines.append("## 3. Sobre la interpretabilidad / 'feature importance'\n")
    lines.append(
        "Naive y ARIMA no generan feature importance tradicional. Por ello se construyó una "
        "interpretabilidad proxy: para Naive se cuantifica cuánto mejora cada regla respecto al baseline "
        "naive_lag_1; para ARIMA se reporta el peso relativo de los coeficientes finales y el efecto "
        "promedio de los hiperparámetros p, d y q sobre el MAE.\n"
    )
    if not base_summary_df.empty:
        lines.append("## 4. Resumen de modelos base\n")
        lines.append(base_summary_df.to_markdown(index=False))
        lines.append("\n")
    if not optimized_summary_df.empty:
        lines.append("## 5. Resumen de modelos optimizados\n")
        lines.append(optimized_summary_df.head(25).to_markdown(index=False))
        lines.append("\n")
    if not selected_summary_df.empty:
        lines.append("## 6. Modelos seleccionados\n")
        lines.append(selected_summary_df.to_markdown(index=False))
        lines.append("\n")
    if not naive_contrib_df.empty:
        agg_naive = (
            naive_contrib_df.groupby("model", dropna=False)[["proxy_importance_share_pct", "improvement_pct_vs_baseline"]]
            .mean().reset_index().sort_values("proxy_importance_share_pct", ascending=False)
        )
        lines.append("## 7. Aporte proxy de variables naive\n")
        lines.append(agg_naive.to_markdown(index=False))
        lines.append("\n")
    if not arima_param_importance_df.empty:
        agg_arima = (
            arima_param_importance_df.groupby(["term", "component_type"], dropna=False)[["proxy_importance_share_pct", "abs_coefficient"]]
            .mean().reset_index().sort_values("proxy_importance_share_pct", ascending=False)
        )
        lines.append("## 8. Aporte proxy de términos ARIMA\n")
        lines.append(agg_arima.to_markdown(index=False))
        lines.append("\n")

    report_path = OUTPUT_DIR / "14_report_model_optimization_optuna_pro.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Reporte markdown: {report_path}")

    # --- Serializar modelo final ---
    artifact = {
        "description": (
            "Modelo final por categoría (naive o ARIMA) seleccionado con Optuna PRO, "
            "penalización de estabilidad, consistencia y transformación logarítmica opcional"
        ),
        "train_end":                     str(TRAIN_END.date()),
        "val_end":                       str(VAL_END.date()),
        "use_log_transform":             USE_LOG_TRANSFORM,
        "selected_models_by_category":   final_models_artifact,
        "base_summary":    base_summary_df.to_dict(orient="records")     if not base_summary_df.empty     else [],
        "selected_summary": selected_summary_df.to_dict(orient="records") if not selected_summary_df.empty else [],
    }

    pkl_path = models_dir / "final_model_optuna.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Modelo final (.pkl): {pkl_path}")

    # --- MLflow logging ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"airflow_arima_optuna_{TRAIN_END.strftime('%Y%m')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_type":        "ARIMA+Optuna",
            "n_trials_arima":    N_TRIALS_ARIMA,
            "n_trials_naive":    N_TRIALS_NAIVE,
            "optuna_seed":       OPTUNA_SEED,
            "arima_p_range":     f"[{ARIMA_P_MIN},{ARIMA_P_MAX}]",
            "arima_d_range":     f"[{ARIMA_D_MIN},{ARIMA_D_MAX}]",
            "arima_q_range":     f"[{ARIMA_Q_MIN},{ARIMA_Q_MAX}]",
            "use_log_transform": USE_LOG_TRANSFORM,
            "train_end":         str(TRAIN_END.date()),
            "val_end":           str(VAL_END.date()),
            "n_categories":      len(categories),
        })

        if not selected_summary_df.empty:
            for _, row in selected_summary_df.iterrows():
                prefix = f"{row['dataset']}_selected"
                for metric in ["mae", "rmse", "mape"]:
                    val = row.get(metric)
                    if val is not None and np.isfinite(val):
                        mlflow.log_metric(f"{prefix}_{metric}", round(float(val), 4))

        if not base_summary_df.empty:
            naive_base = base_summary_df[base_summary_df["model"] == "naive_lag_1"]
            for _, row in naive_base.iterrows():
                prefix = f"{row['dataset']}_naive_lag1"
                for metric in ["mae", "rmse", "mape"]:
                    val = row.get(metric)
                    if val is not None and np.isfinite(val):
                        mlflow.log_metric(f"{prefix}_{metric}", round(float(val), 4))

        mlflow.log_artifact(str(pkl_path))
        for filename in export_map:
            path = OUTPUT_DIR / filename
            if path.exists():
                mlflow.log_artifact(str(path))

    # --- Calcular métricas resumen para save_report ---
    def _mean_metric(df, dataset, metric):
        if df.empty:
            return None
        sub = df[df["dataset"] == dataset]
        if sub.empty or sub[metric].isna().all():
            return None
        return round(float(sub[metric].mean(skipna=True)), 4)

    val_mape_selected  = _mean_metric(selected_summary_df, "validation", "mape")
    bt_mape_selected   = _mean_metric(selected_summary_df, "backtest",   "mape")
    bt_rmse_selected   = _mean_metric(selected_summary_df, "backtest",   "rmse")

    naive_base_summary = base_summary_df[base_summary_df["model"] == "naive_lag_1"] if not base_summary_df.empty else pd.DataFrame()
    val_mape_naive     = _mean_metric(naive_base_summary, "validation", "mape")
    bt_mape_naive      = _mean_metric(naive_base_summary, "backtest",   "mape")

    improvement_pp = (
        round(bt_mape_naive - bt_mape_selected, 4)
        if bt_mape_naive is not None and bt_mape_selected is not None
        else None
    )

    print("\nResultados finales (media entre categorías):")
    print(f"  Validation — Seleccionado MAPE: {val_mape_selected}%  |  Naive MAPE: {val_mape_naive}%")
    print(f"  Backtest   — Seleccionado MAPE: {bt_mape_selected}%   |  Naive MAPE: {bt_mape_naive}%")
    print("Proceso finalizado con la versión PRO.")

    return {
        "model":              "ARIMA+Optuna",
        "n_categories":       len(categories),
        "val_mape_selected":  val_mape_selected,
        "bt_mape_selected":   bt_mape_selected,
        "bt_rmse_selected":   bt_rmse_selected,
        "val_mape_naive":     val_mape_naive,
        "bt_mape_naive":      bt_mape_naive,
        "improvement_pp":     improvement_pp,
        "results_path":       str(OUTPUT_DIR / "07_final_selection_by_category.csv"),
        "pkl_path":           str(pkl_path),
        "train_end":          str(TRAIN_END.date()),
        "val_end":            str(VAL_END.date()),
        "n_trials_arima":     N_TRIALS_ARIMA,
        "n_trials_naive":     N_TRIALS_NAIVE,
    }
