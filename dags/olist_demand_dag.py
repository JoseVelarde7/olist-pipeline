"""
DAG: olist_demand_pipeline
Orquesta el pipeline completo de demand forecasting de Olist.

Flujo — rama inicial (mode=initial, por defecto):
  branch_decision -> run_initial_eda -> clean_raw_data -> load_to_db
                                                              |
                                                       branches_join
                                                              |
                                          validate_connection -> build_master_table
                                              -> build_monthly_base -> train_arima
                                                  -> save_report

Flujo — rama mensual (mode=monthly):
  branch_decision -> ingest_monthly -> clean_monthly -> append_to_db
                                                              |
                                                       branches_join
                                                              (continua igual)

Trigger con config:
  {}                               -> modo initial (carga completa desde cero si BD vacía)
  {"mode": "monthly"}              -> modo mensual (agrega nuevos CSVs de data/monthly/)
  {"mode": "initial", "force_reload": true} -> fuerza recarga aunque BD tenga datos
  {"train_end": "2017-12-01", "val_end": "2018-05-01"} -> overrides del split ARIMA

Schedule: primer día de cada mes.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

sys.path.insert(0, "/opt/airflow")

RAW_DIR         = Path("/opt/airflow/data/raw")
MONTHLY_DIR     = Path("/opt/airflow/data/monthly")
INTERIM_DIR     = Path("/opt/airflow/data/interim")
INTERIM_MONTHLY = Path("/opt/airflow/data/interim/monthly")
REPORTS_DIR     = Path("/opt/airflow/reports/metrics")
MODELS_DIR      = Path("/opt/airflow/models")
SCHEMA_FILE     = Path("/opt/airflow/config/schema.sql")

# Defaults del split ARIMA (notebook 03 del equipo)
ARIMA_TRAIN_END = "2017-12-01"
ARIMA_VAL_END   = "2018-05-01"


@dag(
    dag_id="olist_demand_pipeline",
    description="Pipeline mensual de forecasting de demanda por categoría — Olist",
    schedule="0 0 1 * *",
    start_date=days_ago(1),
    catchup=False,
    tags=["olist", "demand_forecasting", "sprint2"],
    doc_md="""
## Pipeline de Demand Forecasting — Olist (Sprint 2)

Ejecuta el pipeline completo cada primer día del mes:

### Carga de datos (Sprint 1 — intacto)
1. **run_initial_eda** — valida existencia y calidad de los CSVs crudos en data/raw/
2. **clean_raw_data** — limpia los CSVs y exporta data/interim/cleaned_*.csv
3. **load_to_db** — migra los datos limpios a MariaDB (DROP + CREATE + carga)

### Pipeline ML (Sprint 2 — ARIMA por categoría)
4. **validate_connection** — verifica que MariaDB está disponible con datos
5. **build_master_table** — extrae el JOIN completo de MariaDB y calcula las 22
   variables derivadas del script cleaning_eda_olist_mastertable.py del equipo
6. **build_monthly_base** — agrega a nivel categoría-mes y crea demand_next_month
   como target (notebook 02 del equipo)
7. **train_arima** — entrena ARIMA(1,1,1) por categoría con validación y backtest,
   registra resultados en MLflow (notebook 03 del equipo)
8. **save_report** — guarda métricas en reports/metrics/pipeline_summary.json

**Trigger manual:** usa el botón "Trigger DAG" en la UI para ejecutar con datos actuales.
    """,
)
def olist_demand_pipeline():

    # ──────────────────────────────────────────────
    # DECISIÓN — Rama inicial o mensual
    # ──────────────────────────────────────────────
    @task.branch(task_id="branch_decision")
    def branch_decision_task() -> str:
        ctx  = get_current_context()
        conf = ctx["dag_run"].conf or {}
        mode = conf.get("mode", "initial")
        print(f"Modo seleccionado: {mode}")
        if mode == "monthly":
            return "ingest_monthly_task"
        return "run_initial_eda"

    # ──────────────────────────────────────────────
    # RAMA MENSUAL — Tarea A
    # ──────────────────────────────────────────────
    @task(task_id="ingest_monthly_task")
    def ingest_monthly_task() -> dict:
        from src.data.ingest_monthly import validate_monthly_csvs
        summary = validate_monthly_csvs(MONTHLY_DIR)
        print(f"CSVs mensuales encontrados: {len(summary['found'])}")
        for name, shape in summary["found"].items():
            print(f"  {name:<25} {shape[0]:>10,} filas")
        if summary["missing"]:
            print(f"Archivos no encontrados (opcionales): {summary['missing']}")
        return summary

    # ──────────────────────────────────────────────
    # RAMA MENSUAL — Tarea B
    # ──────────────────────────────────────────────
    @task(task_id="clean_monthly_task")
    def clean_monthly_task(ingest_summary: dict) -> dict:
        from src.data.clean_data import run_cleaning
        print(f"Limpiando datos mensuales de {ingest_summary['monthly_dir']}")
        result = run_cleaning(MONTHLY_DIR, INTERIM_MONTHLY)
        print(f"Datasets limpios exportados: {len(result['export_summary'])}")
        return result

    # ──────────────────────────────────────────────
    # RAMA MENSUAL — Tarea C
    # ──────────────────────────────────────────────
    @task(task_id="append_to_db_task")
    def append_to_db_task(clean_result: dict) -> dict:
        from src.data.ingest_monthly import append_to_db
        print(f"Agregando datos de {len(clean_result['export_summary'])} tablas a MariaDB")
        result = append_to_db(INTERIM_MONTHLY)
        print(f"Nuevos registros insertados: {result['total_new']:,}")
        for table, n in result["counts"].items():
            print(f"  {table:<35} {n:>10,} nuevos")
        return result

    # ──────────────────────────────────────────────
    # RAMA INICIAL — Tarea 1: EDA
    # ──────────────────────────────────────────────
    @task(task_id="run_initial_eda")
    def run_initial_eda_task() -> dict:
        from src.data.run_eda import run_initial_eda
        summary = run_initial_eda(RAW_DIR)
        print(f"CSVs validados: {len(summary['shapes'])} datasets")
        print(f"Total filas raw: {summary['total_filas']:,}")
        print(f"Huérfanos FK en datos crudos: {summary['total_huerfanos']:,}")
        return summary

    # ──────────────────────────────────────────────
    # RAMA INICIAL — Tarea 2: Limpieza
    # ──────────────────────────────────────────────
    @task(task_id="clean_raw_data")
    def clean_raw_data_task(eda_summary: dict) -> dict:
        from src.data.clean_data import run_cleaning
        print(f"Iniciando limpieza. Datasets raw: {len(eda_summary['shapes'])}")
        result = run_cleaning(RAW_DIR, INTERIM_DIR)
        print(f"Datasets exportados: {len(result['export_summary'])}")
        for name, info in result["export_summary"].items():
            print(f"  {name:<25} {info['rows']:>10,} filas")
        return result

    # ──────────────────────────────────────────────
    # RAMA INICIAL — Tarea 3: Migración a MariaDB
    # ──────────────────────────────────────────────
    @task(task_id="load_to_db")
    def load_to_db_task(cleaning_result: dict) -> dict:
        from src.data.extract import get_connection
        from src.data.migrate import run_migration

        ctx   = get_current_context()
        conf  = ctx["dag_run"].conf or {}
        force = conf.get("force_reload", False)

        try:
            conn   = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM olist_orders")
            n_existing = cursor.fetchone()[0]
            cursor.close()
            conn.close()
        except Exception:
            n_existing = 0

        if n_existing > 0 and not force:
            print(f"BD ya contiene {n_existing:,} órdenes — migración omitida.")
            print("Para forzar recarga: dag_run.conf = {\"force_reload\": true}")
            return {"counts": {}, "status": "skipped", "n_existing": n_existing}

        if force:
            print("force_reload=true — ejecutando migración completa.")
        else:
            print("BD vacía — ejecutando carga inicial.")

        result = run_migration(INTERIM_DIR, SCHEMA_FILE)
        total  = sum(result["counts"].values())
        print(f"Migración completada. Total registros: {total:,}")
        for table, n in result["counts"].items():
            print(f"  {table:<35} {n:>10,}")
        return result

    # ──────────────────────────────────────────────
    # CONVERGENCIA — Join de ambas ramas
    # ──────────────────────────────────────────────
    @task(task_id="branches_join", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def branches_join_task(load_result: dict | None, append_result: dict | None) -> str:
        if load_result and load_result.get("status") != "skipped":
            n = sum(load_result.get("counts", {}).values())
            print(f"Rama inicial completada — {n:,} registros cargados en BD.")
            return "initial"
        if append_result:
            print(f"Rama mensual completada — {append_result.get('total_new', 0):,} registros nuevos.")
            return "monthly"
        print("BD ya tenía datos — migración omitida, continuando con pipeline ARIMA.")
        return "initial_skipped"

    # ──────────────────────────────────────────────
    # PIPELINE ML — 1. Validar conexión
    # ──────────────────────────────────────────────
    @task(task_id="validate_connection")
    def validate_connection() -> dict:
        from src.data.extract import get_connection

        ctx  = get_current_context()
        conf = ctx["dag_run"].conf or {}
        train_end = conf.get("train_end", ARIMA_TRAIN_END)
        val_end   = conf.get("val_end",   ARIMA_VAL_END)

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM olist_orders WHERE order_status = 'delivered'")
        n_orders = cursor.fetchone()[0]
        cursor.close()
        conn.close()

        print(f"MariaDB OK — órdenes entregadas: {n_orders:,}")
        print(f"Splits ARIMA: train_end={train_end}  val_end={val_end}")
        assert n_orders > 0, "La tabla olist_orders no tiene registros entregados"

        return {
            "n_orders":  n_orders,
            "status":    "ok",
            "train_end": train_end,
            "val_end":   val_end,
        }

    # ──────────────────────────────────────────────
    # PIPELINE ML — 2. Construir master table
    # ──────────────────────────────────────────────
    @task(task_id="build_master_table")
    def build_master_table_task(validation: dict) -> dict:
        """
        Extrae el JOIN completo desde MariaDB y computa las 22 variables derivadas
        definidas en cleaning_eda_olist_mastertable.py del equipo (Sprint 2).
        Aplica la limpieza Stage 1: estandarización de texto y eliminación de
        duplicados exactos.
        Guarda el resultado en data/interim/master_table.parquet.
        """
        from src.data.build_master_table import clean_master_table, extract_master_table
        from src.data.extract import get_connection

        print(f"Extrayendo master table desde MariaDB ({validation['n_orders']:,} órdenes)...")
        conn = get_connection()
        df   = extract_master_table(conn)
        conn.close()
        print(f"Master table extraída: {df.shape[0]:,} filas × {df.shape[1]} columnas")

        df, clean_info = clean_master_table(df)
        print(f"Limpieza Stage 1: {clean_info['duplicates_removed']} duplicados eliminados")
        print(f"Columnas con nulos: {len(clean_info['null_by_col'])}")

        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(INTERIM_DIR / "master_table.parquet")
        df.to_parquet(output_path, index=False)
        print(f"Master table guardada: {output_path}")

        return {
            "master_path":        output_path,
            "n_rows":             len(df),
            "n_cols":             df.shape[1],
            "duplicates_removed": clean_info["duplicates_removed"],
            "train_end":          validation["train_end"],
            "val_end":            validation["val_end"],
        }

    # ──────────────────────────────────────────────
    # PIPELINE ML — 3. Construir base mensual
    # ──────────────────────────────────────────────
    @task(task_id="build_monthly_base")
    def build_monthly_base_task(master_info: dict) -> dict:
        """
        Agrega la master table a nivel (categoría, mes) y crea el target
        demand_next_month = shift(-1).
        Réplica del notebook 02_build_monthly_base_olist.ipynb del equipo.
        Guarda el resultado en data/interim/monthly_base.parquet.
        """
        import pandas as pd

        from src.features.build_monthly_base import get_monthly_base

        master_path = master_info["master_path"]
        df_master   = pd.read_parquet(master_path)
        print(f"Master table cargada: {df_master.shape[0]:,} filas")

        df_monthly = get_monthly_base(df_master)

        n_cat   = df_monthly["product_category_name"].nunique()
        n_meses = df_monthly["year_month"].nunique()
        date_min = df_monthly["year_month"].min().strftime("%Y-%m")
        date_max = df_monthly["year_month"].max().strftime("%Y-%m")

        print(f"Base mensual: {len(df_monthly):,} filas | {n_cat} categorías | {n_meses} meses")
        print(f"Rango temporal: {date_min} → {date_max}")

        output_path = str(INTERIM_DIR / "monthly_base.parquet")
        df_monthly.to_parquet(output_path, index=False)
        print(f"Base mensual guardada: {output_path}")

        return {
            "monthly_path": output_path,
            "n_rows":       len(df_monthly),
            "n_categories": n_cat,
            "n_months":     n_meses,
            "date_range":   f"{date_min} → {date_max}",
            "train_end":    master_info["train_end"],
            "val_end":      master_info["val_end"],
        }

    # ──────────────────────────────────────────────
    # PIPELINE ML — 4. Entrenar ARIMA
    # ──────────────────────────────────────────────
    @task(task_id="train_arima")
    def train_arima_task(monthly_info: dict) -> dict:
        """
        Entrena ARIMA(1,1,1) por categoría con splits train/validation/backtest.
        Réplica corregida del notebook 03_train_arima_sarima_olist.ipynb del equipo:
          - PeriodIndex correcto para statsmodels
          - Backtest con reentrenamiento en train+val
          - Integración con MLflow
        """
        import pandas as pd

        from src.models.arima_model import DEFAULT_ARIMA_ORDER, run_arima_experiment

        train_end = monthly_info["train_end"]
        val_end   = monthly_info["val_end"]

        df_monthly = pd.read_parquet(monthly_info["monthly_path"])
        print(f"Base mensual: {df_monthly.shape} | Categorías: {df_monthly['product_category_name'].nunique()}")
        print(f"Splits — train_end: {train_end}  val_end: {val_end}")
        print(f"Orden ARIMA: {DEFAULT_ARIMA_ORDER}")

        run_name = f"airflow_arima_{train_end[:7].replace('-', '')}"
        output   = run_arima_experiment(
            df=df_monthly,
            train_end=train_end,
            val_end=val_end,
            arima_order=DEFAULT_ARIMA_ORDER,
            run_name=run_name,
        )
        results = output["results"]

        def _mean(split, model, col):
            sub = results[(results["dataset"] == split) & (results["model"] == model)]
            if sub.empty or sub[col].isna().all():
                return None
            return round(float(sub[col].mean(skipna=True)), 4)

        val_mape_arima = _mean("validation", "ARIMA",       "mape")
        val_mape_naive = _mean("validation", "naive_lag_1", "mape")
        bt_mape_arima  = _mean("backtest",   "ARIMA",       "mape")
        bt_mape_naive  = _mean("backtest",   "naive_lag_1", "mape")
        bt_rmse_arima  = _mean("backtest",   "ARIMA",       "rmse")

        print(f"\nResultados (media entre categorías):")
        print(f"  Validation — ARIMA MAPE: {val_mape_arima}%  |  Naive MAPE: {val_mape_naive}%")
        print(f"  Backtest   — ARIMA MAPE: {bt_mape_arima}%   |  Naive MAPE: {bt_mape_naive}%")

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = str(REPORTS_DIR / "arima_results_by_category.csv")
        results.to_csv(results_path, index=False)

        improvement = (
            round(bt_mape_naive - bt_mape_arima, 4)
            if bt_mape_naive is not None and bt_mape_arima is not None
            else None
        )

        return {
            "arima_order":      str(DEFAULT_ARIMA_ORDER),
            "n_categories":     int(results["category"].nunique()),
            "val_mape_arima":   val_mape_arima,
            "val_mape_naive":   val_mape_naive,
            "bt_mape_arima":    bt_mape_arima,
            "bt_mape_naive":    bt_mape_naive,
            "bt_rmse_arima":    bt_rmse_arima,
            "improvement_pp":   improvement,
            "results_path":     results_path,
            "train_end":        train_end,
            "val_end":          val_end,
        }

    # ──────────────────────────────────────────────
    # PIPELINE ML — 5. Reporte final
    # ──────────────────────────────────────────────
    @task(task_id="save_report")
    def save_report(arima_metrics: dict) -> None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        summary = {
            "execution_date": datetime.now().isoformat(),
            "model":          "ARIMA",
            "arima_order":    arima_metrics["arima_order"],
            "n_categories":   arima_metrics["n_categories"],
            "train_end":      arima_metrics["train_end"],
            "val_end":        arima_metrics["val_end"],
            "metrics": {
                "validation": {
                    "arima_mape": arima_metrics["val_mape_arima"],
                    "naive_mape": arima_metrics["val_mape_naive"],
                },
                "backtest": {
                    "arima_mape":    arima_metrics["bt_mape_arima"],
                    "arima_rmse":    arima_metrics["bt_rmse_arima"],
                    "naive_mape":    arima_metrics["bt_mape_naive"],
                    "improvement_pp": arima_metrics["improvement_pp"],
                },
            },
            "objetivo_cumplido": (
                arima_metrics["bt_mape_arima"] is not None
                and arima_metrics["bt_mape_arima"] < 25.0
            ),
        }

        output_path = REPORTS_DIR / "pipeline_summary.json"
        output_path.write_text(json.dumps(summary, indent=2, default=str))
        print(f"Reporte guardado: {output_path}")
        print(json.dumps(summary, indent=2, default=str))

    # ──────────────────────────────────────────────
    # DEPENDENCIAS DEL DAG
    # ──────────────────────────────────────────────

    branch = branch_decision_task()

    # --- Rama inicial ---
    eda_summary     = run_initial_eda_task()
    cleaning_result = clean_raw_data_task(eda_summary)
    load_result     = load_to_db_task(cleaning_result)
    branch >> eda_summary

    # --- Rama mensual ---
    monthly_summary = ingest_monthly_task()
    monthly_clean   = clean_monthly_task(monthly_summary)
    append_result   = append_to_db_task(monthly_clean)
    branch >> monthly_summary

    # --- Convergencia ---
    join = branches_join_task(load_result, append_result)

    # --- Pipeline ARIMA (Sprint 2) ---
    validation   = validate_connection()
    join        >> validation
    master_info  = build_master_table_task(validation)
    monthly_info = build_monthly_base_task(master_info)
    arima_out    = train_arima_task(monthly_info)
    save_report(arima_out)


olist_demand_pipeline()
