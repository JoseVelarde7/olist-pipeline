"""
DAG: olist_demand_pipeline
Orquesta el pipeline completo de demand forecasting de Olist.

Flujo — rama inicial (mode=initial, por defecto):
  branch_decision -> run_initial_eda -> clean_raw_data -> load_to_db
                                                              |
                                                       branches_join
                                                              |
                                             validate_connection -> extract_and_engineer
                                                 -> select_features_task -> train_and_log
                                                     -> save_report

Flujo — rama mensual (mode=monthly):
  branch_decision -> ingest_monthly -> clean_monthly -> append_to_db
                                                              |
                                                       branches_join
                                                              (continua igual)

Trigger con config:
  {}                        -> modo initial (carga completa desde cero si BD vacia)
  {"mode": "monthly"}       -> modo mensual (agrega nuevos CSVs de data/monthly/)
  {"mode": "initial", "force_reload": true} -> fuerza recarga aunque BD tenga datos

Schedule: primer dia de cada mes.
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

# Airflow monta src/ en /opt/airflow/src via volumen del docker-compose
sys.path.insert(0, "/opt/airflow")

RAW_DIR         = Path("/opt/airflow/data/raw")
MONTHLY_DIR     = Path("/opt/airflow/data/monthly")
INTERIM_DIR     = Path("/opt/airflow/data/interim")
INTERIM_MONTHLY = Path("/opt/airflow/data/interim/monthly")
REPORTS_DIR     = Path("/opt/airflow/reports/metrics")
MODELS_DIR      = Path("/opt/airflow/models")
SCHEMA_FILE     = Path("/opt/airflow/config/schema.sql")


@dag(
    dag_id="olist_demand_pipeline",
    description="Pipeline mensual de forecasting de demanda por categoria — Olist",
    schedule="0 0 1 * *",
    start_date=days_ago(1),
    catchup=False,
    tags=["olist", "demand_forecasting", "sprint2"],
    doc_md="""
## Pipeline de Demand Forecasting — Olist

Ejecuta el pipeline completo cada primer dia del mes:

1. **run_initial_eda** — valida existencia y calidad de los CSVs crudos en data/raw/
2. **clean_raw_data** — limpia los CSVs y exporta data/interim/cleaned_*.csv
3. **load_to_db** — migra los datos limpios a MariaDB (DROP + CREATE + carga)
4. **validate_connection** — verifica que MariaDB esta disponible con datos cargados
5. **extract_and_engineer** — carga ordenes desde MariaDB y construye features con DuckDB
6. **select_features_task** — filtra categorias de bajo volumen y elimina features redundantes
7. **train_and_log** — entrena LightGBM y registra metricas en MLflow
8. **save_report** — guarda predicciones y resumen de metricas en reports/

**Trigger manual:** usa el boton "Trigger DAG" en la UI para ejecutar con datos actuales.
    """,
)
def olist_demand_pipeline():

    # ──────────────────────────────────────────────
    # DECISION — Rama inicial o mensual
    # ──────────────────────────────────────────────
    @task.branch(task_id="branch_decision")
    def branch_decision_task() -> str:
        """
        Lee dag_run.conf["mode"] y enruta a la rama correspondiente:
          - "initial" (defecto): carga completa desde data/raw/ -> MariaDB
          - "monthly": ingestion incremental desde data/monthly/ -> MariaDB
        """
        ctx  = get_current_context()
        conf = ctx["dag_run"].conf or {}
        mode = conf.get("mode", "initial")
        print(f"Modo seleccionado: {mode}")
        if mode == "monthly":
            return "ingest_monthly_task"
        return "run_initial_eda"

    # ──────────────────────────────────────────────
    # RAMA MENSUAL — Tarea A: validar CSVs mensuales
    # ──────────────────────────────────────────────
    @task(task_id="ingest_monthly_task")
    def ingest_monthly_task() -> dict:
        """
        Verifica que los CSVs de data/monthly/ existan y reporta sus shapes.
        Al menos olist_orders_dataset.csv es obligatorio.
        """
        from src.data.ingest_monthly import validate_monthly_csvs

        summary = validate_monthly_csvs(MONTHLY_DIR)
        print(f"CSVs mensuales encontrados: {len(summary['found'])}")
        for name, shape in summary["found"].items():
            print(f"  {name:<25} {shape[0]:>10,} filas")
        if summary["missing"]:
            print(f"Archivos no encontrados (opcionales): {summary['missing']}")
        return summary

    # ──────────────────────────────────────────────
    # RAMA MENSUAL — Tarea B: limpiar datos mensuales
    # ──────────────────────────────────────────────
    @task(task_id="clean_monthly_task")
    def clean_monthly_task(ingest_summary: dict) -> dict:
        """
        Aplica las mismas transformaciones de limpieza sobre los CSVs mensuales.
        Exporta los datos limpios a data/interim/monthly/cleaned_*.csv.
        """
        from src.data.clean_data import run_cleaning

        print(f"Limpiando datos mensuales de {ingest_summary['monthly_dir']}")
        result = run_cleaning(MONTHLY_DIR, INTERIM_MONTHLY)
        print(f"Datasets limpios exportados: {len(result['export_summary'])}")
        return result

    # ──────────────────────────────────────────────
    # RAMA MENSUAL — Tarea C: agregar a MariaDB
    # ──────────────────────────────────────────────
    @task(task_id="append_to_db_task")
    def append_to_db_task(clean_result: dict) -> dict:
        """
        Agrega los registros mensuales limpios a MariaDB usando INSERT IGNORE.
        Los registros ya existentes (mismo PK) se omiten sin error.
        """
        from src.data.ingest_monthly import append_to_db

        print(f"Agregando datos de {len(clean_result['export_summary'])} tablas a MariaDB")
        result = append_to_db(INTERIM_MONTHLY)
        print(f"Nuevos registros insertados: {result['total_new']:,}")
        for table, n in result["counts"].items():
            print(f"  {table:<35} {n:>10,} nuevos")
        return result

    # ──────────────────────────────────────────────
    # TAREA 1 — EDA inicial sobre datos crudos
    # ──────────────────────────────────────────────
    @task(task_id="run_initial_eda")
    def run_initial_eda_task() -> dict:
        """
        Carga los 9 CSVs de data/raw/ y ejecuta el EDA inicial:
        shapes, nulos, duplicados en PKs e integridad referencial con DuckDB.
        Los problemas detectados aqui son esperados y se corrigen en clean_raw_data.
        """
        from src.data.run_eda import run_initial_eda

        summary = run_initial_eda(RAW_DIR)
        print(f"CSVs validados: {len(summary['shapes'])} datasets")
        print(f"Total filas raw: {summary['total_filas']:,}")
        print(f"Huerfanos FK en datos crudos: {summary['total_huerfanos']:,} (seran corregidos en limpieza)")
        return summary

    # ──────────────────────────────────────────────
    # TAREA 2 — Limpieza de datos
    # ──────────────────────────────────────────────
    @task(task_id="clean_raw_data")
    def clean_raw_data_task(eda_summary: dict) -> dict:
        """
        Aplica las 8 operaciones de limpieza sobre los CSVs crudos:
        - Deduplicacion geolocation (261,831 duplicados)
        - Reparacion FK rotas en products (13 categorias -> 'outros')
        - Imputacion de nulos en products (610 sin categoria, 2 sin dimensiones)
        - Estandarizacion de strings en customers, sellers y orders
        - Eliminacion de 8 ordenes 'delivered' sin fecha de entrega + cascade
        - Deduplicacion de 814 reviews con review_id duplicado
        Exporta data/interim/cleaned_*.csv listos para migracion.
        """
        from src.data.clean_data import run_cleaning

        print(f"Iniciando limpieza. Datos raw validados: {len(eda_summary['shapes'])} datasets")
        result = run_cleaning(RAW_DIR, INTERIM_DIR)
        print(f"Datasets exportados: {len(result['export_summary'])}")
        for name, info in result["export_summary"].items():
            print(f"  {name:<25} {info['rows']:>10,} filas")
        return result

    # ──────────────────────────────────────────────
    # TAREA 3 — Migracion a MariaDB (condicional)
    # ──────────────────────────────────────────────
    @task(task_id="load_to_db")
    def load_to_db_task(cleaning_result: dict) -> dict:
        """
        Carga inicial de datos a MariaDB desde data/interim/cleaned_*.csv.
        Si la BD ya tiene ordenes cargadas, omite la migracion para preservar
        datos incrementales que hayan llegado entre corridas mensuales.
        Para forzar una recarga completa pasar force_reload=true en dag_run.conf.
        """
        from src.data.extract import get_connection
        from src.data.migrate import run_migration

        ctx   = get_current_context()
        conf  = ctx["dag_run"].conf or {}
        force = conf.get("force_reload", False)

        # Verificar si la BD ya tiene datos
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
            print(f"BD ya contiene {n_existing:,} ordenes — migracion omitida.")
            print("Para forzar recarga completa: dag_run.conf = {\"force_reload\": true}")
            return {"counts": {}, "status": "skipped", "n_existing": n_existing}

        if force:
            print("force_reload=true — ejecutando migracion completa.")
        else:
            print("BD vacia — ejecutando carga inicial.")

        result = run_migration(INTERIM_DIR, SCHEMA_FILE)
        total  = sum(result["counts"].values())
        print(f"Migracion completada. Total registros cargados: {total:,}")
        for table, n in result["counts"].items():
            print(f"  {table:<35} {n:>10,}")
        return result

    # ──────────────────────────────────────────────
    # CONVERGENCIA — Punto de union de ambas ramas
    # ──────────────────────────────────────────────
    @task(task_id="branches_join", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def branches_join_task(load_result: dict | None, append_result: dict | None) -> str:
        """
        Punto de convergencia entre la rama inicial y la mensual.
        Se ejecuta cuando cualquiera de las dos ramas termina exitosamente.
        La rama que no se ejecuto llega como None (skipped).
        """
        if load_result and load_result.get("status") != "skipped":
            n = sum(load_result.get("counts", {}).values())
            print(f"Rama inicial completada — {n:,} registros cargados en BD.")
            return "initial"
        if append_result:
            print(f"Rama mensual completada — {append_result.get('total_new', 0):,} registros nuevos.")
            return "monthly"
        print("BD ya tenia datos — migracion omitida, continuando con ML pipeline.")
        return "initial_skipped"

    # ──────────────────────────────────────────────
    # CONVERGENCIA — Validar conexion a MariaDB
    # ──────────────────────────────────────────────
    @task(task_id="validate_connection")
    def validate_connection() -> dict:
        """
        Verifica que MariaDB este accesible y que la tabla olist_orders
        tenga registros. Lee el periodo de evaluacion desde dag_run.conf
        (train_end, test_start, test_end) con valores por defecto si no se pasa conf.
        """
        from src.data.extract import get_connection
        from src.features.build_features import TRAIN_END, TEST_START, TEST_END

        ctx  = get_current_context()
        conf = ctx["dag_run"].conf or {}
        train_end  = conf.get("train_end",  TRAIN_END)
        test_start = conf.get("test_start", TEST_START)
        test_end   = conf.get("test_end",   TEST_END)

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM olist_orders WHERE order_status = 'delivered'")
        n_orders = cursor.fetchone()[0]
        cursor.close()
        conn.close()

        print(f"MariaDB OK — ordenes entregadas: {n_orders:,}")
        print(f"Periodo: train_end={train_end}  test={test_start} -> {test_end}")
        assert n_orders > 0, "La tabla olist_orders no tiene registros entregados"

        return {
            "n_orders":   n_orders,
            "status":     "ok",
            "train_end":  train_end,
            "test_start": test_start,
            "test_end":   test_end,
        }

    # ──────────────────────────────────────────────
    # TAREA 2 — Extraccion y feature engineering
    # ──────────────────────────────────────────────
    @task(task_id="extract_and_engineer")
    def extract_and_engineer(validation: dict) -> dict:
        """
        Carga ordenes entregadas desde MariaDB y construye la tabla de features
        usando DuckDB (lags, rolling averages, flags temporales).
        Guarda el resultado en data/interim/features_draft.parquet.

        Returns:
            Dict con ruta al parquet y periodo de evaluacion.
        """
        from src.data.extract import get_connection, load_delivered_orders
        from src.features.build_features import get_feature_table

        train_end  = validation["train_end"]
        test_start = validation["test_start"]
        test_end   = validation["test_end"]

        print(f"Extrayendo datos ({validation['n_orders']:,} ordenes disponibles)...")
        print(f"Periodo: train_end={train_end}  test={test_start} -> {test_end}")
        conn = get_connection()
        df_orders = load_delivered_orders(conn)
        conn.close()
        print(f"Ordenes cargadas: {len(df_orders):,} | Categorias: {df_orders['product_category_name'].nunique()}")

        print("Construyendo features con DuckDB...")
        df_features = get_feature_table(df_orders)
        print(f"Feature table: {df_features.shape} | Categorias modelables: {df_features['product_category_name'].nunique()}")

        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(INTERIM_DIR / "features_draft.parquet")
        df_features.to_parquet(output_path, index=False)
        print(f"Features guardadas: {output_path}")

        return {
            "features_path": output_path,
            "train_end":     train_end,
            "test_start":    test_start,
            "test_end":      test_end,
        }

    # ──────────────────────────────────────────────
    # TAREA 3 — Feature selection
    # ──────────────────────────────────────────────
    @task(task_id="select_features_task")
    def select_features_task(extraction: dict) -> dict:
        """
        Aplica el pipeline de seleccion de variables:
          1. Filtra categorias con volumen insuficiente (< 50 ord/mes o < 9 meses)
          2. Elimina features con correlacion > 0.95
          3. Elimina features con importancia < 3% del total
          4. Calcula SHAP values para explicabilidad

        Guarda el dataset filtrado en data/interim/features_selected.parquet.

        Returns:
            Dict con ruta al parquet, lista de features seleccionadas y periodo.
        """
        import pandas as pd

        from src.features.build_features import FEATURE_COLS
        from src.features.select_features import select_features_pipeline
        from src.models.train import DEFAULT_LGB_PARAMS, TARGET_COL, train_model

        features_path = extraction["features_path"]
        train_end  = extraction["train_end"]
        test_start = extraction["test_start"]
        test_end   = extraction["test_end"]

        df_model = pd.read_parquet(features_path)
        print(f"Dataset cargado: {df_model.shape}")
        print(f"Periodo: train_end={train_end}  test={test_start} -> {test_end}")

        # Entrenar modelo base para obtener importancias
        train = df_model[df_model["year_month"] <= train_end]
        model_base = train_model(train[FEATURE_COLS], train[TARGET_COL], DEFAULT_LGB_PARAMS)

        feat_imp = pd.DataFrame({
            "feature":    FEATURE_COLS,
            "importance": model_base.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        df_selected, features_final, shap_imp = select_features_pipeline(
            df_model=df_model,
            feature_cols=FEATURE_COLS,
            feat_imp=feat_imp,
            model=model_base,
            min_mean_demand=50.0,
            min_months=9,
            corr_threshold=0.95,
            min_importance_pct=3.0,
            verbose=True,
        )

        print(f"Categorias seleccionadas: {df_selected['product_category_name'].nunique()}")
        print(f"Features seleccionadas ({len(features_final)}): {features_final}")
        print(f"SHAP top feature: {shap_imp.iloc[0]['feature']} ({shap_imp.iloc[0]['shap_importance']:.2f})")

        output_path = str(INTERIM_DIR / "features_selected.parquet")
        df_selected.to_parquet(output_path, index=False)

        return {
            "selected_path":    output_path,
            "features_final":   features_final,
            "n_categories":     int(df_selected["product_category_name"].nunique()),
            "shap_top_feature": shap_imp.iloc[0]["feature"],
            "train_end":        train_end,
            "test_start":       test_start,
            "test_end":         test_end,
        }

    # ──────────────────────────────────────────────
    # TAREA 4 — Entrenamiento + MLflow
    # ──────────────────────────────────────────────
    @task(task_id="train_and_log")
    def train_and_log(selection: dict) -> dict:
        """
        Entrena LightGBM con walk-forward validation sobre las features seleccionadas
        y registra todo en MLflow: parametros, metricas, artefactos y modelo.

        Returns:
            Dict con metricas globales del modelo.
        """
        import pandas as pd

        from src.models.train import run_experiment

        df_selected = pd.read_parquet(selection["selected_path"])
        features_final = selection["features_final"]
        train_end  = selection["train_end"]
        test_start = selection["test_start"]
        test_end   = selection["test_end"]

        print(f"Entrenando sobre {df_selected['product_category_name'].nunique()} categorias")
        print(f"Features: {features_final}")
        print(f"Periodo: train_end={train_end}  test={test_start} -> {test_end}")

        output = run_experiment(
            df_model=df_selected,
            feature_cols=features_final,
            run_name=f"airflow_monthly_{train_end[:7].replace('-', '')}",
            models_dir=MODELS_DIR,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        metrics = output["metrics"]
        naive   = output["naive"]
        print(f"MAPE LightGBM : {metrics['mape']:.1f}%")
        print(f"MAPE Naive    : {naive['mape']:.1f}%")
        print(f"Mejora        : {naive['mape'] - metrics['mape']:+.1f} pp")

        return {
            "mape":       metrics["mape"],
            "rmse":       metrics["rmse"],
            "mae":        metrics["mae"],
            "mape_naive": naive["mape"],
            "improvement_pp": round(naive["mape"] - metrics["mape"], 2),
        }

    # ──────────────────────────────────────────────
    # TAREA 5 — Reporte final
    # ──────────────────────────────────────────────
    @task(task_id="save_report")
    def save_report(metrics: dict, selection: dict) -> None:
        """
        Guarda el resumen del pipeline en reports/metrics/pipeline_summary.json.
        Incluye metricas del modelo, features seleccionadas y metadatos de ejecucion.
        """
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        summary = {
            "execution_date":   datetime.now().isoformat(),
            "n_categories":     selection["n_categories"],
            "n_features":       len(selection["features_final"]),
            "features_used":    selection["features_final"],
            "shap_top_feature": selection["shap_top_feature"],
            "metrics": {
                "lgbm_mape":      metrics["mape"],
                "lgbm_rmse":      metrics["rmse"],
                "lgbm_mae":       metrics["mae"],
                "naive_mape":     metrics["mape_naive"],
                "improvement_pp": metrics["improvement_pp"],
            },
            "objetivo_cumplido": metrics["mape"] < 25.0,
        }

        output_path = REPORTS_DIR / "pipeline_summary.json"
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"Reporte guardado: {output_path}")
        print(json.dumps(summary, indent=2))

    # ──────────────────────────────────────────────
    # DEPENDENCIAS DEL DAG
    # ──────────────────────────────────────────────

    # Punto de decision
    branch = branch_decision_task()

    # --- Rama inicial ---
    eda_summary     = run_initial_eda_task()
    cleaning_result = clean_raw_data_task(eda_summary)
    load_result     = load_to_db_task(cleaning_result)
    branch >> eda_summary

    # --- Rama mensual ---
    monthly_summary  = ingest_monthly_task()
    monthly_clean    = clean_monthly_task(monthly_summary)
    append_result    = append_to_db_task(monthly_clean)
    branch >> monthly_summary

    # --- Convergencia: ambas ramas se unen aqui ---
    join = branches_join_task(load_result, append_result)

    # --- Pipeline ML (sin cambios) ---
    validation = validate_connection()
    join       >> validation
    extraction = extract_and_engineer(validation)
    selection  = select_features_task(extraction)
    metrics    = train_and_log(selection)
    save_report(metrics, selection)


olist_demand_pipeline()
