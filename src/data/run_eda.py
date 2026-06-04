"""
EDA inicial sobre los CSVs de datos crudos.
Verifica existencia, shapes, nulos, duplicados en PKs y FK criticas con DuckDB.
"""
import logging
from pathlib import Path
from typing import Dict, List

import duckdb
import polars as pl

logger = logging.getLogger(__name__)

DATASETS: Dict[str, str] = {
    "customers":            "olist_customers_dataset.csv",
    "geolocation":          "olist_geolocation_dataset.csv",
    "order_items":          "olist_order_items_dataset.csv",
    "order_payments":       "olist_order_payments_dataset.csv",
    "order_reviews":        "olist_order_reviews_dataset.csv",
    "orders":               "olist_orders_dataset.csv",
    "products":             "olist_products_dataset.csv",
    "sellers":              "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

PRIMARY_KEYS: Dict[str, List[str]] = {
    "customers":            ["customer_id"],
    "orders":               ["order_id"],
    "products":             ["product_id"],
    "sellers":              ["seller_id"],
    "order_items":          ["order_id", "order_item_id"],
    "order_payments":       ["order_id", "payment_sequential"],
    "order_reviews":        ["review_id"],
    "category_translation": ["product_category_name"],
}

FK_CHECKS: Dict[str, str] = {
    "orders.customer_id -> customers": """
        SELECT COUNT(*) FROM orders o
        LEFT JOIN customers c ON o.customer_id = c.customer_id
        WHERE c.customer_id IS NULL
    """,
    "order_items.order_id -> orders": """
        SELECT COUNT(*) FROM order_items oi
        LEFT JOIN orders o ON oi.order_id = o.order_id
        WHERE o.order_id IS NULL
    """,
    "order_items.product_id -> products": """
        SELECT COUNT(*) FROM order_items oi
        LEFT JOIN products p ON oi.product_id = p.product_id
        WHERE p.product_id IS NULL
    """,
    "order_items.seller_id -> sellers": """
        SELECT COUNT(*) FROM order_items oi
        LEFT JOIN sellers s ON oi.seller_id = s.seller_id
        WHERE s.seller_id IS NULL
    """,
    "order_payments.order_id -> orders": """
        SELECT COUNT(*) FROM order_payments op
        LEFT JOIN orders o ON op.order_id = o.order_id
        WHERE o.order_id IS NULL
    """,
    "order_reviews.order_id -> orders": """
        SELECT COUNT(*) FROM order_reviews orv
        LEFT JOIN orders o ON orv.order_id = o.order_id
        WHERE o.order_id IS NULL
    """,
    "products.category -> category_translation": """
        SELECT COUNT(*) FROM products p
        LEFT JOIN category_translation ct
               ON p.product_category_name = ct.product_category_name
        WHERE p.product_category_name IS NOT NULL
          AND ct.product_category_name IS NULL
    """,
}


def run_initial_eda(raw_dir: Path) -> dict:
    """
    Carga y valida los 9 CSVs crudos. Reporta shapes, nulos, duplicados
    en PKs e integridad referencial con DuckDB.

    Returns:
        Dict con shapes, null_summary, dup_summary, fk_summary y status.
    """
    raw_dir = Path(raw_dir)
    logger.info(f"EDA inicial — directorio raw: {raw_dir}")

    # 1. Verificar que todos los archivos existan
    missing = [name for name, fname in DATASETS.items() if not (raw_dir / fname).exists()]
    if missing:
        raise FileNotFoundError(f"CSVs no encontrados en {raw_dir}: {missing}")

    # 2. Cargar CSVs y reportar shapes
    data: Dict[str, pl.DataFrame] = {}
    shapes: Dict[str, tuple] = {}
    logger.info(f"\n  {'Dataset':<25} {'Filas':>10}  {'Columnas':>9}")
    logger.info("  " + "-" * 50)
    for name, fname in DATASETS.items():
        df = pl.read_csv(raw_dir / fname, infer_schema_length=10_000)
        data[name] = df
        shapes[name] = df.shape
        logger.info(f"  {name:<25} {df.shape[0]:>10,}  {df.shape[1]:>9}")
    total_filas = sum(s[0] for s in shapes.values())
    logger.info(f"\n  Total filas: {total_filas:,}")

    # 3. Analisis de nulos (solo columnas con al menos un nulo)
    null_summary: Dict[str, Dict[str, int]] = {}
    logger.info("\nNulos por dataset:")
    for name, df in data.items():
        col_nulls = {col: df[col].null_count() for col in df.columns if df[col].null_count() > 0}
        null_summary[name] = col_nulls
        if col_nulls:
            for col, n in col_nulls.items():
                pct = n / len(df) * 100
                logger.info(f"  [{name}] {col}: {n:,} ({pct:.2f}%)")
        else:
            logger.info(f"  [{name}] sin nulos")

    # 4. Duplicados exactos y en PKs
    dup_summary: Dict[str, Dict[str, int]] = {}
    logger.info(f"\n  {'Dataset':<25}  {'Exactos':>8}  {'Dup PK':>8}  Estado")
    logger.info("  " + "-" * 65)
    for name, df in data.items():
        n_exactos = len(df) - df.unique().height
        pk_dups   = 0
        if name in PRIMARY_KEYS:
            pk_dups = len(df) - df.select(PRIMARY_KEYS[name]).unique().height
        estado = "REVISAR" if (n_exactos > 0 or pk_dups > 0) else "OK"
        dup_summary[name] = {"exactos": n_exactos, "pk_dups": pk_dups}
        logger.info(f"  {name:<25}  {n_exactos:>8,}  {pk_dups:>8,}  {estado}")

    # 5. Integridad referencial con DuckDB
    con = duckdb.connect(":memory:")
    for name, df in data.items():
        con.register(name, df)

    fk_summary: Dict[str, int] = {}
    logger.info("\nIntegridad referencial (datos crudos):")
    total_huerfanos = 0
    for rel, query in FK_CHECKS.items():
        n = con.execute(query).fetchone()[0]
        fk_summary[rel] = n
        total_huerfanos += n
        estado = "OK" if n == 0 else f"PROBLEMA: {n:,} huerfanos"
        logger.info(f"  {rel}: {estado}")
    con.close()

    logger.info(f"\nTotal huerfanos en datos crudos: {total_huerfanos}")
    logger.info("EDA inicial completado — los problemas detectados seran corregidos en clean_raw_data.")

    return {
        "shapes":           shapes,
        "total_filas":      total_filas,
        "null_summary":     null_summary,
        "dup_summary":      dup_summary,
        "fk_summary":       fk_summary,
        "total_huerfanos":  total_huerfanos,
        "status":           "ok",
    }
