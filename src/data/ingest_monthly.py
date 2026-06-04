"""
Ingestion incremental mensual.
Lee CSVs de data/monthly/ (mismo formato que data/raw/), valida su existencia
y agrega los registros nuevos a MariaDB usando INSERT IGNORE para evitar duplicados.
"""
import logging
from pathlib import Path
from typing import Optional

import mysql.connector
import pandas as pd
from mysql.connector import Error

from src.data.extract import load_config

logger = logging.getLogger(__name__)

BATCH_SIZE = 2000

DATASETS = {
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

# Orden de carga respetando dependencias FK
LOAD_ORDER = [
    ("cleaned_category_translation.csv", "product_category_translation"),
    ("cleaned_geolocation.csv",           "olist_geolocation"),
    ("cleaned_customers.csv",             "olist_customers"),
    ("cleaned_sellers.csv",               "olist_sellers"),
    ("cleaned_products.csv",              "olist_products"),
    ("cleaned_orders.csv",                "olist_orders"),
    ("cleaned_order_items.csv",           "olist_order_items"),
    ("cleaned_order_payments.csv",        "olist_order_payments"),
    ("cleaned_order_reviews.csv",         "olist_order_reviews"),
]


def _n(val):
    return None if pd.isna(val) else val


def _get_conn(config: dict, database: Optional[str] = None) -> mysql.connector.MySQLConnection:
    params = dict(
        host=config["host"], port=config["port"],
        user=config["user"], password=config["password"],
        charset="utf8mb4", use_unicode=True,
    )
    if database:
        params["database"] = database
    return mysql.connector.connect(**params)


def _batch_insert_ignore(conn, query: str, rows: list, table: str) -> int:
    """INSERT IGNORE — silently skips records that already exist (same PK)."""
    if not rows:
        return 0
    ignore_query = query.replace("INSERT INTO", "INSERT IGNORE INTO", 1)
    cur  = conn.cursor()
    n_ok = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        try:
            cur.executemany(ignore_query, batch)
            conn.commit()
            n_ok += cur.rowcount  # INSERT IGNORE reporta solo los filas realmente insertadas
        except Error as e:
            logger.warning(f"[{table}] error en batch {i // BATCH_SIZE + 1}: {e}")
    cur.close()
    logger.info(f"  {n_ok:>10,} nuevos registros -> {table}")
    return n_ok


def _append_table(conn, df: pd.DataFrame, table: str) -> int:
    if table == "product_category_translation":
        rows = [(_n(r["product_category_name"]), _n(r["product_category_name_english"])) for _, r in df.iterrows()]
        return _batch_insert_ignore(conn, "INSERT INTO product_category_translation (product_category_name, product_category_name_english) VALUES (%s, %s)", rows, table)

    if table == "olist_geolocation":
        rows = [(str(r["geolocation_zip_code_prefix"]), _n(r["geolocation_lat"]), _n(r["geolocation_lng"]), _n(r["geolocation_city"]), _n(r["geolocation_state"])) for _, r in df.iterrows()]
        return _batch_insert_ignore(conn, "INSERT INTO olist_geolocation (geolocation_zip_code_prefix, geolocation_lat, geolocation_lng, geolocation_city, geolocation_state) VALUES (%s, %s, %s, %s, %s)", rows, table)

    if table == "olist_customers":
        rows = [(r["customer_id"], r["customer_unique_id"], str(r["customer_zip_code_prefix"]), _n(r["customer_city"]), _n(r["customer_state"])) for _, r in df.iterrows()]
        return _batch_insert_ignore(conn, "INSERT INTO olist_customers (customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state) VALUES (%s, %s, %s, %s, %s)", rows, table)

    if table == "olist_sellers":
        rows = [(r["seller_id"], str(r["seller_zip_code_prefix"]), _n(r["seller_city"]), _n(r["seller_state"])) for _, r in df.iterrows()]
        return _batch_insert_ignore(conn, "INSERT INTO olist_sellers (seller_id, seller_zip_code_prefix, seller_city, seller_state) VALUES (%s, %s, %s, %s)", rows, table)

    if table == "olist_products":
        rows = []
        for _, r in df.iterrows():
            rows.append((
                r["product_id"], _n(r["product_category_name"]),
                int(r["product_name_lenght"])        if pd.notna(r["product_name_lenght"])        else None,
                int(r["product_description_lenght"]) if pd.notna(r["product_description_lenght"]) else None,
                int(r["product_photos_qty"])          if pd.notna(r["product_photos_qty"])          else None,
                int(r["product_weight_g"])            if pd.notna(r["product_weight_g"])            else None,
                int(r["product_length_cm"])           if pd.notna(r["product_length_cm"])           else None,
                int(r["product_height_cm"])           if pd.notna(r["product_height_cm"])           else None,
                int(r["product_width_cm"])            if pd.notna(r["product_width_cm"])            else None,
            ))
        return _batch_insert_ignore(conn, "INSERT INTO olist_products (product_id, product_category_name, product_name_lenght, product_description_lenght, product_photos_qty, product_weight_g, product_length_cm, product_height_cm, product_width_cm) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", rows, table)

    if table == "olist_orders":
        rows = [(r["order_id"], r["customer_id"], _n(r["order_status"]), _n(r["order_purchase_timestamp"]), _n(r["order_approved_at"]), _n(r["order_delivered_carrier_date"]), _n(r["order_delivered_customer_date"]), _n(r["order_estimated_delivery_date"])) for _, r in df.iterrows()]
        return _batch_insert_ignore(conn, "INSERT INTO olist_orders (order_id, customer_id, order_status, order_purchase_timestamp, order_approved_at, order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", rows, table)

    if table == "olist_order_items":
        rows = [(r["order_id"], int(r["order_item_id"]), r["product_id"], r["seller_id"], _n(r["shipping_limit_date"]), float(r["price"]) if pd.notna(r["price"]) else None, float(r["freight_value"]) if pd.notna(r["freight_value"]) else None) for _, r in df.iterrows()]
        return _batch_insert_ignore(conn, "INSERT INTO olist_order_items (order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value) VALUES (%s, %s, %s, %s, %s, %s, %s)", rows, table)

    if table == "olist_order_payments":
        rows = [(r["order_id"], int(r["payment_sequential"]) if pd.notna(r["payment_sequential"]) else None, _n(r["payment_type"]), int(r["payment_installments"]) if pd.notna(r["payment_installments"]) else None, float(r["payment_value"]) if pd.notna(r["payment_value"]) else None) for _, r in df.iterrows()]
        return _batch_insert_ignore(conn, "INSERT INTO olist_order_payments (order_id, payment_sequential, payment_type, payment_installments, payment_value) VALUES (%s, %s, %s, %s, %s)", rows, table)

    if table == "olist_order_reviews":
        rows = [(r["review_id"], r["order_id"], int(r["review_score"]) if pd.notna(r["review_score"]) else None, _n(r["review_comment_title"]), _n(r["review_comment_message"]), _n(r["review_creation_date"]), _n(r["review_answer_timestamp"])) for _, r in df.iterrows()]
        return _batch_insert_ignore(conn, "INSERT INTO olist_order_reviews (review_id, order_id, review_score, review_comment_title, review_comment_message, review_creation_date, review_answer_timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s)", rows, table)

    raise ValueError(f"Tabla desconocida: {table}")


def validate_monthly_csvs(monthly_dir: Path) -> dict:
    """
    Verifica que los CSVs mensuales existan en monthly_dir y reporta sus shapes.
    No todos los archivos son obligatorios — al menos olist_orders_dataset.csv debe estar presente.

    Returns:
        Dict con shapes de los CSVs encontrados y lista de faltantes.
    """
    import polars as pl

    monthly_dir = Path(monthly_dir)
    found    = {}
    missing  = []

    for name, fname in DATASETS.items():
        path = monthly_dir / fname
        if path.exists():
            df = pl.read_csv(path, infer_schema_length=10_000)
            found[name] = df.shape
            logger.info(f"  [monthly] {name:<25} {df.shape[0]:>10,} filas")
        else:
            missing.append(fname)
            logger.info(f"  [monthly] {name:<25} no encontrado (opcional)")

    if "orders" not in found:
        raise FileNotFoundError(
            f"olist_orders_dataset.csv no encontrado en {monthly_dir}. "
            "Es el archivo minimo requerido para la ingestion mensual."
        )

    logger.info(f"CSVs mensuales encontrados: {len(found)} / {len(DATASETS)}")
    return {"found": found, "missing": missing, "monthly_dir": str(monthly_dir)}


def append_to_db(interim_monthly_dir: Path) -> dict:
    """
    Agrega los registros de los CSVs limpios mensuales a MariaDB.
    Usa INSERT IGNORE para evitar duplicados — si un registro ya existe (mismo PK),
    se salta silenciosamente sin error.

    Returns:
        Dict con conteo de registros nuevos insertados por tabla.
    """
    interim_monthly_dir = Path(interim_monthly_dir)
    config = load_config()
    conn   = _get_conn(config, database=config["database"])

    counts: dict = {}
    logger.info(f"Agregando datos mensuales a BD '{config['database']}'...")

    for csv_file, table in LOAD_ORDER:
        csv_path = interim_monthly_dir / csv_file
        if not csv_path.exists():
            logger.info(f"  {csv_file} no encontrado — tabla '{table}' omitida")
            continue
        df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
        logger.info(f"Procesando {csv_file} -> {table}  ({len(df):,} registros en CSV)")
        n = _append_table(conn, df, table)
        counts[table] = n

    conn.close()
    total_new = sum(counts.values())
    logger.info(f"Ingestion mensual completada. Nuevos registros: {total_new:,}")
    return {"counts": counts, "total_new": total_new, "status": "ok"}
