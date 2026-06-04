"""
Carga de datos limpios a MariaDB.
Lee de data/interim/cleaned_*.csv y migra a la BD configurada via env vars o config/db_config.json.
Ejecuta DROP + CREATE DATABASE para garantizar inicio limpio (full refresh).
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
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        charset="utf8mb4",
        use_unicode=True,
    )
    if database:
        params["database"] = database
    return mysql.connector.connect(**params)


def _drop_and_recreate(config: dict) -> None:
    db   = config["database"]
    conn = _get_conn(config)
    cur  = conn.cursor()
    cur.execute(f"DROP DATABASE IF EXISTS `{db}`")
    cur.execute(f"CREATE DATABASE `{db}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    cur.close()
    conn.close()
    logger.info(f"BD '{db}' eliminada y recreada.")


def _apply_schema(conn, schema_file: Path) -> None:
    script = schema_file.read_text(encoding="utf-8")
    cur = conn.cursor()
    for stmt in script.split(";"):
        s = stmt.strip()
        if s:
            cur.execute(s)
    conn.commit()
    cur.close()
    logger.info(f"Esquema aplicado desde {schema_file.name}.")


def _batch_insert(conn, query: str, rows: list, table: str) -> int:
    if not rows:
        logger.warning(f"[{table}] sin filas para insertar.")
        return 0
    cur  = conn.cursor()
    n_ok = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        try:
            cur.executemany(query, batch)
            conn.commit()
            n_ok += len(batch)
        except Error as e:
            logger.warning(f"[{table}] error en batch {i // BATCH_SIZE + 1}: {e}")
    cur.close()
    logger.info(f"  {n_ok:>10,} / {len(rows):,} registros -> {table}")
    return n_ok


def _load_table(conn, df: pd.DataFrame, table: str) -> int:
    if table == "product_category_translation":
        rows = [(_n(r["product_category_name"]), _n(r["product_category_name_english"])) for _, r in df.iterrows()]
        return _batch_insert(conn, "INSERT INTO product_category_translation (product_category_name, product_category_name_english) VALUES (%s, %s)", rows, table)

    if table == "olist_geolocation":
        rows = [(str(r["geolocation_zip_code_prefix"]), _n(r["geolocation_lat"]), _n(r["geolocation_lng"]), _n(r["geolocation_city"]), _n(r["geolocation_state"])) for _, r in df.iterrows()]
        return _batch_insert(conn, "INSERT INTO olist_geolocation (geolocation_zip_code_prefix, geolocation_lat, geolocation_lng, geolocation_city, geolocation_state) VALUES (%s, %s, %s, %s, %s)", rows, table)

    if table == "olist_customers":
        rows = [(r["customer_id"], r["customer_unique_id"], str(r["customer_zip_code_prefix"]), _n(r["customer_city"]), _n(r["customer_state"])) for _, r in df.iterrows()]
        return _batch_insert(conn, "INSERT INTO olist_customers (customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state) VALUES (%s, %s, %s, %s, %s)", rows, table)

    if table == "olist_sellers":
        rows = [(r["seller_id"], str(r["seller_zip_code_prefix"]), _n(r["seller_city"]), _n(r["seller_state"])) for _, r in df.iterrows()]
        return _batch_insert(conn, "INSERT INTO olist_sellers (seller_id, seller_zip_code_prefix, seller_city, seller_state) VALUES (%s, %s, %s, %s)", rows, table)

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
        return _batch_insert(conn, "INSERT INTO olist_products (product_id, product_category_name, product_name_lenght, product_description_lenght, product_photos_qty, product_weight_g, product_length_cm, product_height_cm, product_width_cm) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", rows, table)

    if table == "olist_orders":
        rows = [(r["order_id"], r["customer_id"], _n(r["order_status"]), _n(r["order_purchase_timestamp"]), _n(r["order_approved_at"]), _n(r["order_delivered_carrier_date"]), _n(r["order_delivered_customer_date"]), _n(r["order_estimated_delivery_date"])) for _, r in df.iterrows()]
        return _batch_insert(conn, "INSERT INTO olist_orders (order_id, customer_id, order_status, order_purchase_timestamp, order_approved_at, order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", rows, table)

    if table == "olist_order_items":
        rows = [(r["order_id"], int(r["order_item_id"]), r["product_id"], r["seller_id"], _n(r["shipping_limit_date"]), float(r["price"]) if pd.notna(r["price"]) else None, float(r["freight_value"]) if pd.notna(r["freight_value"]) else None) for _, r in df.iterrows()]
        return _batch_insert(conn, "INSERT INTO olist_order_items (order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value) VALUES (%s, %s, %s, %s, %s, %s, %s)", rows, table)

    if table == "olist_order_payments":
        rows = [(r["order_id"], int(r["payment_sequential"]) if pd.notna(r["payment_sequential"]) else None, _n(r["payment_type"]), int(r["payment_installments"]) if pd.notna(r["payment_installments"]) else None, float(r["payment_value"]) if pd.notna(r["payment_value"]) else None) for _, r in df.iterrows()]
        return _batch_insert(conn, "INSERT INTO olist_order_payments (order_id, payment_sequential, payment_type, payment_installments, payment_value) VALUES (%s, %s, %s, %s, %s)", rows, table)

    if table == "olist_order_reviews":
        rows = [(r["review_id"], r["order_id"], int(r["review_score"]) if pd.notna(r["review_score"]) else None, _n(r["review_comment_title"]), _n(r["review_comment_message"]), _n(r["review_creation_date"]), _n(r["review_answer_timestamp"])) for _, r in df.iterrows()]
        return _batch_insert(conn, "INSERT INTO olist_order_reviews (review_id, order_id, review_score, review_comment_title, review_comment_message, review_creation_date, review_answer_timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s)", rows, table)

    raise ValueError(f"Tabla desconocida: {table}")


def run_migration(interim_dir: Path, schema_file: Path) -> dict:
    """
    Ejecuta migracion completa:
      1. DROP + CREATE DATABASE (inicio limpio garantizado)
      2. Aplicar schema.sql
      3. Cargar las 9 tablas desde cleaned_*.csv en orden de dependencias FK
      4. Verificar conteos finales

    La configuracion de conexion se lee desde env vars (Docker/Airflow)
    o config/db_config.json como fallback (uso local).

    Returns:
        Dict con conteos por tabla y status.
    """
    interim_dir = Path(interim_dir)
    schema_file = Path(schema_file)

    config = load_config()
    logger.info(f"Migracion -> BD '{config['database']}' en {config['host']}:{config['port']}")
    logger.info(f"Fuente: {interim_dir}")
    logger.info(f"Schema: {schema_file}")

    # Paso 1: drop + create BD
    _drop_and_recreate(config)

    # Paso 2: aplicar esquema
    conn = _get_conn(config, database=config["database"])
    _apply_schema(conn, schema_file)
    conn.close()

    # Paso 3: cargar tablas en orden FK
    conn   = _get_conn(config, database=config["database"])
    counts: dict = {}
    for csv_file, table in LOAD_ORDER:
        csv_path = interim_dir / csv_file
        if not csv_path.exists():
            logger.warning(f"Archivo no encontrado: {csv_path} — tabla '{table}' omitida")
            counts[table] = 0
            continue
        df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
        logger.info(f"Cargando {csv_file} -> {table}  ({len(df):,} filas)")
        n = _load_table(conn, df, table)
        counts[table] = n
    conn.close()

    # Paso 4: verificar conteos
    conn = _get_conn(config, database=config["database"])
    cur  = conn.cursor()
    logger.info("\nConteo final por tabla:")
    for _, table in LOAD_ORDER:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        n = cur.fetchone()[0]
        logger.info(f"  {table:<35} {n:>10,}")
    cur.close()
    conn.close()

    logger.info("Migracion completada.")
    return {"counts": counts, "status": "ok"}
