"""
Extraccion de datos desde MariaDB.
Lee la configuracion de conexion desde config/db_config.json
y expone funciones para cargar los datos del proyecto Olist.
"""

import json
import os
from pathlib import Path
from typing import Optional

import mysql.connector
import pandas as pd


def load_config(config_path: Optional[Path] = None) -> dict:
    """
    Lee la configuracion de conexion a MariaDB.
    Prioridad: variables de entorno (Docker/Airflow) > db_config.json (local).
    """
    host = os.getenv("OLIST_DB_HOST")
    if host:
        return {
            "host":     host,
            "port":     int(os.getenv("OLIST_DB_PORT", "3306")),
            "user":     os.getenv("OLIST_DB_USER", "teju_admin"),
            "password": os.getenv("OLIST_DB_PASSWORD", ""),
            "database": os.getenv("OLIST_DB_NAME", "olist"),
        }
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config" / "db_config.json"
    return json.loads(config_path.read_text())


def get_connection(config: Optional[dict] = None) -> mysql.connector.MySQLConnection:
    """Retorna una conexion activa a MariaDB."""
    if config is None:
        config = load_config()
    return mysql.connector.connect(**config)


def load_delivered_orders(
    conn: mysql.connector.MySQLConnection,
    categories: Optional[list] = None,
) -> pd.DataFrame:
    """
    Carga ordenes entregadas con su categoria de producto desde MariaDB.

    Hace JOIN entre olist_orders, olist_order_items y olist_products
    para obtener (order_id, timestamp, category) a nivel de item.
    El filtro de categorias es opcional — sin el, trae todas.

    Returns:
        DataFrame con columnas: order_id, order_purchase_timestamp,
        order_status, product_category_name
    """
    category_filter = ""
    if categories:
        placeholders = ", ".join(["%s"] * len(categories))
        category_filter = f"AND p.product_category_name IN ({placeholders})"

    sql = f"""
        SELECT
            o.order_id,
            o.order_purchase_timestamp,
            o.order_status,
            p.product_category_name
        FROM olist_orders o
        JOIN olist_order_items oi ON o.order_id = oi.order_id
        JOIN olist_products    p  ON oi.product_id = p.product_id
        WHERE o.order_status = 'delivered'
          AND p.product_category_name IS NOT NULL
          {category_filter}
    """

    params = categories if categories else []
    cursor = conn.cursor()
    cursor.execute(sql, params)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    cursor.close()

    df = pd.DataFrame(rows, columns=columns)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    return df
