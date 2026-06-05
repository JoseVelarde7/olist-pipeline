"""
Extrae la master table transaccional desde MariaDB.
Replica la lógica de cleaning_eda_olist_mastertable.py partiendo
de los datos normalizados ya cargados en la base de datos.
"""

import numpy as np
import pandas as pd

from src.data.extract import get_connection

_MASTER_SQL = """
SELECT
    o.order_id,
    o.customer_id,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    c.customer_unique_id,
    c.customer_zip_code_prefix,
    c.customer_city,
    c.customer_state,
    p.product_id,
    p.product_category_name,
    p.product_name_lenght,
    p.product_description_lenght,
    p.product_photos_qty,
    p.product_weight_g,
    p.product_length_cm,
    p.product_height_cm,
    p.product_width_cm,
    oi.order_item_id,
    oi.seller_id,
    oi.shipping_limit_date,
    oi.price,
    oi.freight_value,
    s.seller_zip_code_prefix,
    s.seller_city,
    s.seller_state,
    op.payment_sequential,
    op.payment_type,
    op.payment_installments,
    op.payment_value,
    r.review_id,
    r.review_score,
    r.review_comment_title,
    r.review_comment_message,
    r.review_creation_date,
    r.review_answer_timestamp
FROM olist_orders o
JOIN olist_order_items oi ON o.order_id = oi.order_id
JOIN olist_products    p  ON oi.product_id = p.product_id
JOIN olist_customers   c  ON o.customer_id = c.customer_id
LEFT JOIN olist_sellers s ON oi.seller_id = s.seller_id
LEFT JOIN (
    SELECT order_id, payment_sequential, payment_type,
           payment_installments, payment_value
    FROM olist_order_payments
    WHERE payment_sequential = 1
) op ON o.order_id = op.order_id
LEFT JOIN (
    SELECT order_id, review_id, review_score,
           review_comment_title, review_comment_message,
           review_creation_date, review_answer_timestamp
    FROM olist_order_reviews
    WHERE review_id IN (
        SELECT MAX(review_id) FROM olist_order_reviews GROUP BY order_id
    )
) r ON o.order_id = r.order_id
"""

_DATE_COLS = [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date", "shipping_limit_date",
    "review_creation_date", "review_answer_timestamp",
]


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["order_purchase_timestamp"]
    return df.assign(
        purchase_year      = ts.dt.year,
        purchase_month     = ts.dt.month,
        purchase_day       = ts.dt.day,
        purchase_dayofweek = ts.dt.dayofweek,
        purchase_quarter   = ts.dt.quarter,
        purchase_week      = ts.dt.isocalendar().week.astype(int),
        purchase_hour      = ts.dt.hour,
    )


def _add_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        delivery_days = (
            (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
        ),
        estimated_delivery_days = (
            (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.days
        ),
    )


def _add_product_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        product_volume_cm3=(
            df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
        )
    )


def _add_value_features(df: pd.DataFrame) -> pd.DataFrame:
    price_safe = df["price"].replace(0, np.nan)
    return df.assign(
        total_order_value   = df["price"] + df["freight_value"],
        freight_ratio       = (df["freight_value"] / price_safe).round(4),
        payment_price_ratio = (df["payment_value"] / price_safe).round(4),
    )


def _add_review_features(df: pd.DataFrame) -> pd.DataFrame:
    msg_len = df["review_comment_message"].str.len().where(
        df["review_comment_message"].notna(), other=np.nan
    )
    sentiment = pd.cut(
        df["review_score"],
        bins=[0, 2, 3, 5],
        labels=["negative", "neutral", "positive"],
        right=True,
    ).astype(object)
    return df.assign(review_message_length=msg_len, review_sentiment=sentiment)


def _add_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        installment_profile=pd.cut(
            df["payment_installments"],
            bins=[0, 6, float("inf")],
            labels=["low_installments", "high_installments"],
            right=True,
        ).astype(object),
        freight_profile=pd.cut(
            df["freight_ratio"],
            bins=[float("-inf"), 0.10, 0.60, float("inf")],
            labels=["low_freight", "normal_freight", "high_freight"],
            right=True,
        ).astype(object),
        product_segment="standard_product",
        delivery_speed=pd.cut(
            df["delivery_days"],
            bins=[float("-inf"), 7, float("inf")],
            labels=["fast_delivery", "slow_delivery"],
            right=True,
        ).astype(object),
        weight_segment=pd.cut(
            df["product_weight_g"],
            bins=[0, 1000, 5000, float("inf")],
            labels=["light_product", "medium_product", "heavy_product"],
            right=True,
        ).astype(object),
        price_segment=pd.cut(
            df["price"],
            bins=[0, 100, 500, float("inf")],
            labels=["low_ticket", "mid_ticket", "high_ticket"],
            right=True,
        ).astype(object),
    )


def extract_master_table(conn) -> pd.DataFrame:
    """
    Extrae y une todas las tablas desde MariaDB y computa las 22 variables
    derivadas de cleaning_eda_olist_mastertable.py.
    """
    cursor = conn.cursor()
    cursor.execute(_MASTER_SQL)
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    cursor.close()

    df = pd.DataFrame(rows, columns=cols)

    # MariaDB devuelve columnas DECIMAL como decimal.Decimal — convertir a float
    _DECIMAL_COLS = [
        "price", "freight_value", "payment_value",
        "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm",
    ]
    for col in _DECIMAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in _DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = _add_temporal_features(df)
    df = _add_delivery_features(df)
    df = _add_product_features(df)
    df = _add_value_features(df)
    df = _add_review_features(df)
    df = _add_segment_features(df)

    return df


def clean_master_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Limpieza Stage 1 de cleaning_eda_olist_mastertable.py:
    estandariza texto (strip + normalización de NaN) y elimina duplicados exactos.
    """
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": None, "None": None, "": None})

    null_by_col = {
        col: int(df[col].isnull().sum())
        for col in df.columns
        if df[col].isnull().sum() > 0
    }
    n_before = len(df)
    df = df.drop_duplicates().copy()

    return df, {"null_by_col": null_by_col, "duplicates_removed": n_before - len(df)}
