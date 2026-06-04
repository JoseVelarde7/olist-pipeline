"""
Limpieza de datos Olist.
Extrae la logica del notebook 03_data_cleaning.ipynb en una funcion callable.
Lee de data/raw/, aplica 8 operaciones de limpieza y exporta a data/interim/cleaned_*.csv.
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

COLS_INT_IMPUTE   = ["product_name_lenght", "product_description_lenght", "product_photos_qty"]
COLS_FLOAT_IMPUTE = ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]

FK_CHECKS: Dict[str, str] = {
    "orders.customer_id -> customers": "SELECT COUNT(*) FROM orders o LEFT JOIN customers c ON o.customer_id = c.customer_id WHERE c.customer_id IS NULL",
    "order_items.order_id -> orders": "SELECT COUNT(*) FROM order_items oi LEFT JOIN orders o ON oi.order_id = o.order_id WHERE o.order_id IS NULL",
    "order_items.product_id -> products": "SELECT COUNT(*) FROM order_items oi LEFT JOIN products p ON oi.product_id = p.product_id WHERE p.product_id IS NULL",
    "order_items.seller_id -> sellers": "SELECT COUNT(*) FROM order_items oi LEFT JOIN sellers s ON oi.seller_id = s.seller_id WHERE s.seller_id IS NULL",
    "order_payments.order_id -> orders": "SELECT COUNT(*) FROM order_payments op LEFT JOIN orders o ON op.order_id = o.order_id WHERE o.order_id IS NULL",
    "order_reviews.order_id -> orders": "SELECT COUNT(*) FROM order_reviews orv LEFT JOIN orders o ON orv.order_id = o.order_id WHERE o.order_id IS NULL",
    "products.category -> category_translation": "SELECT COUNT(*) FROM products p LEFT JOIN category_translation ct ON p.product_category_name = ct.product_category_name WHERE p.product_category_name IS NOT NULL AND ct.product_category_name IS NULL",
}


def run_cleaning(raw_dir: Path, interim_dir: Path) -> dict:
    """
    Aplica las 8 operaciones de limpieza documentadas en 03_data_cleaning.ipynb:
      1. category_translation: agregar categoria 'outros'
      2. products: reasignar FK rotas, rellenar nulos, imputar numericos
      3. geolocation: deduplicacion exacta
      4. customers: estandarizacion ciudad/estado/zip
      5. sellers: estandarizacion ciudad/estado/zip
      6. orders: drop delivered-sin-fecha, imputar approved_at, std status
      7. order_items / order_payments / order_reviews: cascade + limpieza propia
      8. Verificacion FK post-limpieza con DuckDB

    Exporta los datasets limpios a interim_dir/cleaned_*.csv.

    Returns:
        Dict con transform_log, export_summary y status.
    """
    raw_dir     = Path(raw_dir)
    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)

    transform_log: List[dict] = []

    def log_op(dataset: str, operacion: str, n_antes: int, n_despues: int) -> None:
        delta = n_antes - n_despues
        pct   = delta / n_antes * 100 if n_antes > 0 else 0.0
        transform_log.append({
            "dataset": dataset, "operacion": operacion,
            "antes": n_antes, "despues": n_despues,
            "delta": delta, "pct": round(pct, 3),
        })
        logger.info(f"  [{dataset}] {operacion}: {n_antes:,} -> {n_despues:,}  (delta: {delta:,}  {pct:.2f}%)")

    # Carga de datos crudos
    raw: Dict[str, pl.DataFrame] = {}
    for name, fname in DATASETS.items():
        raw[name] = pl.read_csv(raw_dir / fname, infer_schema_length=10_000)
    logger.info("CSVs cargados: " + ", ".join(f"{k}({len(v):,})" for k, v in raw.items()))

    cleaned: Dict[str, pl.DataFrame] = {}

    # ------------------------------------------------------------------
    # 1. category_translation — agregar 'outros'
    # ------------------------------------------------------------------
    cat = raw["category_translation"].clone()
    if cat.filter(pl.col("product_category_name") == "outros").height == 0:
        cat = pl.concat([
            cat,
            pl.DataFrame({"product_category_name": ["outros"], "product_category_name_english": ["others"]}),
        ], how="vertical")
        logger.info("  [category_translation] 'outros' agregado")
    cleaned["category_translation"] = cat

    # ------------------------------------------------------------------
    # 2. products — FK rota + nulos categoria + imputacion numerica
    # ------------------------------------------------------------------
    prod = raw["products"].clone()
    n0   = len(prod)

    cats_validas = set(cleaned["category_translation"]["product_category_name"].to_list())

    prod = (
        prod
        .with_columns(
            pl.when(
                pl.col("product_category_name").is_not_null()
                & ~pl.col("product_category_name").is_in(list(cats_validas))
            )
            .then(pl.lit("outros"))
            .otherwise(pl.col("product_category_name"))
            .alias("product_category_name")
        )
        .with_columns(pl.col("product_category_name").fill_null(pl.lit("outros")))
    )

    medianas_globales = {col: prod[col].median() or 0.0 for col in COLS_INT_IMPUTE + COLS_FLOAT_IMPUTE}
    exprs_impute = []
    for col in COLS_INT_IMPUTE:
        med_global = int(round(medianas_globales[col]))
        exprs_impute.append(
            pl.col(col)
            .fill_null(pl.col(col).median().over("product_category_name").round(0))
            .fill_null(med_global)
            .cast(pl.Int64)
            .alias(col)
        )
    for col in COLS_FLOAT_IMPUTE:
        exprs_impute.append(
            pl.col(col)
            .fill_null(pl.col(col).median().over("product_category_name"))
            .fill_null(medianas_globales[col])
            .alias(col)
        )
    prod = prod.with_columns(exprs_impute)
    log_op("products", "FK rota + nulos categoria + imputacion numerica", n0, len(prod))
    cleaned["products"] = prod

    # ------------------------------------------------------------------
    # 3. geolocation — deduplicacion exacta
    # ------------------------------------------------------------------
    geo = raw["geolocation"].clone()
    n0  = len(geo)
    geo = geo.unique(maintain_order=True)
    log_op("geolocation", "deduplicacion exacta (todas las columnas)", n0, len(geo))
    cleaned["geolocation"] = geo

    # ------------------------------------------------------------------
    # 4. customers — estandarizacion
    # ------------------------------------------------------------------
    cust = raw["customers"].clone()
    n0   = len(cust)
    cust = cust.with_columns([
        pl.col("customer_city").str.strip_chars().str.to_titlecase().alias("customer_city"),
        pl.col("customer_state").str.strip_chars().str.to_uppercase().alias("customer_state"),
        pl.col("customer_zip_code_prefix").cast(pl.Utf8).str.zfill(5).alias("customer_zip_code_prefix"),
    ])
    log_op("customers", "estandarizacion ciudad/estado/zip", n0, len(cust))
    cleaned["customers"] = cust

    # ------------------------------------------------------------------
    # 5. sellers — estandarizacion
    # ------------------------------------------------------------------
    sell = raw["sellers"].clone()
    n0   = len(sell)
    sell = sell.with_columns([
        pl.col("seller_city").str.strip_chars().str.to_titlecase().alias("seller_city"),
        pl.col("seller_state").str.strip_chars().str.to_uppercase().alias("seller_state"),
        pl.col("seller_zip_code_prefix").cast(pl.Utf8).str.zfill(5).alias("seller_zip_code_prefix"),
    ])
    log_op("sellers", "estandarizacion ciudad/estado/zip", n0, len(sell))
    cleaned["sellers"] = sell

    # ------------------------------------------------------------------
    # 6. orders — drop delivered-sin-fecha + imputar approved_at + std status
    # ------------------------------------------------------------------
    ord_ = raw["orders"].clone()
    n0   = len(ord_)

    mask_invalidas = (
        (pl.col("order_status") == "delivered")
        & pl.col("order_delivered_customer_date").is_null()
    )
    ORDERS_ELIMINADAS: List[str] = (
        ord_.filter(mask_invalidas).select("order_id").to_series().to_list()
    )
    ord_ = ord_.filter(~mask_invalidas)

    ord_ = (
        ord_
        .with_columns(
            pl.when(
                pl.col("order_approved_at").is_null()
                & (pl.col("order_status") != "canceled")
            )
            .then(pl.col("order_purchase_timestamp"))
            .otherwise(pl.col("order_approved_at"))
            .alias("order_approved_at")
        )
        .with_columns(
            pl.col("order_status").str.strip_chars().str.to_lowercase().alias("order_status")
        )
    )
    log_op("orders", "drop delivered-sin-fecha + imputar approved_at + std status", n0, len(ord_))
    cleaned["orders"] = ord_

    if ORDERS_ELIMINADAS:
        logger.info(f"  [orders] {len(ORDERS_ELIMINADAS)} order_ids eliminados — cascade a tablas hijas")

    # ------------------------------------------------------------------
    # 7. order_items — cascade + validacion precios
    # ------------------------------------------------------------------
    items = raw["order_items"].clone()
    if ORDERS_ELIMINADAS:
        items = items.filter(~pl.col("order_id").is_in(ORDERS_ELIMINADAS))
    n0 = len(items)
    log_op("order_items", "cascade + validacion precios (sin eliminaciones)", n0, len(items))
    cleaned["order_items"] = items

    # ------------------------------------------------------------------
    # 8. order_payments — cascade + estandarizacion
    # ------------------------------------------------------------------
    pay = raw["order_payments"].clone()
    if ORDERS_ELIMINADAS:
        pay = pay.filter(~pl.col("order_id").is_in(ORDERS_ELIMINADAS))
    n0  = len(pay)
    pay = pay.with_columns(
        pl.col("payment_type").str.strip_chars().str.to_lowercase().alias("payment_type")
    )
    log_op("order_payments", "cascade + estandarizacion payment_type", n0, len(pay))
    cleaned["order_payments"] = pay

    # ------------------------------------------------------------------
    # 9. order_reviews — cascade + deduplicacion PK review_id
    # ------------------------------------------------------------------
    rev = raw["order_reviews"].clone()
    if ORDERS_ELIMINADAS:
        rev = rev.filter(~pl.col("order_id").is_in(ORDERS_ELIMINADAS))
    n0  = len(rev)
    rev = (
        rev
        .sort("review_answer_timestamp", descending=True, nulls_last=True)
        .unique(subset=["review_id"], keep="first", maintain_order=True)
    )
    log_op("order_reviews", "cascade + deduplicacion PK review_id (mantener mas reciente)", n0, len(rev))
    cleaned["order_reviews"] = rev

    # ------------------------------------------------------------------
    # Verificacion FK post-limpieza con DuckDB
    # ------------------------------------------------------------------
    con = duckdb.connect(":memory:")
    for name, df in cleaned.items():
        con.register(name, df)

    logger.info("\nVerificacion FK post-limpieza:")
    total_huerfanos = 0
    for rel, query in FK_CHECKS.items():
        n = con.execute(query).fetchone()[0]
        total_huerfanos += n
        logger.info(f"  {rel}: {'OK' if n == 0 else f'PROBLEMA {n:,} huerfanos'}")
    con.close()

    if total_huerfanos > 0:
        raise RuntimeError(
            f"Integridad referencial fallida post-limpieza: {total_huerfanos} huerfanos. "
            "Revisar log para detalle."
        )
    logger.info("  Integridad referencial completa.")

    # ------------------------------------------------------------------
    # Exportar datasets limpios
    # ------------------------------------------------------------------
    export_summary: Dict[str, dict] = {}
    logger.info(f"\nExportando a {interim_dir}:")
    for name, df in cleaned.items():
        path_out = interim_dir / f"cleaned_{name}.csv"
        df.write_csv(path_out)
        size_kb = path_out.stat().st_size / 1024
        export_summary[name] = {"rows": len(df), "path": str(path_out)}
        logger.info(f"  {name:<25} {len(df):>10,} filas  ->  {path_out.name}  ({size_kb:,.0f} KB)")

    logger.info("Limpieza completada.")
    return {
        "transform_log":   transform_log,
        "export_summary":  export_summary,
        "status":          "ok",
    }
