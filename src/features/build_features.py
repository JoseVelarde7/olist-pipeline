"""
Feature engineering de series temporales usando DuckDB.
Recibe un DataFrame de ordenes y retorna la tabla de features
lista para entrenar el modelo.
"""

from typing import Optional

import duckdb
import pandas as pd

# Periodo util para modelado (excluye ramp-up y cierre del dataset)
USEFUL_START = "2017-02-01"
TRAIN_END = "2018-02-28"
TEST_START = "2018-03-01"
TEST_END = "2018-08-31"

FEATURE_COLS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_std_3",
    "month_num",
    "quarter",
    "is_november",
    "month_index",
    "category_enc",
]


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega ordenes a nivel (mes, categoria) usando DuckDB SQL.

    Usa COUNT DISTINCT sobre order_id para evitar contar multiples
    items de la misma orden como demanda separada.

    Args:
        df: DataFrame con columnas order_id, order_purchase_timestamp,
            order_status, product_category_name

    Returns:
        DataFrame con columnas: year_month, product_category_name, demand
    """
    con = duckdb.connect()
    con.register("orders_raw", df)

    monthly = con.execute("""
        SELECT
            DATE_TRUNC('month', order_purchase_timestamp) AS year_month,
            product_category_name,
            COUNT(DISTINCT order_id)                      AS demand
        FROM orders_raw
        WHERE order_status = 'delivered'
          AND product_category_name IS NOT NULL
        GROUP BY 1, 2
        ORDER BY product_category_name, year_month
    """).df()

    monthly["year_month"] = pd.to_datetime(monthly["year_month"])
    return monthly


def build_lag_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Crea lag features y rolling features por categoria usando DuckDB window functions.

    Todas las features usan shift >= 1 respecto al mes a predecir:
    no hay data leakage. El target es la demanda del mes actual (demand),
    y todas las features provienen de meses anteriores.

    Args:
        monthly: DataFrame con columnas year_month, product_category_name, demand

    Returns:
        DataFrame con features de series temporales agregadas
    """
    con = duckdb.connect()
    con.register("monthly", monthly)

    # DuckDB no permite window functions anidadas.
    # Solucion: CTE en dos pasos — primero lags, luego rolling sobre los lags.
    features = con.execute("""
        WITH lagged AS (
            SELECT
                year_month,
                product_category_name,
                demand,
                LAG(demand, 1) OVER w AS lag_1,
                LAG(demand, 2) OVER w AS lag_2,
                LAG(demand, 3) OVER w AS lag_3,
                LAG(demand, 6) OVER w AS lag_6
            FROM monthly
            WINDOW w AS (
                PARTITION BY product_category_name
                ORDER BY year_month
            )
        ),
        rolling AS (
            SELECT
                *,
                AVG(lag_1) OVER (
                    PARTITION BY product_category_name
                    ORDER BY year_month
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) AS rolling_mean_3,
                AVG(lag_1) OVER (
                    PARTITION BY product_category_name
                    ORDER BY year_month
                    ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                ) AS rolling_mean_6,
                STDDEV_SAMP(lag_1) OVER (
                    PARTITION BY product_category_name
                    ORDER BY year_month
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) AS rolling_std_3
            FROM lagged
        )
        SELECT * FROM rolling
        ORDER BY product_category_name, year_month
    """).df()

    return features


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables temporales y encoding de categoria."""
    df = df.copy()
    df["month_num"] = df["year_month"].dt.month
    df["quarter"] = df["year_month"].dt.quarter
    df["is_november"] = (df["month_num"] == 11).astype(int)
    df["month_index"] = (
        (df["year_month"].dt.year - 2017) * 12 + df["year_month"].dt.month
    )

    categories = sorted(df["product_category_name"].unique())
    cat_map = {c: i for i, c in enumerate(categories)}
    df["category_enc"] = df["product_category_name"].map(cat_map)

    return df


def get_feature_table(
    df: pd.DataFrame,
    useful_start: Optional[str] = USEFUL_START,
) -> pd.DataFrame:
    """
    Pipeline completo: ordenes crudas → tabla de features lista para modelado.

    Pasos:
        1. Agrega a nivel mensual por categoria (DuckDB)
        2. Crea lag y rolling features (DuckDB window functions)
        3. Agrega variables temporales
        4. Filtra al periodo util y elimina filas con NaN en features criticas

    Args:
        df: DataFrame de ordenes entregadas (salida de extract.load_delivered_orders)
        useful_start: Fecha de inicio del periodo modelable (excluye ramp-up)

    Returns:
        DataFrame listo para entrenamiento con columnas FEATURE_COLS + ['demand']
    """
    monthly = aggregate_monthly(df)
    features = build_lag_features(monthly)
    features = add_temporal_features(features)

    # Filtrar periodo util y eliminar filas sin features completas
    result = (
        features
        .loc[features["year_month"] >= useful_start]
        .dropna(subset=FEATURE_COLS)
        .reset_index(drop=True)
    )

    return result
