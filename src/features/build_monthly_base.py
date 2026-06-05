"""
Construye la base mensual categoría-mes y el target demand_next_month.
Réplica del notebook 02_build_monthly_base_olist.ipynb como módulo importable.
"""

import pandas as pd


def build_monthly_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra órdenes delivered, crea year_month y agrega a nivel (categoría, mes).
    Produce las mismas columnas del notebook 02 del equipo.
    """
    df = df[df["order_status"] == "delivered"].copy()
    df = df.dropna(subset=["product_category_name"])
    df["year_month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()

    monthly = df.groupby(["year_month", "product_category_name"]).agg(
        total_orders  =("order_id",      "nunique"),
        total_revenue =("price",         "sum"),
        avg_price     =("price",         "mean"),
        total_freight =("freight_value", "sum"),
        avg_freight   =("freight_value", "mean"),
        total_payment =("payment_value", "sum"),
        avg_payment   =("payment_value", "mean"),
    ).reset_index()

    monthly["demand"] = monthly["total_orders"]
    return monthly


def fill_missing_months(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Rellena meses faltantes por categoría con demanda 0.
    Garantiza series temporales continuas para ARIMA.
    """
    all_months = pd.date_range(
        start=monthly["year_month"].min(),
        end=monthly["year_month"].max(),
        freq="MS",
    )
    categories = monthly["product_category_name"].unique()
    idx = pd.MultiIndex.from_product(
        [all_months, categories],
        names=["year_month", "product_category_name"],
    )
    template = pd.DataFrame(index=idx).reset_index()
    merged = template.merge(monthly, on=["year_month", "product_category_name"], how="left")

    fill_zero = ["demand", "total_orders", "total_revenue",
                 "avg_price", "total_freight", "avg_freight",
                 "total_payment", "avg_payment"]
    merged[fill_zero] = merged[fill_zero].fillna(0)
    return merged


def create_demand_target(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Crea demand_next_month = shift(-1) por categoría.
    Elimina la última fila de cada categoría (target NaN).
    """
    monthly = monthly.sort_values(["product_category_name", "year_month"]).copy()
    monthly["demand_next_month"] = (
        monthly.groupby("product_category_name")["demand"].shift(-1)
    )
    return monthly.dropna(subset=["demand_next_month"]).copy()


def get_monthly_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo: master table → base mensual con target demand_next_month.
    Equivalente al main() del notebook 02_build_monthly_base_olist.ipynb.
    """
    monthly = build_monthly_aggregation(df)
    monthly = fill_missing_months(monthly)
    monthly = create_demand_target(monthly)
    return monthly
