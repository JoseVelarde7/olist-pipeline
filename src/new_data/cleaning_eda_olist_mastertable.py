# -*- coding: utf-8 -*-
"""
cleaning_eda_olist_mastertable.py

Script de limpieza inicial y EDA (sin feature engineering)
Proyecto: Olist - Grupo 5

Objetivo:
1. Cargar la master table.
2. Realizar validaciones de calidad de datos.
3. Estandarizar tipos (especialmente fechas).
4. Generar salidas limpias y visualizaciones exploratorias.

NOTA:
- Este script NO crea features de modelado.
- Este script NO entrena modelos.
- Este script NO agrega variables de forecasting.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

pd.set_option('display.max_columns', None)
plt.style.use('default')
sns.set_theme(style='whitegrid')
RANDOM_STATE = 42

# Rutas (ajustables)
INPUT_PATH = Path('master1_olist_202605282344.csv')
OUTPUT_DIR = Path('output_eda')
FIGURES_DIR = OUTPUT_DIR / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Columnas esperadas de fecha
DATE_COLUMNS = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]

IMPORTANT_NUMERIC = ['price', 'freight_value', 'payment_value']
IMPORTANT_CATEGORICAL = ['product_category_name', 'review_score']


# ============================================================
# 2. FUNCIONES UTILITARIAS
# ============================================================

def load_data(path: Path) -> pd.DataFrame:
    """Carga el dataset desde CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f'No se encontró el archivo de entrada: {path.resolve()}\n'
            'Ajusta INPUT_PATH con el nombre/ruta correcta.'
        )
    df = pd.read_csv(path)
    print(f'✅ Dataset cargado correctamente: {path.name}')
    return df



def convert_dates(df: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    """Convierte columnas de fecha a datetime cuando existan."""
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    print('✅ Fechas convertidas correctamente')
    return df



def standardize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza ligera de texto en columnas object/category.
    No crea nuevas variables, solo estandariza contenido.
    """
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        # Evita alterar IDs complejos en exceso; solo trim de espacios
        df[col] = df[col].astype(str).str.strip()
        # Restaurar NaN reales donde quedaron como string 'nan'
        df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})
    print('✅ Texto estandarizado (trim de espacios y normalización básica de vacíos)')
    return df



def quality_report(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Genera tabla de nulos y conteo de duplicados."""
    null_count = df.isnull().sum()
    null_percent = ((null_count / len(df)) * 100).round(2)
    report = pd.DataFrame({
        'null_count': null_count,
        'null_percent': null_percent,
        'dtype': df.dtypes.astype(str)
    }).sort_values(['null_count', 'null_percent'], ascending=False)

    duplicates = int(df.duplicated().sum())
    return report, duplicates



def save_quality_outputs(df: pd.DataFrame, report: pd.DataFrame, duplicates: int) -> None:
    report.to_csv(OUTPUT_DIR / 'quality_report.csv', index=True)

    summary_rows = [
        ['n_rows', len(df)],
        ['n_columns', df.shape[1]],
        ['duplicates_full_row', duplicates]
    ]
    pd.DataFrame(summary_rows, columns=['metric', 'value']).to_csv(
        OUTPUT_DIR / 'dataset_summary.csv', index=False
    )



def remove_full_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().copy()
    after = len(df)
    print(f'✅ Duplicados exactos eliminados: {before - after}')
    return df



def print_overview(df: pd.DataFrame) -> None:
    print('\n' + '=' * 60)
    print('OVERVIEW GENERAL DEL DATASET')
    print('=' * 60)
    print(f'Shape: {df.shape}')
    print('\nColumnas:')
    print(df.columns.tolist())
    print('\nTipos de datos:')
    print(df.dtypes)



def save_clean_dataset(df: pd.DataFrame) -> None:
    output_path = OUTPUT_DIR / 'master_olist_clean_stage1.csv'
    df.to_csv(output_path, index=False)
    print(f'✅ Dataset limpio exportado: {output_path}')


# ============================================================
# 3. VISUALIZACIONES EDA
# ============================================================

def plot_numeric_distributions(df: pd.DataFrame) -> None:
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_columns:
        print('⚠️ No se encontraron columnas numéricas para histogramas.')
        return

    for col in numeric_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribución de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'hist_{col}.png', dpi=150)
        plt.close()



def plot_outliers(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[col])
            plt.title(f'Outliers de {col}')
            plt.xlabel(col)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f'boxplot_{col}.png', dpi=150)
            plt.close()



def plot_top_categories(df: pd.DataFrame) -> None:
    if 'product_category_name' not in df.columns:
        print('⚠️ No existe product_category_name para gráficos de categoría.')
        return

    top_order = df['product_category_name'].value_counts().head(15).index

    plt.figure(figsize=(12, 6))
    sns.countplot(data=df[df['product_category_name'].isin(top_order)],
                  x='product_category_name', order=top_order)
    plt.title('Top categorías por cantidad de registros')
    plt.xlabel('Categoría')
    plt.ylabel('Cantidad')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'top_categories_count.png', dpi=150)
    plt.close()

    revenue_by_category = (
        df.groupby('product_category_name', dropna=False)['price']
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )
    if not revenue_by_category.empty:
        plt.figure(figsize=(12, 6))
        revenue_by_category.plot(kind='bar')
        plt.title('Top categorías por revenue')
        plt.xlabel('Categoría')
        plt.ylabel('Revenue')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'top_categories_revenue.png', dpi=150)
        plt.close()



def plot_review_score(df: pd.DataFrame) -> None:
    if 'review_score' not in df.columns:
        print('⚠️ No existe review_score para análisis de reseñas.')
        return

    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x='review_score')
    plt.title('Distribución de review_score')
    plt.xlabel('Review score')
    plt.ylabel('Cantidad')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'review_score_distribution.png', dpi=150)
    plt.close()



def plot_monthly_trends(df: pd.DataFrame) -> None:
    if 'order_purchase_timestamp' not in df.columns:
        print('⚠️ No existe order_purchase_timestamp para series temporales.')
        return

    temp = df.dropna(subset=['order_purchase_timestamp']).copy()
    if temp.empty:
        print('⚠️ No hay datos válidos de fecha para análisis temporal.')
        return

    temp['year_month'] = temp['order_purchase_timestamp'].dt.to_period('M')
    temp['purchase_month'] = temp['order_purchase_timestamp'].dt.month

    monthly_demand = temp.groupby('year_month').size()
    monthly_demand.index = monthly_demand.index.to_timestamp()
    plt.figure(figsize=(14, 5))
    monthly_demand.plot()
    plt.title('Demanda mensual')
    plt.xlabel('Fecha')
    plt.ylabel('Cantidad de órdenes/registros')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'monthly_demand.png', dpi=150)
    plt.close()

    if 'price' in temp.columns:
        monthly_revenue = temp.groupby('year_month')['price'].sum()
        monthly_revenue.index = monthly_revenue.index.to_timestamp()
        plt.figure(figsize=(14, 5))
        monthly_revenue.plot()
        plt.title('Revenue mensual')
        plt.xlabel('Fecha')
        plt.ylabel('Revenue')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'monthly_revenue.png', dpi=150)
        plt.close()

    seasonality = temp.groupby('purchase_month').size().reindex(range(1, 13), fill_value=0)
    plt.figure(figsize=(10, 5))
    seasonality.plot(kind='bar')
    plt.title('Estacionalidad por mes')
    plt.xlabel('Mes')
    plt.ylabel('Cantidad de órdenes/registros')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'seasonality_by_month.png', dpi=150)
    plt.close()



def plot_correlation_matrix(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        print('⚠️ No hay suficientes columnas numéricas para correlación.')
        return

    corr = numeric_df.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title('Matriz de correlación')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'correlation_matrix.png', dpi=150)
    plt.close()


# ============================================================
# 4. FUNCIÓN PRINCIPAL
# ============================================================

def main() -> None:
    # Carga
    df = load_data(INPUT_PATH)

    # Overview inicial
    print_overview(df)

    # Limpieza inicial / stage 1
    df = convert_dates(df, DATE_COLUMNS)
    df = standardize_text(df)

    # Reporte de calidad antes de eliminar duplicados
    quality_before, duplicates_before = quality_report(df)
    print('\n' + '=' * 60)
    print('REPORTE DE CALIDAD - ANTES DE ELIMINAR DUPLICADOS')
    print('=' * 60)
    print(quality_before[quality_before['null_count'] > 0].head(20))
    print(f'\nDuplicados exactos: {duplicates_before}')

    # Eliminación de duplicados exactos
    df = remove_full_duplicates(df)

    # Reporte de calidad final
    quality_after, duplicates_after = quality_report(df)
    save_quality_outputs(df, quality_after, duplicates_after)

    print('\n' + '=' * 60)
    print('ESTADÍSTICAS DESCRIPTIVAS')
    print('=' * 60)
    print(df.describe(include='all', datetime_is_numeric=True).transpose().head(30))

    # Exportar dataset limpio (sin features nuevas)
    save_clean_dataset(df)

    # Visualizaciones EDA
    plot_numeric_distributions(df)
    plot_outliers(df, IMPORTANT_NUMERIC)
    plot_top_categories(df)
    plot_review_score(df)
    plot_monthly_trends(df)
    plot_correlation_matrix(df)

    print('\n✅ EDA y limpieza inicial completados correctamente.')
    print(f'📁 Resultados guardados en: {OUTPUT_DIR.resolve()}')
    print(f'🖼️ Figuras guardadas en: {FIGURES_DIR.resolve()}')


if __name__ == '__main__':
    main()
