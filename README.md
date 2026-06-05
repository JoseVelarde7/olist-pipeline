# Pipeline de Pronóstico de Demanda — Dataset Olist

Proyecto desarrollado para el Diplomado en Ciencia de Datos (Maestría). El objetivo principal es construir un pipeline automatizado de extremo a extremo que, a partir del dataset público de Olist, genere pronósticos mensuales de demanda por categoría de producto.

---

## Contexto y Objetivo

Olist es una plataforma de comercio electrónico brasileña que conecta vendedores con marketplaces. El dataset público disponible en Kaggle cubre transacciones entre 2016 y 2018, con información de órdenes, productos, clientes, vendedores, pagos y reseñas.

El proyecto busca responder: ¿es posible pronosticar la demanda mensual de cada categoría de productos con suficiente precisión para apoyar decisiones de inventario y logística? Para esto se implementó un pipeline orquestado con Apache Airflow que automatiza la carga de datos, el preprocesamiento, el entrenamiento de modelos ARIMA por categoría y el registro de experimentos en MLflow.

Se optó por ARIMA sobre modelos tabulares (como LightGBM) porque la demanda mensual por categoría exhibe dependencia temporal de corto plazo y el tamaño del dataset —alrededor de 30 meses por serie— no justifica la cantidad de parámetros de un modelo de árbol. ARIMA(1,1,1) es lo suficientemente flexible para capturar tendencia y autocorrelación sin sobreajustar.

---

## Arquitectura del Pipeline

```
[data/raw/]          CSVs originales del dataset Olist
      │
      ▼
  run_eda            Análisis exploratorio inicial
      │
      ▼
  clean_raw_data     Limpieza con Polars (9 tablas)
      │
      ▼
  load_to_db         Migración a MariaDB (schema.sql)
      │
      ▼
  build_master_table JOIN de las 8 tablas + 22 variables derivadas
      │
      ▼
  build_monthly_base Agregación mensual por categoría → target demand_next_month
      │
      ▼
  train_arima        ARIMA(1,1,1) por categoría — train / validation / backtest
      │
      ▼
  save_report        Métricas a reports/metrics/ + log en MLflow
```

El DAG admite dos modos de ejecución: carga inicial completa (trigger sin configuración) y actualización mensual incremental (`{"mode": "monthly"}`).

**Splits temporales:**
- Train: hasta 2017-12-01
- Validation: 2018-01-01 → 2018-05-01
- Backtest: 2018-05-02 → 2018-08-31 (modelo re-entrenado sobre train + val)

---

## Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Orquestación | Apache Airflow 2.9.0 (LocalExecutor) |
| Base de datos | MariaDB (existente en el host, puerto 3306) |
| Metadata Airflow | PostgreSQL 15 (contenedor Docker) |
| Tracking de experimentos | MLflow 2.20.0 |
| Modelo | ARIMA(1,1,1) via statsmodels |
| Preprocesamiento | Polars, Pandas |
| SQL analítico | DuckDB |
| Contenerización | Docker + Docker Compose |

---

## Resultados

El modelo se evaluó sobre 72 categorías de productos. La métrica principal es MAPE (Mean Absolute Percentage Error).

| Split | Modelo | MAPE promedio |
|---|---|---|
| Validation | ARIMA(1,1,1) | 62.17% |
| Validation | Naive (lag 1) | 40.58% |
| Backtest | ARIMA(1,1,1) | 61.28% |
| Backtest | Naive (lag 1) | 38.15% |

En promedio agregado, el baseline naive supera a ARIMA. Sin embargo, esto esconde variación considerable: categorías con series más estables y mayor volumen (por ejemplo, `beleza_saude`, `esporte_lazer`, `utilidades_domesticas`) obtienen MAPE en backtest por debajo del 10% con ARIMA. Las categorías con series muy cortas o alta volatilidad (moda, electrónica de nicho) degradan significativamente el promedio global.

Los resultados detallados por categoría están en `reports/metrics/arima_results_by_category.csv`.

---

## Requisitos Previos

- Docker Desktop (Windows/Mac) o Docker Engine (Linux)
- MariaDB corriendo en el host con la base de datos `olist` ya creada
- Dataset Olist (CSVs originales) disponibles localmente
- Git

No se requiere Python local; todo corre dentro de los contenedores de Airflow.

---

## Instalación y Configuración

### 1. Clonar el repositorio

```bash
git clone https://github.com/JoseVelarde7/olist-pipeline.git
cd olist-pipeline
```

### 2. Configurar las credenciales de MariaDB

Copiar el archivo de ejemplo y completar con las credenciales reales:

```bash
cp .env.example .env
```

Editar `.env`:

```
OLIST_DB_HOST=host.docker.internal
OLIST_DB_PORT=3306
OLIST_DB_USER=tu_usuario
OLIST_DB_PASSWORD=tu_contraseña
OLIST_DB_NAME=olist
```

> `host.docker.internal` resuelve automáticamente a la IP del host desde dentro de Docker. Si MariaDB corre en otro servidor, reemplazar por la IP correspondiente.

### 3. Copiar los datos raw

Colocar los CSVs del dataset Olist en `data/raw/`:

```
data/raw/
  olist_orders_dataset.csv
  olist_order_items_dataset.csv
  olist_order_payments_dataset.csv
  olist_order_reviews_dataset.csv
  olist_customers_dataset.csv
  olist_sellers_dataset.csv
  olist_products_dataset.csv
  olist_geolocation_dataset.csv
  product_category_name_translation.csv
```

### 4. Crear la base de datos en MariaDB

Si la base `olist` no existe todavía, ejecutar el schema:

```sql
CREATE DATABASE olist CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

```bash
mysql -u tu_usuario -p olist < config/schema.sql
```

### 5. Construir y levantar los contenedores

```bash
docker-compose up -d --build
```

La primera vez Docker descargará las imágenes y compilará `Dockerfile.mlflow`. Puede tardar varios minutos dependiendo de la conexión.

Verificar que los servicios estén corriendo:

```bash
docker-compose ps
```

Se espera ver cuatro servicios en estado `running`: `olist_airflow_postgres`, `olist_airflow_webserver`, `olist_airflow_scheduler`, `olist_mlflow`.

---

## Uso

### Interfaz de Airflow

Abrir `http://localhost:8080` en el navegador.

- Usuario: `admin`
- Contraseña: `admin123`

Buscar el DAG `olist_demand_pipeline`. Por defecto aparece pausado; activarlo con el toggle.

**Ejecutar carga inicial completa:**

Hacer clic en *Trigger DAG* sin configuración adicional (o con `{}`). Esto ejecutará EDA, limpieza, migración a MariaDB y entrenamiento ARIMA.

**Ejecutar actualización mensual incremental:**

Colocar los CSVs nuevos en `data/monthly/` y disparar el DAG con:

```json
{"mode": "monthly"}
```

**Forzar recarga completa:**

```json
{"force_reload": true}
```

### Tracking de experimentos en MLflow

Abrir `http://localhost:5000` para ver los runs registrados, los parámetros utilizados (orden ARIMA, fechas de corte) y las métricas por split.

---

## Estructura del Proyecto

```
olist-pipeline/
├── dags/
│   └── olist_demand_dag.py        # DAG principal con TaskFlow API
├── src/
│   ├── data/
│   │   ├── extract.py             # Conexión a MariaDB
│   │   ├── clean_data.py          # Limpieza con Polars (9 tablas)
│   │   ├── migrate.py             # Carga inicial a MariaDB
│   │   ├── ingest_monthly.py      # Ingesta incremental (INSERT IGNORE)
│   │   ├── run_eda.py             # Análisis exploratorio inicial
│   │   └── build_master_table.py  # JOIN completo + 22 variables derivadas
│   ├── features/
│   │   └── build_monthly_base.py  # Agregación mensual + target demand_next_month
│   └── models/
│       ├── arima_model.py         # ARIMA por categoría + integración MLflow
│       └── evaluate.py            # Métricas MAPE, RMSE, MAE
├── config/
│   ├── schema.sql                 # Schema MariaDB (9 tablas con FKs)
│   └── db_config.example.json
├── data/
│   ├── raw/                       # CSVs originales (no incluidos en el repo)
│   ├── interim/                   # Datos intermedios procesados
│   ├── monthly/                   # CSVs para actualizaciones incrementales
│   └── processed/
├── reports/
│   └── metrics/
│       ├── pipeline_summary.json
│       └── arima_results_by_category.csv
├── models/                        # Artefactos de modelo (generados en ejecución)
├── logs/
├── docker-compose.yml
├── Dockerfile.mlflow
├── pyproject.toml
└── .env.example
```

---

## Notas de Implementación

- Los datos raw no se incluyen en el repositorio por su tamaño. Se pueden obtener del [dataset público de Olist en Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
- El pipeline usa `INSERT IGNORE` en la ingesta incremental, por lo que datos ya existentes no generan errores ni duplicados.
- Para el backtest, el modelo ARIMA se re-entrena sobre train + validation antes de predecir, evitando filtración de información del futuro.
- Las series con menos de 6 meses de datos de entrenamiento se excluyen del modelo ARIMA pero sí aparecen en los resultados del baseline naive.
- Las dependencias de Python se instalan automáticamente en los contenedores de Airflow al iniciar (`_PIP_ADDITIONAL_REQUIREMENTS` en `docker-compose.yml`). No es necesario instalar nada manualmente.

---

## Equipo

Proyecto de Maestría — Diplomado en Ciencia de Datos
