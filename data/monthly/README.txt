Carpeta para datos mensuales incrementales.

Depositar aqui los mismos archivos CSV del dataset Olist pero con los registros del mes nuevo:

  olist_orders_dataset.csv          (obligatorio)
  olist_order_items_dataset.csv
  olist_order_payments_dataset.csv
  olist_order_reviews_dataset.csv
  olist_customers_dataset.csv
  olist_sellers_dataset.csv
  olist_products_dataset.csv
  olist_geolocation_dataset.csv
  product_category_name_translation.csv

Solo incluir los archivos que tengan datos nuevos. El pipeline limpiara e insertara
los registros usando INSERT IGNORE — los registros ya existentes en la BD se omiten
automaticamente sin error.

Para ejecutar el pipeline mensual desde la UI de Airflow:
  Trigger DAG with config: {"mode": "monthly"}

Para ejecutar el pipeline inicial (carga completa desde cero):
  Trigger DAG with config: {"mode": "initial"}
  o simplemente Trigger DAG sin config (modo por defecto).
