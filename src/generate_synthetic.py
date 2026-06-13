import uuid, random, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

RAW     = Path("/opt/airflow/data/raw")
MONTHLY = Path("/opt/airflow/data/monthly")
MONTHLY.mkdir(exist_ok=True)

products  = pd.read_csv(RAW / "olist_products_dataset.csv")
sellers   = pd.read_csv(RAW / "olist_sellers_dataset.csv")
geo       = pd.read_csv(RAW / "olist_geolocation_dataset.csv")

product_ids = products["product_id"].tolist()
seller_ids  = sellers["seller_id"].tolist()
zip_codes   = geo["geolocation_zip_code_prefix"].unique().tolist()
states      = ["SP","RJ","MG","RS","PR","SC","BA","GO","PE","CE"]
state_w     = [0.42,0.13,0.12,0.06,0.05,0.04,0.04,0.03,0.03,0.08]
pay_types   = ["credit_card","boleto","voucher","debit_card"]
pay_w       = [0.74, 0.19, 0.06, 0.01]

N_ORDERS = 500

def uid():
    return uuid.uuid4().hex

def rand_date(start, end):
    delta = (end - start).total_seconds()
    return start + timedelta(seconds=random.random() * delta)

SEP_START = datetime(2018, 9, 1,  6, 0, 0)
SEP_END   = datetime(2018, 9, 30, 23, 59, 59)

# 1. ORDERS
print("Generando ordenes...")
orders_rows = []
for _ in range(N_ORDERS):
    purchase  = rand_date(SEP_START, SEP_END)
    approved  = purchase + timedelta(hours=random.uniform(0.2, 24))
    carrier   = approved + timedelta(days=random.uniform(1, 5))
    delivery  = carrier  + timedelta(days=random.uniform(2, 10))
    estimated = purchase + timedelta(days=random.randint(10, 30))
    orders_rows.append({
        "order_id":                     uid(),
        "customer_id":                  uid(),
        "order_status":                 "delivered",
        "order_purchase_timestamp":     purchase.strftime("%Y-%m-%d %H:%M:%S"),
        "order_approved_at":            approved.strftime("%Y-%m-%d %H:%M:%S"),
        "order_delivered_carrier_date": carrier.strftime("%Y-%m-%d %H:%M:%S"),
        "order_delivered_customer_date":delivery.strftime("%Y-%m-%d %H:%M:%S"),
        "order_estimated_delivery_date":estimated.strftime("%Y-%m-%d 00:00:00"),
    })

df_orders = pd.DataFrame(orders_rows)
df_orders.to_csv(MONTHLY / "olist_orders_dataset.csv", index=False)
print(f"  olist_orders_dataset.csv         — {len(df_orders)} filas")

# 2. CUSTOMERS
print("Generando clientes...")
cust_rows = []
for row in orders_rows:
    state = np.random.choice(states, p=state_w)
    cust_rows.append({
        "customer_id":              row["customer_id"],
        "customer_unique_id":       uid(),
        "customer_zip_code_prefix": str(random.choice(zip_codes)),
        "customer_city":            "sao paulo",
        "customer_state":           state,
    })
df_customers = pd.DataFrame(cust_rows)
df_customers.to_csv(MONTHLY / "olist_customers_dataset.csv", index=False)
print(f"  olist_customers_dataset.csv      — {len(df_customers)} filas")

# 3. ORDER ITEMS
print("Generando items...")
items_rows = []
for row in orders_rows:
    n_items = np.random.choice([1,2,3,4], p=[0.65, 0.22, 0.09, 0.04])
    for i in range(1, n_items + 1):
        purchase_dt    = datetime.strptime(row["order_purchase_timestamp"], "%Y-%m-%d %H:%M:%S")
        shipping_limit = purchase_dt + timedelta(days=random.randint(2, 7))
        price   = round(float(np.random.lognormal(mean=4.3, sigma=0.9)), 2)
        price   = max(5.0, min(price, 2000.0))
        freight = round(random.uniform(8, 50), 2)
        items_rows.append({
            "order_id":           row["order_id"],
            "order_item_id":      i,
            "product_id":         random.choice(product_ids),
            "seller_id":          random.choice(seller_ids),
            "shipping_limit_date":shipping_limit.strftime("%Y-%m-%d %H:%M:%S"),
            "price":              price,
            "freight_value":      freight,
        })
df_items = pd.DataFrame(items_rows)
df_items.to_csv(MONTHLY / "olist_order_items_dataset.csv", index=False)
print(f"  olist_order_items_dataset.csv    — {len(df_items)} filas")

# 4. ORDER PAYMENTS
print("Generando pagos...")
order_totals = df_items.groupby("order_id")[["price","freight_value"]].sum()
pays_rows = []
for order_id, row in order_totals.iterrows():
    total = round(float(row["price"] + row["freight_value"]), 2)
    ptype = np.random.choice(pay_types, p=pay_w)
    installments = 1 if ptype != "credit_card" else int(np.random.choice(
        [1,2,3,4,6,8,10,12], p=[0.40,0.15,0.12,0.10,0.09,0.06,0.05,0.03]))
    pays_rows.append({
        "order_id":             order_id,
        "payment_sequential":   1,
        "payment_type":         ptype,
        "payment_installments": installments,
        "payment_value":        total,
    })
df_pays = pd.DataFrame(pays_rows)
df_pays.to_csv(MONTHLY / "olist_order_payments_dataset.csv", index=False)
print(f"  olist_order_payments_dataset.csv — {len(df_pays)} filas")

# 5. ORDER REVIEWS (70% de las ordenes)
print("Generando resenas...")
scores_w = [0.03, 0.05, 0.08, 0.20, 0.64]
rev_rows = []
for row in orders_rows:
    if random.random() > 0.70:
        continue
    delivery_dt = datetime.strptime(row["order_delivered_customer_date"], "%Y-%m-%d %H:%M:%S")
    creation    = delivery_dt + timedelta(days=random.randint(1, 7))
    answer      = creation   + timedelta(hours=random.uniform(12, 72))
    rev_rows.append({
        "review_id":               uid(),
        "order_id":                row["order_id"],
        "review_score":            int(np.random.choice([1,2,3,4,5], p=scores_w)),
        "review_comment_title":    None,
        "review_comment_message":  None,
        "review_creation_date":    creation.strftime("%Y-%m-%d %H:%M:%S"),
        "review_answer_timestamp": answer.strftime("%Y-%m-%d %H:%M:%S"),
    })
df_reviews = pd.DataFrame(rev_rows)
df_reviews.to_csv(MONTHLY / "olist_order_reviews_dataset.csv", index=False)
print(f"  olist_order_reviews_dataset.csv  — {len(df_reviews)} filas")

print("\n=== Resumen ===")
print(f"Ordenes:  {len(df_orders)}")
print(f"Items:    {len(df_items)}")
print(f"Pagos:    {len(df_pays)}")
print(f"Resenas:  {len(df_reviews)}")
print(f"Clientes: {len(df_customers)}")
print("\nCSVs listos en data/monthly/ para correr mode=monthly")
