import pandas as pd
geo = pd.read_csv("olist_geolocation_dataset.csv")
print(geo.duplicated().sum())         # duplicados exactos
print(geo.duplicated(subset=["geolocation_zip_code_prefix"]).sum())