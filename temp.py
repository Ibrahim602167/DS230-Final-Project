import pandas as pd

aisles = pd.read_csv("aisles (1).csv")
departments = pd.read_csv("departments (1).csv")

orders = pd.read_csv("orders.csv (1).zip")
order_products_prior = pd.read_csv("order_products__prior.csv (1).zip")
order_products_train = pd.read_csv("order_products__train.csv (1).zip")
products = pd.read_csv("products.csv (2).zip")

products_full = (
    products
    .merge(aisles, on="aisle_id", how="left")
    .merge(departments, on="department_id", how="left")
)
print("products_full:", products_full.shape)


prior_full = order_products_prior.merge(products_full, on="product_id", how="left")
print("prior_full:", prior_full.shape)


train_full = order_products_train.merge(products_full, on="product_id", how="left")
print("train_full:", train_full.shape)


prior_full = prior_full.merge(orders, on="order_id", how="left")
train_full = train_full.merge(orders, on="order_id", how="left")

print("prior_full (after orders):", prior_full.shape)
print("train_full (after orders):", train_full.shape)
