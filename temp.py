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


print("Number of users:", prior_full["user_id"].nunique())
print("Number of products:", prior_full["product_id"].nunique())
print("Number of orders:", prior_full["order_id"].nunique())

prior_full["reordered"].value_counts(normalize=True)

prior_full["order_hour_of_day"].value_counts().sort_index()
prior_full["order_dow"].value_counts().sort_index()

user_orders = prior_full.groupby("user_id")["order_id"].nunique()
user_orders.describe()

product_orders = prior_full.groupby("product_id")["order_id"].count()
product_orders.describe()

prior_full["days_since_prior_order"].isna().mean()
prior_full["days_since_prior_order"] = prior_full["days_since_prior_order"].fillna(0)

int_cols = [
    "user_id", "order_id", "product_id",
    "order_number", "add_to_cart_order",
    "reordered", "order_dow", "order_hour_of_day"
]

for col in int_cols:
    if col in prior_full.columns:
        prior_full[col] = pd.to_numeric(prior_full[col], downcast="integer")
    if col in train_full.columns:
        train_full[col] = pd.to_numeric(train_full[col], downcast="integer")

prior_full.memory_usage(deep=True).sum() / 1024**2
train_full.memory_usage(deep=True).sum() / 1024**2
