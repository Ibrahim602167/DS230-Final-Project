import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================
# LOAD DATA (your original)
# =========================
aisles = pd.read_csv("aisles (1).csv")
departments = pd.read_csv("departments (1).csv")

orders = pd.read_csv("orders.csv (1).zip")
order_products_prior = pd.read_csv("order_products__prior.csv (1).zip")
order_products_train = pd.read_csv("order_products__train.csv (1).zip")
products = pd.read_csv("products.csv (2).zip")

# ✅ ADDED: show raw shapes (sanity)
print("aisles:", aisles.shape)
print("departments:", departments.shape)
print("orders:", orders.shape)
print("order_products_prior:", order_products_prior.shape)
print("order_products_train:", order_products_train.shape)
print("products:", products.shape)

# ✅ ADDED: quick column check
print("\norders columns:", orders.columns.tolist())
print("prior columns:", order_products_prior.columns.tolist())
print("products columns:", products.columns.tolist())

# =========================
# JOINS (your original)
# =========================
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

# =========================
# ✅ MISSING + BASIC EDA COUNTS (your original + small adds)
# =========================
print("Number of users:", prior_full["user_id"].nunique())
print("Number of products:", prior_full["product_id"].nunique())
print("Number of orders:", prior_full["order_id"].nunique())

# ✅ ADDED: Target distribution + plot (required)
reorder_rate = prior_full["reordered"].value_counts(normalize=True)
print("\nReordered distribution (normalized):\n", reorder_rate)

plt.figure(figsize=(4,4))
plt.bar(reorder_rate.index.astype(str), reorder_rate.values, alpha=0.7)
plt.title("Target Distribution: reordered")
plt.xlabel("reordered (0/1)")
plt.ylabel("Proportion")
plt.tight_layout()
plt.show()

# Existing quick counts (your original)
prior_full["order_hour_of_day"].value_counts().sort_index()
prior_full["order_dow"].value_counts().sort_index()

user_orders = prior_full.groupby("user_id")["order_id"].nunique()
user_orders.describe()

product_orders = prior_full.groupby("product_id")["order_id"].count()
product_orders.describe()

# =========================
# ✅ ADDED: Missing evidence before fill + justification
# =========================
print("\nMissing days_since_prior_order ratio BEFORE fill:", prior_full["days_since_prior_order"].isna().mean())

# NaN for first order of each user -> fill 0
prior_full["days_since_prior_order"] = prior_full["days_since_prior_order"].fillna(0)
train_full["days_since_prior_order"] = train_full["days_since_prior_order"].fillna(0)

print("Missing days_since_prior_order ratio AFTER  fill:", prior_full["days_since_prior_order"].isna().mean())

# =========================
# ✅ ADDED: Memory optimization (strong) + evidence
# =========================
mem_before = prior_full.memory_usage(deep=True).sum() / 1024**2
print(f"\n[Memory] prior_full BEFORE drop product_name: {mem_before:.2f} MB")

# product_name is huge text -> not needed for EDA/features
if "product_name" in prior_full.columns:
    prior_full.drop(columns=["product_name"], inplace=True)
if "product_name" in train_full.columns:
    train_full.drop(columns=["product_name"], inplace=True)

mem_after = prior_full.memory_usage(deep=True).sum() / 1024**2
print(f"[Memory] prior_full AFTER  drop product_name: {mem_after:.2f} MB")
print(f"[Memory] Saved: {mem_before - mem_after:.2f} MB")

# =========================
# Downcast (your original)
# =========================
int_cols = ["user_id", "order_id", "product_id",
            "order_number", "add_to_cart_order",
            "reordered", "order_dow", "order_hour_of_day"]
for col in int_cols:
    if col in prior_full.columns:
        prior_full[col] = pd.to_numeric(prior_full[col], downcast="integer")
    if col in train_full.columns:
        train_full[col] = pd.to_numeric(train_full[col], downcast="integer")

# ✅ ADDED: convert repeated strings to category (memory)
for c in ["eval_set", "aisle", "department"]:
    if c in prior_full.columns:
        prior_full[c] = prior_full[c].astype("category")
    if c in train_full.columns:
        train_full[c] = train_full[c].astype("category")

# ✅ ADDED: memory print (evidence)
prior_mem = prior_full.memory_usage(deep=True).sum() / 1024**2
train_mem = train_full.memory_usage(deep=True).sum() / 1024**2
print(f"\nprior_full memory (MB): {prior_mem:.2f}")
print(f"train_full memory (MB): {train_mem:.2f}")

# =========================
# Cleaning filters (your original)
# =========================
prior_full = prior_full[(prior_full["order_hour_of_day"] >= 0) & (prior_full["order_hour_of_day"] <= 23)]
prior_full = prior_full[prior_full["days_since_prior_order"] >= 0]

# ✅ ADDED: cleaning checks for required fields
print("\n[Cleaning] order_number min/max:",
      prior_full["order_number"].min(), prior_full["order_number"].max())
print("[Cleaning] add_to_cart_order min/max:",
      prior_full["add_to_cart_order"].min(), prior_full["add_to_cart_order"].max())

bad_order_number = (prior_full["order_number"] <= 0).sum()
bad_cart_order = (prior_full["add_to_cart_order"] <= 0).sum()
print("[Cleaning] invalid order_number<=0 count:", bad_order_number)
print("[Cleaning] invalid add_to_cart_order<=0 count:", bad_cart_order)

prior_full = prior_full[prior_full["order_number"] > 0]
prior_full = prior_full[prior_full["add_to_cart_order"] > 0]

# =========================
# ✅ FIX: missing percent should be *100 not *70
# =========================
missing_percent = prior_full.isna().mean() * 100
missing_percent = missing_percent[missing_percent > 0]
if not missing_percent.empty:
    plt.figure(figsize=(10,4))
    plt.bar(missing_percent.index, missing_percent.values, alpha=0.7)
    plt.ylabel("% Missing")
    plt.title("Missing Values by Column (prior_full)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("\nNo missing values after current preprocessing.")

# =========================
# ✅ ADDED: Cardinality analysis (top-k) - required
# =========================
top_products = prior_full["product_id"].value_counts().head(10)
plt.figure(figsize=(7,4))
plt.bar(top_products.index.astype(str), top_products.values, alpha=0.7)
plt.title("Top-10 Most Ordered product_id (prior)")
plt.xlabel("product_id")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

if "aisle" in prior_full.columns:
    top_aisles = prior_full["aisle"].value_counts().head(10)
    plt.figure(figsize=(7,4))
    plt.bar(top_aisles.index.astype(str), top_aisles.values, alpha=0.7)
    plt.title("Top-10 Aisles (prior)")
    plt.xlabel("aisle")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

if "department" in prior_full.columns:
    top_depts = prior_full["department"].value_counts().head(10)
    plt.figure(figsize=(7,4))
    plt.bar(top_depts.index.astype(str), top_depts.values, alpha=0.7)
    plt.title("Top-10 Departments (prior)")
    plt.xlabel("department")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

print("\n[Cardinality] nunique summary:")
for c in ["user_id", "product_id", "aisle_id", "department_id", "aisle", "department", "order_id"]:
    if c in prior_full.columns:
        print(f" - {c}: {prior_full[c].nunique()}")

# =========================
# Numeric distributions (your original)
# =========================
numeric_cols = ["add_to_cart_order","days_since_prior_order", "order_number"]
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    plt.hist(prior_full[col], bins=15, density=True, alpha=0.5)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()

# =========================
# ✅ ADDED: Boxplots (outlier detection) + Outlier treatment (required)
# -------------------------
# First: BEFORE treatment (optional view)
for col in numeric_cols:
    plt.figure(figsize=(5,3))
    plt.boxplot(prior_full[col].dropna(), vert=False)
    plt.title(f"Boxplot BEFORE clipping: {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# Treatment: clipping/winsorizing-like at 1% and 99%
for col in numeric_cols:
    lo = prior_full[col].quantile(0.01)
    hi = prior_full[col].quantile(0.99)
    print(f"\n[Outliers] {col}: clip range [{lo:.3f}, {hi:.3f}]")
    out_low = (prior_full[col] < lo).sum()
    out_high = (prior_full[col] > hi).sum()
    print(f"[Outliers] {col}: below_lo={out_low}, above_hi={out_high}")
    prior_full[col] = prior_full[col].clip(lower=lo, upper=hi)

# AFTER treatment evidence
for col in numeric_cols:
    plt.figure(figsize=(5,3))
    plt.boxplot(prior_full[col].dropna(), vert=False)
    plt.title(f"Boxplot AFTER clipping: {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# =========================
# Categorical counts (your original)
# =========================
categorical_cols = ["order_dow", "order_hour_of_day"]
for col in categorical_cols:
    counts = prior_full[col].value_counts().sort_index()
    plt.figure(figsize=(4,4))
    plt.bar(counts.index, counts.values, alpha=0.6)
    plt.title(f"Counts of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# =========================
# ✅ ADDED: Pairwise scatter (required) - small sample to be fast
# =========================
sample_df = prior_full[["order_number", "add_to_cart_order", "days_since_prior_order"]].sample(
    n=min(5000, len(prior_full)), random_state=42
)
plt.figure(figsize=(5,4))
plt.scatter(sample_df["order_number"], sample_df["days_since_prior_order"], alpha=0.2, s=10)
plt.title("Scatter: order_number vs days_since_prior_order (sample)")
plt.xlabel("order_number")
plt.ylabel("days_since_prior_order")
plt.tight_layout()
plt.show()

# =========================
# Correlation (your original)
# =========================
corr = prior_full[numeric_cols].corr()
plt.figure(figsize=(5,4))
plt.imshow(corr, cmap='Blues', interpolation='none', aspect='auto')
plt.xticks(range(len(numeric_cols)), numeric_cols, fontsize=8, rotation=45)
plt.yticks(range(len(numeric_cols)), numeric_cols, fontsize=8)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# =========================
# Seasonality (your original)
# =========================
hour_counts = prior_full["order_hour_of_day"].value_counts().sort_index()
plt.figure(figsize=(6,4))
plt.bar(hour_counts.index, hour_counts.values, alpha=0.7)
plt.title("Orders by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

dow_counts = prior_full["order_dow"].value_counts().sort_index()
plt.figure(figsize=(6,4))
plt.bar(dow_counts.index, dow_counts.values, alpha=0.7)
plt.title("Orders by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# =========================
# ✅ ADDED: Monthly seasonality note (dataset limitation)
# =========================
print("\n[Seasonality] Monthly seasonality plot is NOT available because there is no timestamp/month field in the dataset "
      "(orders has order_dow and order_hour_of_day only).")
# 1) total #orders per user
user_total_orders_df = (
    prior_full.groupby("user_id")["order_id"]
    .nunique()
    .reset_index(name="user_total_orders")
)

# 2) average basket size per user
order_basket_df = (
    prior_full.groupby(["user_id", "order_id"])["product_id"]
    .count()
    .reset_index(name="basket_size")
)

user_avg_basket_df = (
    order_basket_df.groupby("user_id")["basket_size"]
    .mean()
    .reset_index(name="user_avg_basket_size")
)

# 3) reorder ratio per user
user_reorder_ratio_df = (
    prior_full.groupby("user_id")["reordered"]
    .mean()
    .reset_index(name="user_reorder_ratio")
)

# 4) mean days between orders per user
user_mean_days_between_df = (
    prior_full.groupby("user_id")["days_since_prior_order"]
    .mean()
    .reset_index(name="user_mean_days_between_orders")
)

# 5) last order recency (proxy): days_since_prior_order for user's last prior order
idx_last_user = prior_full.groupby("user_id")["order_number"].idxmax()
user_last_recency_df = (
    prior_full.loc[idx_last_user, ["user_id", "days_since_prior_order"]]
    .rename(columns={"days_since_prior_order": "user_last_order_recency"})
    .reset_index(drop=True)
)

# Merge user features
user_features_df = (
    user_total_orders_df
    .merge(user_avg_basket_df, on="user_id", how="left")
    .merge(user_reorder_ratio_df, on="user_id", how="left")
    .merge(user_mean_days_between_df, on="user_id", how="left")
    .merge(user_last_recency_df, on="user_id", how="left")
)

print("\nuser_features_df:", user_features_df.shape)
print(user_features_df.head())

# ----------------------------
# B) PRODUCT-LEVEL FEATURES
# ----------------------------

# 1) overall #orders (popularity)
prod_orders_df = (
    prior_full.groupby("product_id")["order_id"]
    .count()
    .reset_index(name="prod_orders")
)

# 2) overall reorder rate
prod_reorder_rate_df = (
    prior_full.groupby("product_id")["reordered"]
    .mean()
    .reset_index(name="prod_reorder_rate")
)

# 3) average position in cart
prod_avg_cart_pos_df = (
    prior_full.groupby("product_id")["add_to_cart_order"]
    .mean()
    .reset_index(name="prod_avg_add_to_cart_order")
)

# 4) popularity over time (simple proxy): avg order_number when product is bought
prod_popularity_over_time_df = (
    prior_full.groupby("product_id")["order_number"]
    .mean()
    .reset_index(name="prod_avg_order_number_when_bought")
)

product_features_df = (
    prod_orders_df
    .merge(prod_reorder_rate_df, on="product_id", how="left")
    .merge(prod_avg_cart_pos_df, on="product_id", how="left")
    .merge(prod_popularity_over_time_df, on="product_id", how="left")
)

print("\nproduct_features_df:", product_features_df.shape)
print(product_features_df.head())

# ----------------------------
# C) USER × PRODUCT FEATURES
# ----------------------------

# 1) prior purchase count
user_prod_prior_count_df = (
    prior_full.groupby(["user_id", "product_id"])["order_id"]
    .count()
    .reset_index(name="user_prod_prior_purchase_count")
)

# 2) average reorder probability for user-product
user_prod_avg_reorder_prob_df = (
    prior_full.groupby(["user_id", "product_id"])["reordered"]
    .mean()
    .reset_index(name="user_prod_avg_reorder_prob")
)

# 3) days since last purchase (proxy using order_number gap)
user_prod_last_order_number_df = (
    prior_full.groupby(["user_id", "product_id"])["order_number"]
    .max()
    .reset_index(name="user_prod_last_order_number")
)

user_last_order_number_df = (
    prior_full.groupby("user_id")["order_number"]
    .max()
    .reset_index(name="user_last_order_number")
)

user_prod_recency_df = user_prod_last_order_number_df.merge(
    user_last_order_number_df, on="user_id", how="left"
)

user_prod_recency_df["user_prod_days_since_last_purchase_orders"] = (
    user_prod_recency_df["user_last_order_number"]
    - user_prod_recency_df["user_prod_last_order_number"]
)

user_product_features_df = (
    user_prod_prior_count_df
    .merge(user_prod_avg_reorder_prob_df, on=["user_id", "product_id"], how="left")
    .merge(
        user_prod_recency_df[["user_id", "product_id", "user_prod_days_since_last_purchase_orders"]],
        on=["user_id", "product_id"], how="left"
    )
)

print("\nuser_product_features_df:", user_product_features_df.shape)
print(user_product_features_df.head())

# ----------------------------
# D) TEMPORAL FEATURES (context for next order)
# ----------------------------
# Use each user's TRAIN order metadata as "next order context" (dow/hour)
# Also gives a regression target option: days_since_prior_order for the train order
train_order_meta = (
    train_full[["user_id", "order_id", "order_dow", "order_hour_of_day", "days_since_prior_order"]]
    .drop_duplicates("user_id")
    .rename(columns={"days_since_prior_order": "target_days_to_next_order"})
)

print("\ntrain_order_meta:", train_order_meta.shape)
print(train_order_meta.head())

# ============================================================
# OPTIONAL (but useful now): Build datasets for Task A & B
# ============================================================

# ----------------------------
# Task A (Classification dataset)
# Label comes from train_full: (user_id, product_id) reordered in train
# Candidate pairs: pairs that exist in prior (keeps dataset manageable)
# ----------------------------
y_cls = train_full[["user_id", "product_id", "reordered"]].rename(columns={"reordered": "y_reordered_next"})

X_cls = (
    user_product_features_df
    .merge(user_features_df, on="user_id", how="left")
    .merge(product_features_df, on="product_id", how="left")
    .merge(products_full.drop(columns=["product_name"], errors="ignore")[["product_id", "aisle_id", "department_id", "aisle", "department"]],
           on="product_id", how="left")
    .merge(train_order_meta[["user_id", "order_dow", "order_hour_of_day"]], on="user_id", how="left")
)

Xy_cls = X_cls.merge(y_cls, on=["user_id", "product_id"], how="left")
Xy_cls["y_reordered_next"] = Xy_cls["y_reordered_next"].fillna(0).astype("int8")

print("\nXy_cls (Task A):", Xy_cls.shape)
print("Task A class balance:", Xy_cls["y_reordered_next"].value_counts(normalize=True))

# ----------------------------
# Task B (Regression dataset) — user-level target
# Predict days until next order (proxy): train order days_since_prior_order
# ----------------------------
X_reg = (
    user_features_df
    .merge(train_order_meta[["user_id", "order_dow", "order_hour_of_day", "target_days_to_next_order"]],
           on="user_id", how="inner")
)

print("\nX_reg (Task B):", X_reg.shape)
print(X_reg.head())

# ----------------------------
# Save (optional)
# ----------------------------
# If parquet gives you issues, change to .to_csv(...)
user_features_df.to_csv("fe_user_features.csv", index=False)
product_features_df.to_csv("fe_product_features.csv", index=False)
user_product_features_df.to_csv("fe_user_product_features.csv", index=False)

Xy_cls.to_csv("dataset_taskA_classification.csv", index=False)
X_reg.to_csv("dataset_taskB_regression.csv", index=False)

print("\nSaved CSVs:")
print("- fe_user_features.csv")
print("- fe_product_features.csv")
print("- fe_user_product_features.csv")
print("- dataset_taskA_classification.csv")
print("- dataset_taskB_regression.csv")
# =========================
# STEP 3 (Task B - Regression)  — ADD THIS AT THE END OF YOUR ORIGINAL FILE
# =========================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===== تجهيز X و y =====
X = X_reg.drop(columns=["target_days_to_next_order"])
y = X_reg["target_days_to_next_order"]

# ✅ Make time features categorical (better than treating them as continuous numbers)
if "order_dow" in X.columns:
    X["order_dow"] = X["order_dow"].astype("category")
if "order_hour_of_day" in X.columns:
    X["order_hour_of_day"] = X["order_hour_of_day"].astype("category")

# ✅ FIX: numeric columns must include ALL numeric dtypes (int8/int16/int32/float32 too)
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("\n[Task B] Numeric columns:", num_cols)
print("[Task B] Categorical columns:", cat_cols)

# ===== Preprocessing =====
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop"
)

# ===== Split =====
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Pipeline (clean + prevents mistakes)
pipe_reg = Pipeline([
    ("prep", preprocessor),
    ("model", LinearRegression())
])

# ===== Train =====
pipe_reg.fit(X_train, y_train)

# ===== Predict =====
y_pred = pipe_reg.predict(X_val)

# ===== Evaluate =====
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print("\nTask B – Regression Results (Linear Regression)")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
import pandas as pd
import numpy as np

# ===== Downsample to avoid MemoryError (keep all positives + 5x negatives) =====
pos = Xy_cls[Xy_cls["y_reordered_next"] == 1]
neg = Xy_cls[Xy_cls["y_reordered_next"] == 0]

neg_sample = neg.sample(n=min(len(pos) * 5, len(neg)), random_state=42)
Xy_small = pd.concat([pos, neg_sample], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

print("\n[Task A] Using downsampled data:", Xy_small.shape)
print("[Task A] Downsampled class balance:\n", Xy_small["y_reordered_next"].value_counts(normalize=True))

# ===== تجهيز X و y =====
X = Xy_small.drop(columns=["y_reordered_next"])
y = Xy_small["y_reordered_next"]

# ✅ FIX: include all numeric dtypes (int8/int16/int32/float32...)
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("\n[Task A] Numeric columns:", len(num_cols))
print("[Task A] Categorical columns:", len(cat_cols))

# ===== Preprocessing (light + sparse-friendly) =====
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),  # lighter than median
    ("scaler", StandardScaler(with_mean=False))  # IMPORTANT for sparse
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ],
    remainder="drop",
    sparse_threshold=0.3
)

# ===== Split =====
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== Model =====
pipe_cls = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

pipe_cls.fit(X_train, y_train)

# ===== Predict + Evaluate =====
y_pred = pipe_cls.predict(X_val)
y_pred_proba = pipe_cls.predict_proba(X_val)[:, 1]

print("\nTask A – Classification Results (Logistic Regression, downsampled)")
print("Accuracy :", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred))
print("Recall   :", recall_score(y_val, y_pred))
print("F1-score :", f1_score(y_val, y_pred))
print("ROC-AUC  :", roc_auc_score(y_val, y_pred_proba))
print("PR-AUC   :", average_precision_score(y_val, y_pred_proba))
