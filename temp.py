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
# =========================
# STEP 4 — FULL MODEL SUITE + COMPARISON (Task A + Task B) ✅ READY TO PASTE
# Paste this whole block at "Step 4" and run ONLY this block (not the whole file)
# Requires: Xy_cls, X_reg already created above
# =========================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# ----- Task A models -----
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# ----- Task B models -----
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# Helper: get score for AUC metrics
# =========================
def _get_score(model, X):
    """Return a continuous score for ROC/PR AUC.
    Prefer predict_proba[:,1], else decision_function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


# =========================
# TASK A — Classification: full suite (stable)
# =========================
def run_taskA_suite(Xy_cls, neg_ratio=5, heavy_sample=200000, random_state=42):

    # ---- Downsample for feasibility (all pos + neg_ratio*pos negatives) ----
    pos = Xy_cls[Xy_cls["y_reordered_next"] == 1]
    neg = Xy_cls[Xy_cls["y_reordered_next"] == 0]
    neg_sample = neg.sample(n=min(len(pos) * neg_ratio, len(neg)), random_state=random_state)

    Xy_small = (
        pd.concat([pos, neg_sample], axis=0)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    print("\n[Task A] Using downsampled data:", Xy_small.shape)
    print("[Task A] Downsampled class balance:\n",
          Xy_small["y_reordered_next"].value_counts(normalize=True))

    X = Xy_small.drop(columns=["y_reordered_next"])
    y = Xy_small["y_reordered_next"]

    # Identify columns
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Sparse-friendly preprocessor (fast linear models)
    num_sparse = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value=0)),
        ("sc", StandardScaler(with_mean=False))
    ])
    cat_sparse = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])
    preproc_sparse = ColumnTransformer(
        transformers=[("num", num_sparse, num_cols), ("cat", cat_sparse, cat_cols)],
        remainder="drop"
    )

    # Dense preprocessor (for heavy models: kNN/Tree/RF/HistGB)
    num_dense = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    cat_dense = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    preproc_dense = ColumnTransformer(
        transformers=[("num", num_dense, num_cols), ("cat", cat_dense, cat_cols)],
        remainder="drop"
    )

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    results = []

    # ---------- FAST models (train on FULL train split) ----------
    fast_models = [
        ("LogReg", preproc_sparse, LogisticRegression(max_iter=2000, class_weight="balanced")),
        ("SGD_LogLoss", preproc_sparse, SGDClassifier(loss="log_loss", class_weight="balanced", random_state=random_state)),
        ("LinearSVC", preproc_sparse, LinearSVC(class_weight="balanced", random_state=random_state)),
        ("BernoulliNB", preproc_sparse, BernoulliNB())
    ]

    for name, prep, clf in fast_models:
        pipe = Pipeline([("prep", prep), ("model", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_val)
        score = _get_score(pipe, X_val)

        results.append([
            name,
            accuracy_score(y_val, y_pred),
            precision_score(y_val, y_pred, zero_division=0),
            recall_score(y_val, y_pred, zero_division=0),
            f1_score(y_val, y_pred, zero_division=0),
            roc_auc_score(y_val, score),
            average_precision_score(y_val, score),
            f"full_train({len(X_train)})"
        ])
        print(f"[Task A] Done: {name}")

    # ---------- HEAVY models (train on SUBSAMPLE only) ----------
    n_heavy = min(heavy_sample, len(X_train))
    rng = np.random.RandomState(random_state)
    heavy_idx = rng.choice(len(X_train), size=n_heavy, replace=False)

    X_train_h = X_train.iloc[heavy_idx]
    y_train_h = y_train.iloc[heavy_idx]

    heavy_models = [
        ("kNN", preproc_dense, KNeighborsClassifier(n_neighbors=15)),
        ("DecisionTree", preproc_dense,
         DecisionTreeClassifier(max_depth=12, random_state=random_state, class_weight="balanced")),
        ("RandomForest", preproc_dense,
         RandomForestClassifier(n_estimators=120, random_state=random_state, n_jobs=-1,
                                class_weight="balanced_subsample")),
        ("HistGB", preproc_dense, HistGradientBoostingClassifier(random_state=random_state))
    ]

    for name, prep, clf in heavy_models:
        pipe = Pipeline([("prep", prep), ("model", clf)])
        pipe.fit(X_train_h, y_train_h)

        y_pred = pipe.predict(X_val)
        score = _get_score(pipe, X_val)

        results.append([
            name,
            accuracy_score(y_val, y_pred),
            precision_score(y_val, y_pred, zero_division=0),
            recall_score(y_val, y_pred, zero_division=0),
            f1_score(y_val, y_pred, zero_division=0),
            roc_auc_score(y_val, score),
            average_precision_score(y_val, score),
            f"sub_train({n_heavy})"
        ])
        print(f"[Task A] Done: {name}")

    df = pd.DataFrame(
        results,
        columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "TrainSize"]
    ).sort_values(by="PR_AUC", ascending=False).reset_index(drop=True)

    print("\n=== Task A Model Comparison (sorted by PR_AUC) ===")
    print(df)
    return df


# =========================
# TASK B — Regression: full suite (fix HistGB sparse issue)
# =========================
def run_taskB_suite(X_reg, random_state=42, svr_sample=30000):

    X = X_reg.drop(columns=["target_days_to_next_order"]).copy()
    y = X_reg["target_days_to_next_order"].copy()

    # Treat time-like as categorical
    if "order_dow" in X.columns:
        X["order_dow"] = X["order_dow"].astype("category")
    if "order_hour_of_day" in X.columns:
        X["order_hour_of_day"] = X["order_hour_of_day"].astype("category")

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # ---- Sparse preprocessor (default for most models) ----
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    cat_pipe_sparse = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])
    preproc_sparse = ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe_sparse, cat_cols)],
        remainder="drop"
    )

    # ---- Dense preprocessor (for HistGBReg ONLY; it needs dense X) ----
    cat_pipe_dense = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    preproc_dense = ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe_dense, cat_cols)],
        remainder="drop"
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    models = [
        ("LinearRegression", "sparse", LinearRegression()),
        ("Ridge", "sparse", Ridge(alpha=1.0, random_state=random_state)),
        ("Lasso", "sparse", Lasso(alpha=0.001, random_state=random_state)),
        ("kNN_Reg", "sparse", KNeighborsRegressor(n_neighbors=25)),
        ("DecisionTreeReg", "sparse", DecisionTreeRegressor(max_depth=12, random_state=random_state)),
        ("RandomForestReg", "sparse", RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)),
        ("HistGBReg", "dense", HistGradientBoostingRegressor(random_state=random_state)),
        ("SVR_RBF", "sparse", SVR(C=5.0, epsilon=1.0)),
    ]

    results = []
    for name, mode, reg in models:

        prep = preproc_dense if mode == "dense" else preproc_sparse
        pipe = Pipeline([("prep", prep), ("model", reg)])

        # SVR can be slow -> train on subset only
        if name == "SVR_RBF":
            n_svr = min(svr_sample, len(X_train))
            rng = np.random.RandomState(random_state)
            idx = rng.choice(len(X_train), size=n_svr, replace=False)
            pipe.fit(X_train.iloc[idx], y_train.iloc[idx])
            y_pred = pipe.predict(X_val)
            train_note = f"sub_train({n_svr})"
        else:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            train_note = f"full_train({len(X_train)})"

        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        results.append([name, mae, rmse, r2, train_note])
        print(f"[Task B] Done: {name}")

    df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2", "TrainSize"])
    df = df.sort_values(by="MAE", ascending=True).reset_index(drop=True)

    print("\n=== Task B Model Comparison (sorted by MAE) ===")
    print(df)
    return df


# =========================
# RUN BOTH SUITES
# =========================
taskA_results = run_taskA_suite(Xy_cls, neg_ratio=5, heavy_sample=200000, random_state=42)
taskB_results = run_taskB_suite(X_reg, random_state=42, svr_sample=30000)
# =========================
# STEP 5 — Hyperparameter Tuning + (Time-aware split for Task B when possible)
# Paste هذا الجزء بعد Step 4 مباشرة (بعد ما يكون عندك Xy_cls و X_reg جاهزين)
# =========================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# -------------------------
# Helper: downsample Task A (مثل اللي عندك) عشان التونينغ ما يطول
# -------------------------
def _downsample_taskA(Xy_cls, neg_ratio=5, random_state=42):
    pos = Xy_cls[Xy_cls["y_reordered_next"] == 1]
    neg = Xy_cls[Xy_cls["y_reordered_next"] == 0]
    neg_sample = neg.sample(n=min(len(pos) * neg_ratio, len(neg)), random_state=random_state)

    Xy_small = (
        pd.concat([pos, neg_sample], axis=0)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    return Xy_small


# =========================
# TASK A — Tune HistGBClassifier (أفضل موديل عندك غالباً)
# Dense-only preprocessing (Ordinal for categories) عشان HistGB ما يحب sparse
# =========================
def tune_taskA_histgb(
    Xy_cls,
    neg_ratio=5,
    train_cap=250_000,         # سقف لحجم التدريب لتخفيف الوقت
    random_state=42,
    n_iter=20,                 # عدد تجارب التونينغ (خفيف)
):
    Xy_small = _downsample_taskA(Xy_cls, neg_ratio=neg_ratio, random_state=random_state)

    print("\n[Task A - Tuning] downsampled:", Xy_small.shape)
    print("[Task A - Tuning] class balance:\n", Xy_small["y_reordered_next"].value_counts(normalize=True))

    X = Xy_small.drop(columns=["y_reordered_next"])
    y = Xy_small["y_reordered_next"]

    # أعمدة
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Dense preprocessor (HistGB يحتاج dense)
    num_dense = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    cat_dense = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preproc_dense = ColumnTransformer(
        transformers=[("num", num_dense, num_cols), ("cat", cat_dense, cat_cols)],
        remainder="drop"
    )

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # قص التدريب (لتسريع التونينغ)
    if len(X_train) > train_cap:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X_train), size=train_cap, replace=False)
        X_train = X_train.iloc[idx]
        y_train = y_train.iloc[idx]
        print(f"[Task A - Tuning] Training capped to: {train_cap}")

    model = HistGradientBoostingClassifier(random_state=random_state)

    pipe = Pipeline([
        ("prep", preproc_dense),
        ("model", model)
    ])

    # مساحة بحث خفيفة
    param_dist = {
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__max_depth": [3, 4, 6, 8, None],
        "model__max_iter": [200, 300, 500],
        "model__min_samples_leaf": [20, 50, 100, 200],
        "model__l2_regularization": [0.0, 0.1, 0.5, 1.0],
    }

    # PR-AUC هو الأفضل مع عدم توازن الكلاسات
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="average_precision",
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=random_state
    )

    search.fit(X_train, y_train)

    best_pipe = search.best_estimator_
    print("\n[Task A - Tuning] Best params:\n", search.best_params_)
    print("[Task A - Tuning] Best CV PR-AUC:", search.best_score_)

    # تقييم على validation
    y_pred = best_pipe.predict(X_val)
    # continuous score for AUCs
    if hasattr(best_pipe, "predict_proba"):
        score = best_pipe.predict_proba(X_val)[:, 1]
    else:
        score = best_pipe.decision_function(X_val)

    pr_auc = average_precision_score(y_val, score)
    roc_auc = roc_auc_score(y_val, score)

    print("\n[Task A - Tuning] Validation PR-AUC:", pr_auc)
    print("[Task A - Tuning] Validation ROC-AUC:", roc_auc)

    return best_pipe, search.best_params_, {"val_pr_auc": pr_auc, "val_roc_auc": roc_auc}


# =========================
# TASK B — Tune HistGBRegressor + Time-aware split (إذا نقدر)
# - إذا عندك عمود order_number أو order_id: بنعمل split "زمني" تقريبي
# - غير هيك: بنرجع split عادي ونطبع ملاحظة
# =========================
def tune_taskB_histgb(
    X_reg,
    random_state=42,
    n_iter=25,
):
    X = X_reg.drop(columns=["target_days_to_next_order"]).copy()
    y = X_reg["target_days_to_next_order"].copy()

    # خلي الوقت كـ categorical
    for c in ["order_dow", "order_hour_of_day"]:
        if c in X.columns:
            X[c] = X[c].astype("category")

    # محاولة ترتيب زمني
    time_col = None
    if "order_number" in X.columns:
        time_col = "order_number"
    elif "order_id" in X.columns:
        time_col = "order_id"

    if time_col is not None:
        order_idx = np.argsort(X[time_col].values)
        X = X.iloc[order_idx].reset_index(drop=True)
        y = y.iloc[order_idx].reset_index(drop=True)
        print(f"\n[Task B - Tuning] Using time-aware ordering by: {time_col}")
        tscv = TimeSeriesSplit(n_splits=3)
        cv_used = tscv
    else:
        print("\n[Task B - Tuning] NOTE: No order_number/order_id found -> using normal CV (not time-aware).")
        cv_used = 3

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Dense preprocessor (HistGBReg يحتاج dense)
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    cat_pipe_dense = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preproc_dense = ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe_dense, cat_cols)],
        remainder="drop"
    )

    model = HistGradientBoostingRegressor(random_state=random_state)
    pipe = Pipeline([
        ("prep", preproc_dense),
        ("model", model)
    ])

    # مساحة بحث
    param_dist = {
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__max_depth": [3, 4, 6, 8, None],
        "model__max_iter": [200, 300, 500],
        "model__min_samples_leaf": [20, 50, 100, 200],
        "model__l2_regularization": [0.0, 0.1, 0.5, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",  # نريد MAE أقل
        cv=cv_used,
        verbose=1,
        n_jobs=-1,
        random_state=random_state
    )

    search.fit(X, y)

    best_pipe = search.best_estimator_
    print("\n[Task B - Tuning] Best params:\n", search.best_params_)
    print("[Task B - Tuning] Best CV MAE:", -search.best_score_)

    # تقييم بسيط على آخر 20% (time-aware holdout إذا عندنا time_col)
    split_point = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]

    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print("\n[Task B - Tuning] Holdout (last 20%) MAE :", mae)
    print("[Task B - Tuning] Holdout (last 20%) RMSE:", rmse)
    print("[Task B - Tuning] Holdout (last 20%) R2  :", r2)

    return best_pipe, search.best_params_, {"holdout_mae": mae, "holdout_rmse": rmse, "holdout_r2": r2}


# =========================
# RUN STEP 5 (TUNING)
# =========================
bestA_pipe, bestA_params, bestA_metrics = tune_taskA_histgb(
    Xy_cls,
    neg_ratio=5,
    train_cap=250_000,
    random_state=42,
    n_iter=20
)

bestB_pipe, bestB_params, bestB_metrics = tune_taskB_histgb(
    X_reg,
    random_state=42,
    n_iter=25
)

print("\n===== STEP 5 SUMMARY =====")
print("[Task A] best params:", bestA_params)
print("[Task A] val metrics:", bestA_metrics)
print("[Task B] best params:", bestB_params)
print("[Task B] holdout metrics:", bestB_metrics)
