import pandas as pd
import numpy as np
import sqlite3
import os

# ============================================================
# STEP 1 — LOAD DATA FROM SQLITE AND SAVE TO data_raw/
# ============================================================

def load_data_from_sqlite(db_path, save_dir="data_raw"):
    os.makedirs(save_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)

    df_purchases = pd.read_sql("SELECT * FROM purchases", conn)
    df_coeffs = pd.read_sql("SELECT * FROM personal_data_coeffs", conn)
    df_personal = pd.read_sql("SELECT * FROM personal_data", conn)

    conn.close()

    df_purchases.to_csv(f"{save_dir}/purchases.csv", index=False)
    df_coeffs.to_csv(f"{save_dir}/personal_data_coeffs.csv", index=False)
    df_personal.to_csv(f"{save_dir}/personal_data.csv", index=False)

    print("✓ Step 1 completed: raw tables saved.")
    return df_purchases, df_coeffs, df_personal


# ============================================================
# STEP 2 — FILTER AND CLEAN DATA
# ============================================================

def filter_and_clean(df_purchases, df_coeffs, df_personal):
    # Filter by country = 32
    df_personal = df_personal[df_personal["country"] == 32].copy()
    ids = df_personal["id"].tolist()

    df_coeffs = df_coeffs[df_coeffs["id"].isin(ids)].copy()
    df_purchases = df_purchases[df_purchases["id"].isin(ids)].copy()

    # Fix missing colours
    df_purchases["colour"] = df_purchases["colour"].fillna("unknown")

    # Simplify colours
    df_purchases["colour_simple"] = df_purchases["colour"].apply(
        lambda x: x.split("/")[0] if "/" in str(x) else x
    )

    # Fix product_sex
    mode_sex = df_purchases["product_sex"].mode()[0]
    df_purchases["product_sex"] = df_purchases["product_sex"].fillna(mode_sex)

    # Fix negative costs
    df_purchases.loc[df_purchases["cost"] < 0, "cost"] = abs(
        df_purchases.loc[df_purchases["cost"] < 0, "cost"]
    )

    # Fix ac_coef = -inf
    df_coeffs["ac_coef"] = df_coeffs["ac_coef"].replace(float("-inf"), np.nan)
    ac_median = df_coeffs["ac_coef"].dropna().median()
    df_coeffs["ac_coef"] = df_coeffs["ac_coef"].fillna(ac_median)

    print("✓ Step 2 completed: data filtered and cleaned.")
    return df_purchases, df_coeffs, df_personal


# ============================================================
# STEP 3 — MERGE TABLES AND BUILD CUSTOMER FEATURES
# ============================================================

def build_customers(df_purchases, df_coeffs, df_personal, save_dir="data_processed"):
    os.makedirs(save_dir, exist_ok=True)

    # Merge personal + coeffs
    df_merged = pd.merge(df_personal, df_coeffs, on="id", how="inner")

    # Aggregate purchases
    customer_purchases = df_purchases.groupby("id").agg({
        "product": "count",
        "cost": "sum",
        "base_sale": "mean",
        "product_sex": "mean",
        "dt": ["min", "max"]
    })

    customer_purchases.columns = [
        "total_purchases",
        "total_spent",
        "discount_rate",
        "avg_product_sex",
        "first_purchase_day",
        "last_purchase_day"
    ]

    customer_purchases = customer_purchases.reset_index()

    # Merge with main table
    df_final = pd.merge(df_merged, customer_purchases, on="id", how="left")

    # Fill missing values for customers without purchases
    df_final["total_purchases"] = df_final["total_purchases"].fillna(0)
    df_final["total_spent"] = df_final["total_spent"].fillna(0)
    df_final["discount_rate"] = df_final["discount_rate"].fillna(0)
    df_final["avg_product_sex"] = df_final["avg_product_sex"].fillna(0.5)
    df_final["first_purchase_day"] = df_final["first_purchase_day"].fillna(-1)
    df_final["last_purchase_day"] = df_final["last_purchase_day"].fillna(-1)

    # Save final dataset
    df_final.to_csv(f"{save_dir}/customers_final.csv", index=False)

    print("✓ Step 3 completed: customers_final.csv saved.")
    return df_final


# ============================================================
# STEP 4 — LOAD ADDITIONAL personal_data.csv.gz
# ============================================================

def load_additional_data(gz_path, save_dir="data_processed"):
    os.makedirs(save_dir, exist_ok=True)

    df_additional = pd.read_csv(gz_path, compression="gzip")
    df_additional.to_csv(f"{save_dir}/df_additional.csv", index=False)

    print("✓ Step 4 completed: df_additional.csv saved.")
    return df_additional


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=== RUNNING DATA PREPARATION PIPELINE ===")

    # Step 1
    df_purchases, df_coeffs, df_personal = load_data_from_sqlite(
        r"C:\Users\lenovo\Desktop\skillbox\d_a\result_data\shop_database.db"
    )

    # Step 2
    df_purchases, df_coeffs, df_personal = filter_and_clean(
        df_purchases, df_coeffs, df_personal
    )

    # Step 3
    df_final = build_customers(df_purchases, df_coeffs, df_personal)

    # Step 4
    load_additional_data(
        r"C:\Users\lenovo\Desktop\skillbox\d_a\personal_data.csv.gz"
    )

    print("=== DATA PREPARATION COMPLETED ===")
