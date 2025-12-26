import pandas as pd
import re
import os

# ============================================================
# STEP 9 — A/B TEST PREPARATION
# ============================================================

def load_ids(path):
    """Extract all numbers from file and return as list of ints."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    ids = re.findall(r"\d+", text)
    return list(map(int, ids))


def prepare_ab_test(
    customers_path="data_processed/customers_final.csv",
    purchases_path="data_raw/purchases.csv",
    coeffs_path="data_raw/personal_data_coeffs.csv",
    test_ids_path=r"C:\Users\lenovo\Desktop\skillbox\d_a\result_data\ids_first_company_positive.txt",
    control_ids_path=r"C:\Users\lenovo\Desktop\skillbox\d_a\result_data\ids_first_company_negative.txt",
    save_dir="data_processed"
):
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    df_personal = pd.read_csv(customers_path)
    df_purchases = pd.read_csv(purchases_path)
    df_coeffs = pd.read_csv(coeffs_path)

    print("Loaded:")
    print(" - customers_final:", df_personal.shape)
    print(" - purchases:", df_purchases.shape)
    print(" - coeffs:", df_coeffs.shape)

    # Merge personal + coeffs
    df_all = df_personal.merge(df_coeffs, on="id", how="left")

    # Aggregate purchases
    purchase_stats = df_purchases.groupby("id").agg({
        "cost": ["sum", "count"],
        "base_sale": "mean",
        "product_sex": "mean"
    })

    purchase_stats.columns = [
        "total_spent",
        "total_purchases",
        "discount_rate",
        "avg_product_sex"
    ]

    purchase_stats = purchase_stats.reset_index()

    # Merge with personal data
    df_all_customers = df_all.merge(purchase_stats, on="id", how="left")

    # Fill missing values for customers without purchases
    df_all_customers["total_spent"] = df_all_customers["total_spent"].fillna(0)
    df_all_customers["total_purchases"] = df_all_customers["total_purchases"].fillna(0)
    df_all_customers["discount_rate"] = df_all_customers["discount_rate"].fillna(0)
    df_all_customers["avg_product_sex"] = df_all_customers["avg_product_sex"].fillna(0)

    # Filter only country = 32
    df_all_customers = df_all_customers[df_all_customers["country"] == 32].copy()

    print("Filtered customers:", df_all_customers.shape)

    # Load test/control IDs
    test_ids = load_ids(test_ids_path)
    control_ids = load_ids(control_ids_path)

    print(f"Test IDs loaded: {len(test_ids)}")
    print(f"Control IDs loaded: {len(control_ids)}")

    # Match IDs
    test_group = df_all_customers[df_all_customers["id"].isin(test_ids)].copy()
    test_group["group"] = "test"

    control_group = df_all_customers[df_all_customers["id"].isin(control_ids)].copy()
    control_group["group"] = "control"

    print(f"Matched test: {len(test_group)}")
    print(f"Matched control: {len(control_group)}")

    # Combine
    ab_data = pd.concat([test_group, control_group], ignore_index=True)

    # Save
    ab_data.to_csv(f"{save_dir}/ab_test_participants_prepared.csv", index=False)

    print("\nA/B test dataset ready!")
    print(ab_data["group"].value_counts())

    return ab_data


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=== RUNNING A/B TEST PREPARATION ===")
    prepare_ab_test()
    print("=== A/B TEST PREPARATION COMPLETED ===")
