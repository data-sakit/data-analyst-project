import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

# ============================================================
# STEP 7 — FEATURE ENGINEERING (TRAIN/TEST PREPARATION)
# ============================================================

def build_train_test(
    customers_path="data_processed/customers_final.csv",
    additional_path="data_processed/df_additional.csv",
    save_dir="data_processed"
):
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    df_final = pd.read_csv(customers_path)
    df_additional = pd.read_csv(additional_path)

    print("Loaded:")
    print(" - customers_final:", df_final.shape)
    print(" - df_additional:", df_additional.shape)

    # Target
    y_train = df_final["gender"].copy()

    # Features
    common_features = ["age", "education", "city", "country"]

    x_train = df_final[common_features].copy()
    x_test = df_additional[common_features].copy()

    # OneHotEncoder for education
    encoder = OneHotEncoder(sparse_output=False, drop="first")

    train_encoded = encoder.fit_transform(x_train[["education"]])
    test_encoded = encoder.transform(x_test[["education"]])

    # Remove original column
    x_train = x_train.drop("education", axis=1)
    x_test = x_test.drop("education", axis=1)

    # Add encoded column
    encoded_col = encoder.get_feature_names_out()[0]

    x_train[encoded_col] = train_encoded[:, 0]
    x_test[encoded_col] = test_encoded[:, 0]

    # Save results
    x_train.to_csv(f"{save_dir}/X_train_prepared.csv", index=False)
    x_test.to_csv(f"{save_dir}/X_test_prepared.csv", index=False)
    y_train.to_csv(f"{save_dir}/y_train.csv", index=False)

    print("✓ Feature engineering completed.")
    print("Saved:")
    print(" - X_train_prepared.csv")
    print(" - X_test_prepared.csv")
    print(" - y_train.csv")

    return x_train, x_test, y_train


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=== RUNNING FEATURE ENGINEERING ===")
    build_train_test()
    print("=== FEATURE ENGINEERING COMPLETED ===")
