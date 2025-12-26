import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ============================================================
# STEP 12 — PROPENSITY MODEL FOR CITY 1188
# ============================================================

def run_propensity_model(
    customers_path="data_processed/customers_clustered.csv",
    purchases_path="data_raw/purchases.csv",
    save_dir="data_processed"
):
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    customers = pd.read_csv(customers_path)
    purchases = pd.read_csv(purchases_path)

    print("Loaded:")
    print(" - customers_clustered:", customers.shape)
    print(" - purchases:", purchases.shape)

    # Filter city 1188
    city = customers[customers["city"] == 1188].copy()
    print(f"Customers in city 1188: {city.shape[0]}")

    # Find top product in this city
    top_product = (
        purchases[purchases["id"].isin(city["id"])]["product"]
        .value_counts()
        .idxmax()
    )
    print("Top product:", top_product)

    # Create target: bought / not bought
    buyers = purchases[purchases["product"] == top_product]["id"].unique()
    city["bought_target"] = city["id"].isin(buyers).astype(int)

    # Features
    features = [
        "age", "gender", "personal_coef",
        "total_purchases", "total_spent",
        "discount_rate", "avg_product_sex",
        "cluster"
    ]

    X = city[features].copy()
    y = city["bought_target"]

    # One-hot encode cluster
    X = pd.get_dummies(X, columns=["cluster"], drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    results = {}

    # Train and evaluate
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        results[name] = auc
        print(f"{name}: ROC-AUC = {auc:.3f}")

    # Best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    print(f"\nBest model: {best_model_name}")

    # Full propensity score
    city["propensity_score"] = best_model.predict_proba(
        scaler.transform(X)
    )[:, 1]

    # Deciles
    city["decile"] = pd.qcut(
        city["propensity_score"],
        q=10,
        labels=False,
        duplicates="drop"
    ) + 1

    # Decile analysis
    decile_table = city.groupby("decile")["bought_target"].mean()
    print("\nDecile response rates:")
    print(decile_table)

    # Save results
    city.to_csv(f"{save_dir}/city_1188_propensity_results.csv", index=False)
    print("\nSaved: city_1188_propensity_results.csv")

    return city


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=== RUNNING PROPENSITY MODEL ===")
    run_propensity_model()
    print("=== PROPENSITY MODEL COMPLETED ===")
