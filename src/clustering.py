import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# STEP 10 — CUSTOMER CLUSTERING
# STEP 11 — CLUSTER VISUALIZATION
# ============================================================

def run_clustering(
    customers_path="data_processed/customers_final.csv",
    save_dir="data_processed",
    report_dir="reports"
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # Load data
    customers = pd.read_csv(customers_path)
    print(f"Loaded {customers.shape[0]} customers")

    # Features for clustering
    features = [
        "age",
        "city",
        "personal_coef",
        "total_purchases",
        "total_spent",
        "discount_rate",
        "avg_product_sex"
    ]

    X = customers[features].copy()

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal k
    wcss = []
    sil_scores = []

    print("\nSearching for optimal number of clusters...")
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        sil = silhouette_score(X_scaled, km.labels_)
        sil_scores.append(sil)
        print(f"k={k}: silhouette={sil:.3f}")

    optimal_k = sil_scores.index(max(sil_scores)) + 2
    print(f"\nOptimal number of clusters: {optimal_k}")

    # Train final KMeans
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    customers["cluster"] = kmeans.fit_predict(X_scaled)

    # Save clustered dataset
    customers.to_csv(f"{save_dir}/customers_clustered.csv", index=False)
    print("✓ Saved: customers_clustered.csv")

    # ============================================================
    # VISUALIZATIONS
    # ============================================================

    sns.set(style="whitegrid", palette="Set2")

    # 1. Cluster distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x="cluster", data=customers)
    plt.title("Распределение клиентов по кластерам")
    plt.xlabel("Кластер")
    plt.ylabel("Количество клиентов")
    plt.savefig(f"{report_dir}/cluster_distribution.png")
    plt.close()

    # 2. Average spending by cluster
    plt.figure(figsize=(8, 5))
    sns.barplot(x="cluster", y="total_spent", data=customers, estimator="mean")
    plt.title("Средние траты по кластерам")
    plt.xlabel("Кластер")
    plt.ylabel("Средняя сумма трат")
    plt.savefig(f"{report_dir}/cluster_spending.png")
    plt.close()

    # 3. Average purchases by cluster
    plt.figure(figsize=(8, 5))
    sns.barplot(x="cluster", y="total_purchases", data=customers, estimator="mean")
    plt.title("Среднее количество покупок по кластерам")
    plt.xlabel("Кластер")
    plt.ylabel("Среднее число покупок")
    plt.savefig(f"{report_dir}/cluster_purchases.png")
    plt.close()

    # 4. Heatmap of cluster means
    cluster_means = customers.groupby("cluster")[[
        "age", "personal_coef", "total_purchases",
        "total_spent", "discount_rate", "avg_product_sex"
    ]].mean()

    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_means, annot=True, cmap="YlGnBu", fmt=".1f")
    plt.title("Средние значения признаков по кластерам")
    plt.savefig(f"{report_dir}/cluster_heatmap.png")
    plt.close()

    print("✓ Cluster visualizations saved to /reports")

    return customers


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=== RUNNING CLUSTERING ===")
    run_clustering()
    print("=== CLUSTERING COMPLETED ===")
