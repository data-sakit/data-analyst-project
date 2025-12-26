import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import os

# ============================================================
# STEP 8 — GENDER MODEL TRAINING
# ============================================================

def train_gender_model(
    train_path="data_processed/X_train_prepared.csv",
    test_path="data_processed/X_test_prepared.csv",
    target_path="data_processed/y_train.csv",
    save_dir="data_processed"
):
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)
    y_train = pd.read_csv(target_path).iloc[:, 0]

    print("Loaded:")
    print(" - X_train:", X_train.shape)
    print(" - X_test:", X_test.shape)
    print(" - y_train:", y_train.shape)

    # Filter test by country = 32 (same as training)
    X_test_filtered = X_test[X_test["country"] == 32].copy()

    # Train/validation split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # ------------------------------
    # 1. Logistic Regression
    # ------------------------------
    logreg = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced"
    )
    logreg.fit(X_train_split, y_train_split)
    y_pred_logreg = logreg.predict(X_val_split)

    logreg_acc = accuracy_score(y_val_split, y_pred_logreg)
    logreg_f1 = f1_score(y_val_split, y_pred_logreg)

    # ------------------------------
    # 2. Random Forest
    # ------------------------------
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(X_train_split, y_train_split)
    y_pred_rf = rf.predict(X_val_split)

    rf_acc = accuracy_score(y_val_split, y_pred_rf)
    rf_f1 = f1_score(y_val_split, y_pred_rf)

    # ------------------------------
    # 3. Choose best model
    # ------------------------------
    if rf_f1 > logreg_f1:
        best_model_name = "Random Forest"
        final_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    else:
        best_model_name = "Logistic Regression"
        final_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        )

    # Train best model on full training data
    final_model.fit(X_train, y_train)

    # Train metrics
    y_train_pred = final_model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)

    # Predict on test
    test_predictions = final_model.predict(X_test_filtered)

    # Save predictions
    df_test = X_test_filtered.copy()
    df_test["gender_pred"] = test_predictions
    df_test["gender_label"] = df_test["gender_pred"].map({0: "мужчина", 1: "женщина"})

    df_test.to_csv(f"{save_dir}/gender_predictions.csv", index=False)

    # Print summary
    print("\n=== MODEL TRAINING SUMMARY ===")
    print("Best model:", best_model_name)
    print(f"Validation Logistic Regression: acc={logreg_acc:.4f}, f1={logreg_f1:.4f}")
    print(f"Validation Random Forest:       acc={rf_acc:.4f}, f1={rf_f1:.4f}")
    print(f"Train-final: acc={train_acc:.4f}, f1={train_f1:.4f}")
    print("Saved: gender_predictions.csv")

    return final_model


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=== RUNNING MODEL TRAINING ===")
    train_gender_model()
    print("=== MODEL TRAINING COMPLETED ===")
