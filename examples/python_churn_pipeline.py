# Synthetic subscription-churn-style pipeline using pandas + sklearn.

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def train_churn_model(csv_path: str = "subscriptions.csv") -> float:
    # Load subscription data
    df = pd.read_csv(csv_path)

    # Filter active users
    df = df[df["is_active"] == 1]

    # Create simple features
    df["sessions_per_month"] = df["sessions_last_30d"] / 1.0
    df["tenure_months"] = df["tenure_days"] / 30.0

    feature_cols = ["sessions_per_month", "tenure_months", "plan_tier"]
    X = df[feature_cols]
    y = df["churned"]

    # Train a basic classifier
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    # Evaluate with AUROC
    preds = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, preds)
    return auc


if __name__ == "__main__":
    auc_score = train_churn_model()
    print(f"Train AUROC: {auc_score:.3f}")
