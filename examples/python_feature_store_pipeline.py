# Synthetic example showing a feature-store-like client and model training.

class FeatureStoreClient:
    def load_features(self, entity_ids, feature_names):
        # Dummy implementation; in real life this would query a feature store.
        return {
            eid: {fname: 1.0 for fname in feature_names}
            for eid in entity_ids
        }


def train_purchase_model(entity_ids):
    fs = FeatureStoreClient()
    feature_names = ["user_activity_7d", "user_tenure_days"]

    feature_data = fs.load_features(entity_ids, feature_names)

    # Convert to simple lists for demonstration
    X = [[vals[fname] for fname in feature_names] for vals in feature_data.values()]
    y = [0 for _ in entity_ids]  # synthetic target

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(X, y)
    return model


if __name__ == "__main__":
    train_purchase_model([1, 2, 3])
