import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def run_training():
    print("Starting Model Training - Red Wine Quality...")

    # 1. Load processed training data
    X_train_final = np.load("data/processed/X_train_final.npy")
    y_train = np.load("data/processed/y_train.npy")

    # 2. Train baseline Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    rf_model.fit(X_train_final, y_train)

    # 3. Train baseline Decision Tree
    dt_model = DecisionTreeClassifier(
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )
    dt_model.fit(X_train_final, y_train)

    # 4. Save models
    joblib.dump(rf_model, "models/random_forest_baseline.pkl")
    joblib.dump(dt_model, "models/decision_tree_baseline.pkl")

    print("[SUCCESS] Baseline models trained and saved to models/ directory!")


if __name__ == "__main__":
    run_training()
