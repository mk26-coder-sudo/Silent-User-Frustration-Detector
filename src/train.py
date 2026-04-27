import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


def train_model():

    # 1. Load Dataset
    df = pd.read_csv("data/raw/user_behavior.csv")

    # 2. Separate Features and Target
    X = df.drop("frustrated", axis=1)
    y = df["frustrated"]

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Random Forest ──────────────────────────────────────────
    rf_model = RandomForestClassifier(class_weight="balanced", random_state=42)

    rf_cv = cross_val_score(rf_model, X, y, cv=5)
    print("\nRF Cross Validation Mean:", rf_cv.mean())

    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    print("\n── Random Forest ──")
    print("Accuracy:", accuracy_score(y_test, rf_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
    print("Classification Report:\n", classification_report(y_test, rf_pred))

    # ── Logistic Regression ────────────────────────────────────
    lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

    lr_cv = cross_val_score(lr_model, X, y, cv=5)
    print("\nLR Cross Validation Mean:", lr_cv.mean())

    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    print("\n── Logistic Regression ──")
    print("Accuracy:", accuracy_score(y_test, lr_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
    print("Classification Report:\n", classification_report(y_test, lr_pred))

    # ── Model Comparison Table ─────────────────────────────────
    comparison = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression"],
        "Accuracy": [
            round(accuracy_score(y_test, rf_pred), 2),
            round(accuracy_score(y_test, lr_pred), 2)
        ],
        "F1 (Frustrated)": [
            round(f1_score(y_test, rf_pred), 2),
            round(f1_score(y_test, lr_pred), 2)
        ],
        "CV Mean Accuracy": [
            round(rf_cv.mean(), 2),
            round(lr_cv.mean(), 2)
        ],
        "Interpretable": ["SHAP", "Coefficients"]
    })

    print("\n── Model Comparison ──")
    print(comparison)

    # ── Feature Importance (RF) ────────────────────────────────
    importances = rf_model.feature_importances_
    feature_names = X.columns

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance:\n", importance_df)

    plt.figure()
    plt.bar(importance_df["Feature"], importance_df["Importance"])
    plt.xticks(rotation=90)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

    # ── Save Everything ────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, "models/frustration_model.pkl")
    joblib.dump(lr_model, "models/lr_model.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_names.pkl")
    comparison.to_csv("models/model_comparison.csv", index=False)  # dashboard reads this

    print("\nAll models saved successfully!")
    print("Features:", X.columns.tolist())

    return rf_model


if __name__ == "__main__":
    train_model()