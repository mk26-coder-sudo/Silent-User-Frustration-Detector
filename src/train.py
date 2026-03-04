import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_model():

    # 1. Load Dataset
    df = pd.read_csv("data/raw/user_behavior.csv")

    # 2. Separate Features and Target
    X = df.drop("frustrated", axis=1)
    y = df["frustrated"]

    # 3. Define Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 4. Cross Validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("\nCross Validation Accuracy Scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())

    # 5. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Train Model
    model.fit(X_train, y_train)

    # 7. Predictions
    y_pred = model.predict(X_test)

    # 8. Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 9. Feature Importance
    importances = model.feature_importances_
    feature_names = X.columns

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance:\n")
    print(importance_df)

    plt.figure()
    plt.bar(importance_df["Feature"], importance_df["Importance"])
    plt.xticks(rotation=90)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

    # 10. Save Model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "models/frustration_model.pkl")

    print("Model saved successfully!")

    return model


if __name__ == "__main__":
    train_model()