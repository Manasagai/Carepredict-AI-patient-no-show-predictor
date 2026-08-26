import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

DATA_PATH = "data/processed_appointments.csv"

if not os.path.exists(DATA_PATH):
    print("Processed dataset not found.")
    print("Please run preprocess.py first.")
    exit()

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)
print("Shape:", df.shape)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("No-show", axis=1)
y = df["No-show"]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 4. LOGISTIC REGRESSION
# ============================================================

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)
logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]


# ============================================================
# 5. RANDOM FOREST
# ============================================================

# Smaller and more efficient Random Forest
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(X_train, y_train)

rf_predictions = random_forest_model.predict(X_test)
rf_probabilities = random_forest_model.predict_proba(X_test)[:, 1]


# ============================================================
# 6. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, y_true, predictions, probabilities):

    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, predictions))

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc
    }


# ============================================================
# 7. EVALUATE BOTH MODELS
# ============================================================

logistic_results = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_predictions,
    logistic_probabilities
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions,
    rf_probabilities
)


# ============================================================
# 8. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame([
    logistic_results,
    rf_results
])

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison.to_string(index=False))


# ============================================================
# 9. SELECT BETTER MODEL
# ============================================================

if (
    rf_results["F1-score"] > logistic_results["F1-score"]
    and rf_results["ROC-AUC"] >= logistic_results["ROC-AUC"]
):
    selected_model = "Random Forest"
else:
    selected_model = "Logistic Regression"

print("\n" + "=" * 60)
print("SELECTED MODEL")
print("=" * 60)

print(selected_model)


# ============================================================
# 10. SAVE MODELS
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    logistic_model,
    "models/logistic_regression.pkl",
    compress=3
)

joblib.dump(
    random_forest_model,
    "models/random_forest.pkl",
    compress=3
)

joblib.dump(
    list(X.columns),
    "models/feature_columns.pkl"
)


print("\nModels saved successfully.")

print("\nRandom Forest model size:")

model_size_mb = (
    os.path.getsize("models/random_forest.pkl")
    / (1024 * 1024)
)

print(f"{model_size_mb:.2f} MB")

print("\nTraining completed successfully.")