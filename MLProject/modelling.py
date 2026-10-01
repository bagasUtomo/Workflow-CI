
import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print("=== MLflow Project: Model Training ===")

train_path = "heart-disease_preprocessing/heart-disease_train.csv"
test_path = "heart-disease_preprocessing/heart-disease_test.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

X_train = train_data.drop("target", axis=1)
y_train = train_data["target"]

X_test = test_data.drop("target", axis=1)
y_test = test_data["target"]

print(f"Training data : {X_train.shape}")
print(f"Testing data  : {X_test.shape}")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("Training Random Forest...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n=== Evaluation Results ===")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {roc_auc:.4f}")

os.makedirs("artifacts", exist_ok=True)

model_path = "artifacts/random_forest_model.joblib"

joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")
print("=== Training completed successfully ===")
