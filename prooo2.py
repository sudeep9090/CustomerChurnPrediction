from pathlib import Path
from xgboost import XGBClassifier

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# ==========================================
# LOAD DATASET
# ==========================================

data_path = (
    Path(__file__).parent
    / "Customer_Churn_Prediction"
    / "CustomerChurn.csv"
)

df = pd.read_csv(data_path)

print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)

print(df.head())

# ==========================================
# DATASET INFORMATION
# ==========================================

print("\nDATASET INFO")
df.info()

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# ==========================================
# REMOVE DUPLICATES
# ==========================================

print("\nRemoving Duplicate Rows...")

print("Before:", df.shape)

df = df.drop_duplicates()

print("After:", df.shape)

# ==========================================
# EDA
# ==========================================

plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

df.hist(figsize=(15, 12))
plt.tight_layout()
plt.show()

# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop("Churn", axis=1)

y = df["Churn"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed")

# ==========================================
# LOGISTIC REGRESSION
# ==========================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

prob_lr = lr.predict_proba(X_test)[:, 1]

print("Accuracy :", accuracy_score(y_test, pred_lr))
print("Recall :", recall_score(y_test, pred_lr))
print("ROC AUC :", roc_auc_score(y_test, prob_lr))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, pred_lr))

print("\nClassification Report")
print(classification_report(y_test, pred_lr))
# ==========================================
# XGBOOST
# ==========================================

print("\n" + "=" * 60)
print("XGBOOST")
print("=" * 60)

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

pred_xgb = xgb.predict(X_test)

prob_xgb = xgb.predict_proba(X_test)[:,1]

print("Accuracy :", accuracy_score(y_test, pred_xgb))
print("Recall :", recall_score(y_test, pred_xgb))
print("ROC AUC :", roc_auc_score(y_test, prob_xgb))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, pred_xgb))

print("\nClassification Report")
print(classification_report(y_test, pred_xgb))

# ==========================================
# RANDOM FOREST
# ==========================================

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

prob_rf = rf.predict_proba(X_test)[:, 1]

print("Accuracy :", accuracy_score(y_test, pred_rf))
print("Recall :", recall_score(y_test, pred_rf))
print("ROC AUC :", roc_auc_score(y_test, prob_rf))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, pred_rf))

print("\nClassification Report")
print(classification_report(y_test, pred_rf))

# ==========================================
# ROC CURVE
# ==========================================

fpr1, tpr1, _ = roc_curve(y_test, prob_lr)
fpr2, tpr2, _ = roc_curve(y_test, prob_rf)
fpr3, tpr3, _ = roc_curve(y_test, prob_xgb)

plt.figure(figsize=(8, 6))

plt.plot(fpr1, tpr1, label="Logistic Regression")

plt.plot(fpr2, tpr2, label="Random Forest")
plt.plot(fpr3, tpr3, label="XGBoost")

plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.show()
# ==========================================
# MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy": [
        accuracy_score(y_test, pred_lr),
        accuracy_score(y_test, pred_rf),
        accuracy_score(y_test, pred_xgb)
    ],
    "Recall": [
        recall_score(y_test, pred_lr),
        recall_score(y_test, pred_rf),
        recall_score(y_test, pred_xgb)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, prob_lr),
        roc_auc_score(y_test, prob_rf),
        roc_auc_score(y_test, prob_xgb)
    ]
})

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison)
# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns,
).sort_values(ascending=False)

print("\nFeature Importance")

print(importance)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=importance.values,
    y=importance.index,
)

plt.title("Feature Importance")

plt.show()

joblib.dump(rf, "customer_churn_model.pkl")

print("\nBest model saved as customer_churn_model.pkl")

print("\nProject Completed Successfully!")