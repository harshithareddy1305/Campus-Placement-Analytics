import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv("../data/Placement_Data_Full_Class.csv")
print("Original Dataset Shape:", df.shape)

# Drop ID column
df.drop(columns=["sl_no"], inplace=True)

# ============================================
# FEATURE ENGINEERING
# ============================================

# Binary target for classification
df["placed_binary"] = df["status"].map({"Placed": 1, "Not Placed": 0})

# Average academic score
df["avg_academic_score"] = (
    df["ssc_p"] +
    df["hsc_p"] +
    df["degree_p"] +
    df["mba_p"]
) / 4

# Work experience binary
df["has_workex"] = df["workex"].map({"Yes": 1, "No": 0})

# Drop original categorical column
df.drop(columns=["workex"], inplace=True)

# ============================================
# ENCODE CATEGORICAL FEATURES
# ============================================

cat_cols = [
    "gender",
    "ssc_b",
    "hsc_b",
    "hsc_s",
    "degree_t",
    "specialisation"
]

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# SAFETY CHECK — MUST BE EMPTY
print("Remaining object columns:", df.select_dtypes(include="object").columns)

# ============================================
# PART 1: CLASSIFICATION (ML CORE)
# Predict Placement Outcome
# ============================================

X_cls = df.drop(columns=["status", "salary", "placed_binary"])
y_cls = df["placed_binary"]

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls,
    y_cls,
    test_size=0.2,
    random_state=42,
    stratify=y_cls
)

# Logistic Regression (Baseline)
log_clf = LogisticRegression(max_iter=1000)
log_clf.fit(X_train_cls, y_train_cls)
log_pred = log_clf.predict(X_test_cls)

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_cls, y_train_cls)
rf_pred = rf_clf.predict(X_test_cls)

print("\n=== PLACEMENT CLASSIFICATION RESULTS ===")

print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(y_test_cls, log_pred))
print("Precision:", precision_score(y_test_cls, log_pred))
print("Recall:", recall_score(y_test_cls, log_pred))
print("F1 Score:", f1_score(y_test_cls, log_pred))

print("\nRandom Forest Classifier:")
print("Accuracy:", accuracy_score(y_test_cls, rf_pred))
print("Precision:", precision_score(y_test_cls, rf_pred))
print("Recall:", recall_score(y_test_cls, rf_pred))
print("F1 Score:", f1_score(y_test_cls, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_cls, rf_pred))

# ============================================
# PART 2: REGRESSION (FEASIBILITY STUDY)
# Predict Salary for Placed Students
# ============================================

df_reg = df[df["placed_binary"] == 1].copy()
df_reg.dropna(subset=["salary"], inplace=True)

X_reg = df_reg.drop(columns=["salary", "status", "placed_binary"])
y_reg = df_reg["salary"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)
lr_pred = lr.predict(X_test_reg)

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)
rf_reg_pred = rf_reg.predict(X_test_reg)

print("\n=== SALARY REGRESSION RESULTS ===")
print(f"Linear Regression  | MAE: {mean_absolute_error(y_test_reg, lr_pred):.2f} | R²: {r2_score(y_test_reg, lr_pred):.2f}")
print(f"Random Forest      | MAE: {mean_absolute_error(y_test_reg, rf_reg_pred):.2f} | R²: {r2_score(y_test_reg, rf_reg_pred):.2f}")

# ============================================
# FEATURE IMPORTANCE (INTERPRETABILITY)
# ============================================

importances = rf_clf.feature_importances_
features = X_cls.columns

feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_imp.head(10).plot(kind="bar")
plt.title("Top Feature Importances - Placement Classification")
plt.tight_layout()
plt.savefig("../images/plots/feature_importance_classification.png")
plt.close()

print("\nTop Influential Features:\n", feat_imp.head(10))
# ============================================
# IMPORTS
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================
# LOAD DATA
# ============================================

df = pd.read_csv("../data/Placement_Data_Full_Class.csv")
print("Original Dataset Shape:", df.shape)

# Drop ID column
df.drop(columns=["sl_no"], inplace=True)

# ============================================
# FEATURE ENGINEERING
# ============================================

# Binary target for classification
df["placed_binary"] = df["status"].map({"Placed": 1, "Not Placed": 0})

# Average academic score
df["avg_academic_score"] = (
    df["ssc_p"] +
    df["hsc_p"] +
    df["degree_p"] +
    df["mba_p"]
) / 4

# Work experience binary
df["has_workex"] = df["workex"].map({"Yes": 1, "No": 0})

# Drop original categorical column
df.drop(columns=["workex"], inplace=True)

# ============================================
# ENCODE CATEGORICAL FEATURES
# ============================================

cat_cols = [
    "gender",
    "ssc_b",
    "hsc_b",
    "hsc_s",
    "degree_t",
    "specialisation"
]

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# SAFETY CHECK — MUST BE EMPTY
print("Remaining object columns:", df.select_dtypes(include="object").columns)

# ============================================
# PART 1: CLASSIFICATION (ML CORE)
# Predict Placement Outcome
# ============================================

X_cls = df.drop(columns=["status", "salary", "placed_binary"])
y_cls = df["placed_binary"]

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls,
    y_cls,
    test_size=0.2,
    random_state=42,
    stratify=y_cls
)

# Logistic Regression (Baseline)
log_clf = LogisticRegression(max_iter=1000)
log_clf.fit(X_train_cls, y_train_cls)
log_pred = log_clf.predict(X_test_cls)

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_cls, y_train_cls)
rf_pred = rf_clf.predict(X_test_cls)

print("\n=== PLACEMENT CLASSIFICATION RESULTS ===")

print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(y_test_cls, log_pred))
print("Precision:", precision_score(y_test_cls, log_pred))
print("Recall:", recall_score(y_test_cls, log_pred))
print("F1 Score:", f1_score(y_test_cls, log_pred))

print("\nRandom Forest Classifier:")
print("Accuracy:", accuracy_score(y_test_cls, rf_pred))
print("Precision:", precision_score(y_test_cls, rf_pred))
print("Recall:", recall_score(y_test_cls, rf_pred))
print("F1 Score:", f1_score(y_test_cls, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_cls, rf_pred))

# ============================================
# PART 2: REGRESSION (FEASIBILITY STUDY)
# Predict Salary for Placed Students
# ============================================

df_reg = df[df["placed_binary"] == 1].copy()
df_reg.dropna(subset=["salary"], inplace=True)

X_reg = df_reg.drop(columns=["salary", "status", "placed_binary"])
y_reg = df_reg["salary"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)
lr_pred = lr.predict(X_test_reg)

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)
rf_reg_pred = rf_reg.predict(X_test_reg)

print("\n=== SALARY REGRESSION RESULTS ===")
print(f"Linear Regression  | MAE: {mean_absolute_error(y_test_reg, lr_pred):.2f} | R²: {r2_score(y_test_reg, lr_pred):.2f}")
print(f"Random Forest      | MAE: {mean_absolute_error(y_test_reg, rf_reg_pred):.2f} | R²: {r2_score(y_test_reg, rf_reg_pred):.2f}")

# ============================================
# FEATURE IMPORTANCE (INTERPRETABILITY)
# ============================================

importances = rf_clf.feature_importances_
features = X_cls.columns

feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_imp.head(10).plot(kind="bar")
plt.title("Top Feature Importances - Placement Classification")
plt.tight_layout()
plt.savefig("../images/plots/feature_importance_classification.png")
plt.close()

print("\nTop Influential Features:\n", feat_imp.head(10))
