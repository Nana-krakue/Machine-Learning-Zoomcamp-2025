# Binary Classification Model Evaluation Metrics  
---

## 📘 Overview

This repository provides detailed notes and Python code examples for **evaluating binary classification models**.  
It covers the most important metrics — accuracy, precision, recall, F1, and ROC–AUC — and explains how to use **cross-validation and hyperparameter tuning** to build robust models.

These notes and scripts are inspired by **Machine Learning Zoomcamp – Module 4: Evaluation Metrics for Classification**.

---

## 🎯 Why Evaluation Metrics Matter

A **metric** is a single number that quantifies a model’s performance.  
Different metrics emphasize different aspects of performance.  
Understanding them helps you:
- Detect **class imbalance** problems
- Evaluate **precision vs recall trade-offs**
- Tune your model for **real-world objectives**
- Ensure **generalization** through cross-validation

---

## 🧮 Key Metrics Explained

### **1. Accuracy**
Measures the overall correctness of the model’s predictions.

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

✅ **Use when:** classes are balanced  
⚠️ **Avoid when:** one class dominates (class imbalance)

---

### **2. Confusion Matrix**

|                | Predicted Positive | Predicted Negative |
|----------------|-------------------|-------------------|
| Actual Positive | TP | FN |
| Actual Negative | FP | TN |

- **TP (True Positive):** Model correctly predicts positive  
- **TN (True Negative):** Model correctly predicts negative  
- **FP (False Positive):** Model incorrectly predicts positive  
- **FN (False Negative):** Model incorrectly predicts negative  

It helps visualize *where* your model is making mistakes.

---

### **3. Precision and Recall**

#### **Precision**
\[
\text{Precision} = \frac{TP}{TP + FP}
\]

Measures how well the model avoids **false positives**.  
✅ High precision → few false alarms.

#### **Recall (Sensitivity)**
\[
\text{Recall} = \frac{TP}{TP + FN}
\]

Measures how well the model avoids **false negatives**.  
✅ High recall → fewer missed positive cases.

Use **Precision–Recall trade-off** when false positives and false negatives have different costs.

---

### **4. F1-Score**

\[
\text{F1} = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

A harmonic mean of precision and recall — balances both metrics.

✅ **Use when:** you care equally about precision and recall  
⚠️ **Avoid when:** you can optimize one metric clearly over the other

---

### **5. ROC Curve and AUC**

The **ROC (Receiver Operating Characteristic)** curve plots:

\[
\text{TPR (Recall)} = \frac{TP}{TP + FN}
\]
\[
\text{FPR} = \frac{FP}{FP + TN}
\]

The **AUC (Area Under Curve)** summarizes performance:
- **AUC = 1.0:** Perfect model  
- **AUC = 0.5:** Random guessing  

✅ **Use when:** evaluating ranking or discrimination power

---

## ⚙️ Parameter Tuning and Cross-Validation

Hyperparameter tuning optimizes a model’s parameters (e.g., regularization strength `C` in logistic regression).  
To avoid overfitting to a single validation set, we use **K-Fold Cross-Validation**.

### **K-Fold Cross-Validation Steps**
1. Split data into \(k\) folds (e.g., \(k=5\)).  
2. Train on \(k-1\) folds, validate on 1 fold.  
3. Repeat \(k\) times.  
4. Average the metric (e.g., AUC, accuracy) across all folds.  

✅ Provides a stable estimate of performance  
✅ Detects overfitting and variance issues

---

## 💻 Python Example (Logistic Regression with Cross-Validation)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 1️⃣ Generate synthetic data (imbalanced)
X, y = make_classification(
    n_samples=1000, n_features=20,
    n_informative=5, n_redundant=2,
    weights=[0.9, 0.1], random_state=42
)

# 2️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3️⃣ Pipeline: Scaling + Model
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(solver='liblinear', random_state=42))
])

# 4️⃣ Hyperparameter tuning using 5-fold CV
param_grid = {'clf__C': [0.01, 0.1, 1, 10, 100]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc')
grid.fit(X_train, y_train)

print("Best C:", grid.best_params_)
print("Best CV AUC:", grid.best_score_)

# 5️⃣ Evaluate on test set
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_proba)
}
print(pd.Series(metrics))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6️⃣ Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend(loc='best')
plt.show()
