"""
===========================================
  STEP 4: MODEL TESTING
  Resume Screening AI Project
===========================================
What this script does:
  - Loads saved models
  - Evaluates on test data
  - Shows confusion matrix
  - Plots actual vs predicted scores
  - Tests with a custom sample resume
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             mean_squared_error, r2_score)
import os
import warnings
warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────
MODELS_DIR  = "../models"
CHARTS_DIR  = "../data/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# ─── Load Models ──────────────────────────────────────────
print("📂 Loading saved models...")
with open(f"{MODELS_DIR}/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open(f"{MODELS_DIR}/regression_model.pkl", "rb") as f:
    reg_model = pickle.load(f)
with open(f"{MODELS_DIR}/classifier_model.pkl", "rb") as f:
    cls_model = pickle.load(f)
with open(f"{MODELS_DIR}/test_data.pkl", "rb") as f:
    test_data = pickle.load(f)

X_test      = test_data["X_test"]
y_reg_test  = test_data["y_reg_test"]
y_cls_test  = test_data["y_cls_test"]
print("✅ Models loaded successfully!")

# ─── Regression Evaluation ────────────────────────────────
print("\n" + "="*50)
print("📊 Regression Model Evaluation")
print("="*50)
y_reg_pred = reg_model.predict(X_test)
r2  = r2_score(y_reg_test, y_reg_pred)
mse = mean_squared_error(y_reg_test, y_reg_pred)
print(f"R² Score : {r2:.4f}")
print(f"RMSE     : {np.sqrt(mse):.4f}")
print(f"MAE      : {np.mean(np.abs(y_reg_test - y_reg_pred)):.4f}")

# ─── Classifier Evaluation ────────────────────────────────
print("\n" + "="*50)
print("📊 Classifier Model Evaluation")
print("="*50)
y_cls_pred = cls_model.predict(X_test)
print(classification_report(y_cls_test, y_cls_pred))

# ─── CHART 7: Confusion Matrix ────────────────────────────
print("📊 Generating Confusion Matrix...")
labels = ["High Match", "Low Match", "Medium Match"]
cm = confusion_matrix(y_cls_test, y_cls_pred, labels=labels)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor='gray')
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("Actual Label", fontsize=12)
ax.set_title("Confusion Matrix — Match Label Classifier", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/7_confusion_matrix.png", dpi=150)
plt.close()
print("   ✅ Saved: 7_confusion_matrix.png")

# ─── CHART 8: Actual vs Predicted Scores ─────────────────
print("📊 Generating Actual vs Predicted chart...")
sample_idx = np.random.choice(len(y_reg_test), size=200, replace=False)
actual   = np.array(y_reg_test)[sample_idx]
predicted = y_reg_pred[sample_idx]

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(actual, predicted, alpha=0.5, color='#4C72B0', edgecolors='white', s=50)
ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Perfect Prediction')
ax.set_xlabel("Actual Match Score", fontsize=12)
ax.set_ylabel("Predicted Match Score", fontsize=12)
ax.set_title("Actual vs Predicted Match Scores", fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/8_actual_vs_predicted.png", dpi=150)
plt.close()
print("   ✅ Saved: 8_actual_vs_predicted.png")

# ─── TEST WITH CUSTOM RESUME ──────────────────────────────
print("\n" + "="*50)
print("🧪 Testing With a Sample Resume")
print("="*50)

sample_resume = """
Python machine learning data science tensorflow keras scikit-learn
pandas numpy deep learning nlp natural language processing
computer vision sql database management bachelor engineering
data analyst experience projects classification regression
"""

sample_job_req = """
python machine learning deep learning data science sql experience required
"""

sample_input = sample_resume + " " + sample_job_req
X_sample = tfidf.transform([sample_input])

pred_score  = reg_model.predict(X_sample)[0]
pred_label  = cls_model.predict(X_sample)[0]

print(f"\n📄 Resume Summary: Data Science / ML skills")
print(f"💼 Job Requirements: ML Engineer")
print(f"\n🎯 Predicted Match Score : {pred_score:.2f} / 1.00")
print(f"🏷️  Predicted Match Label : {pred_label}")

if pred_score >= 0.75:
    print("💚 Result: STRONG CANDIDATE — Recommend for interview!")
elif pred_score >= 0.50:
    print("🟡 Result: MODERATE FIT — Worth a closer look")
else:
    print("🔴 Result: LOW MATCH — Likely not the right fit")

print(f"\n🎉 Testing Complete! Charts saved to → {CHARTS_DIR}/")
