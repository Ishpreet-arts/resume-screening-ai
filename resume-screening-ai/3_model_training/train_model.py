"""
===========================================
  STEP 3: MODEL TRAINING
  Resume Screening AI Project
===========================================
What this script does:
  - Loads cleaned data
  - Converts text to numbers using TF-IDF
  - Trains TWO models:
      1. Regression  → predicts exact match score (0-1)
      2. Classifier  → predicts High / Medium / Low match
  - Saves both models + the TF-IDF vectorizer
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, classification_report)
import warnings
warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────
DATA_PATH   = "../data/cleaned_resume_data.csv"
MODELS_DIR  = "../models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Load Data ────────────────────────────────────────────
print("📂 Loading cleaned data...")
df = pd.read_csv(DATA_PATH)
df['combined_text'] = df['combined_text'].fillna("").astype(str)
df['skills_required'] = df['skills_required'].fillna("").astype(str)
df['matched_score'] = pd.to_numeric(df['matched_score'], errors='coerce')
df = df.dropna(subset=['matched_score'])
print(f"✅ {len(df)} rows ready for training")

# ─── Feature Engineering ──────────────────────────────────
print("\n🔢 Vectorizing text with TF-IDF...")

# Combine resume text + job requirements for richer features
df['full_input'] = df['combined_text'] + " " + df['skills_required']

X = df['full_input']
y_reg   = df['matched_score']            # regression target
y_class = df['match_label']              # classification target

# TF-IDF: converts text → numerical matrix
tfidf = TfidfVectorizer(
    max_features=3000,     # keep top 3000 words
    ngram_range=(1, 2),    # single words + pairs
    sublinear_tf=True      # smooths word frequency
)
X_tfidf = tfidf.fit_transform(X)
print(f"✅ TF-IDF matrix shape: {X_tfidf.shape}")

# ─── Train / Test Split ───────────────────────────────────
X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X_tfidf, y_reg, y_class,
    test_size=0.2,
    random_state=42
)
print(f"\n📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Testing set:  {X_test.shape[0]} samples")

# ─── MODEL 1: Regression (predict score) ──────────────────
print("\n" + "="*50)
print("🤖 Training Model 1: Score Predictor (Regression)")
print("="*50)

reg_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
reg_model.fit(X_train, y_reg_train)
y_reg_pred = reg_model.predict(X_test)

mse = mean_squared_error(y_reg_test, y_reg_pred)
r2  = r2_score(y_reg_test, y_reg_pred)
print(f"\n📈 Regression Results:")
print(f"   ✅ R² Score (higher = better): {r2:.4f}")
print(f"   ✅ RMSE  (lower = better):     {np.sqrt(mse):.4f}")

# ─── MODEL 2: Classifier (High/Med/Low) ───────────────────
print("\n" + "="*50)
print("🤖 Training Model 2: Match Label Classifier")
print("="*50)

cls_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
cls_model.fit(X_train, y_cls_train)
y_cls_pred = cls_model.predict(X_test)

acc = accuracy_score(y_cls_test, y_cls_pred)
print(f"\n📈 Classifier Results:")
print(f"   ✅ Accuracy: {acc*100:.2f}%")
print(f"\n📋 Detailed Report:\n")
print(classification_report(y_cls_test, y_cls_pred))

# ─── Save All Models ──────────────────────────────────────
print("\n💾 Saving models...")

with open(f"{MODELS_DIR}/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
print("   ✅ Saved: tfidf_vectorizer.pkl")

with open(f"{MODELS_DIR}/regression_model.pkl", "wb") as f:
    pickle.dump(reg_model, f)
print("   ✅ Saved: regression_model.pkl")

with open(f"{MODELS_DIR}/classifier_model.pkl", "wb") as f:
    pickle.dump(cls_model, f)
print("   ✅ Saved: classifier_model.pkl")

# Save test data for testing script
test_data = {
    "X_test": X_test,
    "y_reg_test": y_reg_test,
    "y_cls_test": y_cls_test
}
with open(f"{MODELS_DIR}/test_data.pkl", "wb") as f:
    pickle.dump(test_data, f)
print("   ✅ Saved: test_data.pkl")

print(f"\n🎉 Training Complete! All models saved to → {MODELS_DIR}/")
