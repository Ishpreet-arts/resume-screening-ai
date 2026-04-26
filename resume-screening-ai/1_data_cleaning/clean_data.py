"""
===========================================
  STEP 1: DATA CLEANING
  Resume Screening AI Project
===========================================
What this script does:
  - Loads the raw dataset
  - Removes unnecessary/empty columns
  - Cleans text (skills, career objective, responsibilities)
  - Handles missing values
  - Saves a clean version for the next step
"""

import pandas as pd
import numpy as np
import re
import os

# ─── Paths ────────────────────────────────────────────────
RAW_DATA_PATH   = "../data/resume_data.csv"
CLEAN_DATA_PATH = "../data/cleaned_resume_data.csv"

# ─── Load Data ────────────────────────────────────────────
print("📂 Loading raw data...")
df = pd.read_csv(RAW_DATA_PATH)

# Fix BOM character in column name
df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)

print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")

# ─── Select Useful Columns ────────────────────────────────
useful_cols = [
    'career_objective',
    'skills',
    'major_field_of_studies',
    'positions',
    'responsibilities',
    'job_position_name',
    'skills_required',
    'matched_score'
]

df = df[useful_cols]
print(f"\n✂️  Kept {len(useful_cols)} useful columns")

# ─── Drop Rows With No Score ──────────────────────────────
before = len(df)
df = df.dropna(subset=['matched_score', 'job_position_name'])
print(f"🗑️  Dropped {before - len(df)} rows missing score or job title")

# ─── Clean Text Function ──────────────────────────────────
def clean_text(text):
    """Remove special chars, extra spaces, lowercase everything."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"[\[\]'\"{}()]", " ", text)   # remove brackets/quotes
    text = re.sub(r"http\S+", " ", text)           # remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s,.]", " ", text) # keep letters/numbers
    text = re.sub(r"\s+", " ", text).strip()       # collapse whitespace
    return text.lower()

# ─── Apply Cleaning ───────────────────────────────────────
print("\n🧹 Cleaning text columns...")
text_cols = ['career_objective', 'skills', 'responsibilities',
             'skills_required', 'positions', 'major_field_of_studies']

for col in text_cols:
    df[col] = df[col].apply(clean_text)

df['job_position_name'] = df['job_position_name'].str.strip()

# ─── Create Combined Text Feature ─────────────────────────
print("🔗 Creating combined text feature...")
df['combined_text'] = (
    df['career_objective'] + " " +
    df['skills'] + " " +
    df['responsibilities'] + " " +
    df['positions']
)

# ─── Fill Remaining Nulls ─────────────────────────────────
df.fillna("", inplace=True)

# ─── Label: Score Buckets (for classification too) ────────
def score_label(score):
    if score >= 0.75:
        return "High Match"
    elif score >= 0.50:
        return "Medium Match"
    else:
        return "Low Match"

df['match_label'] = df['matched_score'].apply(score_label)

# ─── Save ─────────────────────────────────────────────────
os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
df.to_csv(CLEAN_DATA_PATH, index=False)

print(f"\n✅ Cleaned data saved → {CLEAN_DATA_PATH}")
print(f"📊 Final shape: {df.shape}")
print(f"\n📌 Match label distribution:\n{df['match_label'].value_counts()}")
print(f"\n📌 Top 5 job positions:\n{df['job_position_name'].value_counts().head()}")
print("\n🎉 Data Cleaning Complete!")
