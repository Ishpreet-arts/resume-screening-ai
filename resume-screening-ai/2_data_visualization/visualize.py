"""
===========================================
  STEP 2: DATA VISUALIZATION
  Resume Screening AI Project
===========================================
What this script does:
  - Job category distribution (bar chart)
  - Match score distribution (histogram)
  - Top skills word cloud
  - Match label pie chart
  - Score by job position (box plot)
  - Saves all charts to a /charts/ folder
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
import os
import warnings
warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────
DATA_PATH   = "../data/cleaned_resume_data.csv"
CHARTS_DIR  = "../data/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# ─── Load Data ────────────────────────────────────────────
print("📂 Loading cleaned data...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {df.shape[0]} rows")

# Color palette
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
          "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

sns.set_theme(style="whitegrid", palette="muted")

# ─── 1. Top 15 Job Positions ──────────────────────────────
print("\n📊 Chart 1: Top Job Positions...")
top_jobs = df['job_position_name'].value_counts().head(15)

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(top_jobs.index[::-1], top_jobs.values[::-1], color=COLORS[0], edgecolor='white')
for bar in bars:
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            str(int(bar.get_width())), va='center', fontsize=9)
ax.set_xlabel("Number of Resumes", fontsize=12)
ax.set_title("Top 15 Job Positions in Dataset", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/1_top_job_positions.png", dpi=150)
plt.close()
print("   ✅ Saved: 1_top_job_positions.png")

# ─── 2. Match Score Distribution ─────────────────────────
print("📊 Chart 2: Match Score Distribution...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['matched_score'], bins=30, color=COLORS[1], edgecolor='white', alpha=0.85)
ax.axvline(df['matched_score'].mean(), color='red', linestyle='--',
           linewidth=1.5, label=f"Mean: {df['matched_score'].mean():.2f}")
ax.set_xlabel("Match Score", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Distribution of Resume Match Scores", fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/2_match_score_distribution.png", dpi=150)
plt.close()
print("   ✅ Saved: 2_match_score_distribution.png")

# ─── 3. Match Label Pie Chart ─────────────────────────────
print("📊 Chart 3: Match Label Pie Chart...")
label_counts = df['match_label'].value_counts()
fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    label_counts.values,
    labels=label_counts.index,
    autopct='%1.1f%%',
    colors=[COLORS[2], COLORS[0], COLORS[3]],
    startangle=140,
    wedgeprops=dict(edgecolor='white', linewidth=2)
)
for text in autotexts:
    text.set_fontsize(12)
ax.set_title("Resume Match Label Distribution", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/3_match_label_pie.png", dpi=150)
plt.close()
print("   ✅ Saved: 3_match_label_pie.png")

# ─── 4. Word Cloud of Skills ──────────────────────────────
print("📊 Chart 4: Skills Word Cloud...")
all_skills = " ".join(df['skills'].dropna().astype(str).tolist())
wordcloud = WordCloud(
    width=1200, height=600,
    background_color='white',
    colormap='Blues',
    max_words=100,
    collocations=False
).generate(all_skills)

fig, ax = plt.subplots(figsize=(14, 7))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
ax.set_title("Most Common Skills in Resumes", fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/4_skills_wordcloud.png", dpi=150)
plt.close()
print("   ✅ Saved: 4_skills_wordcloud.png")

# ─── 5. Box Plot: Score by Match Label ────────────────────
print("📊 Chart 5: Score by Match Label...")
fig, ax = plt.subplots(figsize=(9, 6))
order = ["Low Match", "Medium Match", "High Match"]
palette = {"Low Match": COLORS[3], "Medium Match": COLORS[0], "High Match": COLORS[2]}
sns.boxplot(data=df, x='match_label', y='matched_score',
            order=order, palette=palette, ax=ax, width=0.5)
ax.set_xlabel("Match Category", fontsize=12)
ax.set_ylabel("Match Score", fontsize=12)
ax.set_title("Match Score by Category", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/5_score_by_label_boxplot.png", dpi=150)
plt.close()
print("   ✅ Saved: 5_score_by_label_boxplot.png")

# ─── 6. Top Skills Bar Chart ──────────────────────────────
print("📊 Chart 6: Top Individual Skills...")
from collections import Counter
skills_flat = []
for s in df['skills'].dropna():
    skills_flat.extend([x.strip() for x in str(s).split(',') if len(x.strip()) > 2])

top_skills = pd.Series(Counter(skills_flat)).sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(top_skills.index[::-1], top_skills.values[::-1], color=COLORS[4], edgecolor='white')
ax.set_xlabel("Frequency", fontsize=12)
ax.set_title("Top 20 Most Mentioned Skills", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/6_top_skills_bar.png", dpi=150)
plt.close()
print("   ✅ Saved: 6_top_skills_bar.png")

print(f"\n🎉 All 6 charts saved to → {CHARTS_DIR}/")
print("You can open them from the data/charts/ folder!")
