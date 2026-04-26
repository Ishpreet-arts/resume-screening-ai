# 📄 Resume Screening AI

> An end-to-end Machine Learning project that predicts how well a resume matches a job requirement.

---

## 🗂️ Project Structure

```
resume-screening-ai/
│
├── data/
│   ├── resume_data.csv              ← Raw dataset
│   ├── cleaned_resume_data.csv      ← After cleaning (auto-generated)
│   └── charts/                      ← All visualization charts (auto-generated)
│
├── 1_data_cleaning/
│   └── clean_data.py                ← Cleans and prepares the data
│
├── 2_data_visualization/
│   └── visualize.py                 ← Creates 6 insightful charts
│
├── 3_model_training/
│   └── train_model.py               ← Trains regression + classifier models
│
├── 4_model_testing/
│   └── test_model.py                ← Evaluates models, confusion matrix
│
├── 5_deployment/
│   └── app.py                       ← Streamlit web application
│
├── models/                          ← Saved models (auto-generated)
│   ├── tfidf_vectorizer.pkl
│   ├── regression_model.pkl
│   └── classifier_model.pkl
│
└── requirements.txt
```

---

## 🚀 How to Run (Step by Step)

### Step 0 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 1 — Clean the Data
```bash
cd 1_data_cleaning
python clean_data.py
```

### Step 2 — Visualize the Data
```bash
cd 2_data_visualization
python visualize.py
```
> Charts will be saved in `data/charts/`

### Step 3 — Train the Models
```bash
cd 3_model_training
python train_model.py
```
> Models will be saved in `models/`

### Step 4 — Test the Models
```bash
cd 4_model_testing
python test_model.py
```

### Step 5 — Launch the Web App
```bash
cd 5_deployment
streamlit run app.py
```
> Opens in browser at http://localhost:8501

---

## 🧠 What the Models Do

| Model | Type | Output |
|-------|------|--------|
| Regression Model | Random Forest Regressor | Match score (0.0 – 1.0) |
| Classifier Model | Random Forest Classifier | High / Medium / Low Match |

---

## 📊 Visualizations Generated

1. Top 15 Job Positions in Dataset
2. Match Score Distribution (Histogram)
3. Match Label Pie Chart
4. Skills Word Cloud
5. Score by Category (Box Plot)
6. Top 20 Most Common Skills
7. Confusion Matrix (from testing)
8. Actual vs Predicted Scores (from testing)

---

## 💡 Tech Stack

- **Python** — Core language
- **Pandas / NumPy** — Data handling
- **Scikit-learn** — ML models + TF-IDF
- **Matplotlib / Seaborn** — Charts
- **WordCloud** — Skill word cloud
- **Streamlit** — Web app deployment

---

## 🎓 Key Concepts You'll Learn

- NLP text preprocessing
- TF-IDF vectorization
- Random Forest (regression + classification)
- Model evaluation (R², RMSE, accuracy, confusion matrix)
- Building and deploying an ML web app

---

*Built as a beginner ML portfolio project*
