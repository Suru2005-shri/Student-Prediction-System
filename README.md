#  Student Performance Prediction System v2.0

> End-to-end ML pipeline that predicts student grades, identifies at-risk learners, and generates personalized academic recommendations — built for Data Science & ML portfolio showcase.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  Overview

This project builds a complete, production-style machine learning system that predicts student academic performance (A/B/C/D/F) and flags students at risk of failing — using entirely **synthetic, public-domain data** that mimics real-world school datasets.

### Why this project matters

| Use Case | How it helps |
|---|---|
| Early intervention | Flags at-risk students before exams |
| Personalized learning | Recommends action items per student |
| Dropout prevention | Tracks engagement & attendance trends |
| Resource allocation | Helps schools target tutoring support |

---

##  Architecture

```
Student Data (1,000 records)
    │
    ▼
Data Generation (simulate_students)
    │  ← 17 raw features + 6 engineered
    ▼
Preprocessing Pipeline
    │  ← cleaning · encoding · scaling · imputation
    ▼
Feature Engineering
    │  ← engagement_score · study_x_attend · consistency · ...
    ▼
ML Model Training (4 models compared)
    │  ← Random Forest · Gradient Boosting · SVM · Logistic Regression
    ▼
Best Model Selection (Random Forest ~91% accuracy)
    │
    ├── Grade Prediction (A/B/C/D/F)
    ├── At-Risk Flag (binary)
    └── Personalized Recommendations
```

---

##  Features

### Input Features (17 raw + 6 engineered)

| Feature | Type | Description |
|---|---|---|
| `study_hours_per_day` | float | Daily study time |
| `attendance_pct` | int | Class attendance percentage |
| `previous_marks` | int | Marks in previous exam |
| `assignments_completed_pct` | int | Homework completion rate |
| `sleep_hours` | float | Nightly sleep duration |
| `extracurricular_activities` | int | Number of activities (0–5) |
| `tutoring` | binary | Whether student has a tutor |
| `internet_access` | binary | Home internet availability |
| `parental_education` | ordinal | Highest parental education level |
| `social_media_hours` | float | Daily social media use |
| `engagement_score`  | float | *Engineered*: weighted composite |
| `study_x_attend`  | float | *Engineered*: interaction feature |
| `consistency`  | float | *Engineered*: assignment vs marks gap |
| `attendance_band`  | int | *Engineered*: ordinal attendance tier |
| `healthy_sleep`  | binary | *Engineered*: 1 if 6–9 hours |
| `learning_support`  | float | *Engineered*: internet + tutoring composite |

### Target Variables

| Target | Type | Values |
|---|---|---|
| `grade` | Multi-class | A / B / C / D / F |
| `at_risk` | Binary | 0 = Safe, 1 = At Risk |
| `final_score` | Regression | 0–100 |

---

## 🤖 Model Performance

| Model | Accuracy | F1 (weighted) | CV Score |
|---|---|---|---|
| **Random Forest**  | **91.4%** | **0.902** | 90.8% ±1.2% |
| Gradient Boosting | 89.7% | 0.881 | 89.1% ±1.5% |
| SVM | 85.3% | 0.839 | 84.6% ±1.8% |
| Logistic Regression | 82.1% | 0.807 | 81.5% ±2.1% |

---

##  Folder Structure

```
Student-Performance-Prediction/
│
├── data/
│   └── student_performance.csv    ← generated dataset
│
├── src/
│   ├── generate_data.py           ← synthetic data simulator
│   ├── preprocess.py              ← cleaning + feature engineering
│   ├── train_models.py            ← 4 ML models + comparison
│   ├── predict.py                 ← inference + recommendations
│   └── visualize.py               ← EDA + model charts
│
├── models/
│   ├── best_model.pkl             ← serialized Random Forest
│   ├── scaler.pkl                 ← StandardScaler
│   ├── imputer.pkl                ← SimpleImputer
│   └── label_encoder.pkl          ← Grade LabelEncoder
│
├── outputs/
│   └── model_results.json         ← accuracy, F1, confusion matrices
│
├── images/
│   ├── score_distribution.png
│   ├── correlation_heatmap.png
│   ├── study_attendance_vs_score.png
│   ├── feature_importance.png
│   └── model_comparison.png
│
├── app.py                         ← Streamlit web app
├── main.py                        ← full pipeline runner
├── requirements.txt
└── README.md
```

---

##  Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/Student-Performance-Prediction.git
cd Student-Performance-Prediction
pip install -r requirements.txt
```

### 2. Run full pipeline

```bash
python main.py
```

This will:
- Generate 1,000 synthetic student records
- Preprocess and engineer features
- Train 4 ML models and compare them
- Save the best model
- Generate 5 visualizations
- Run a sample prediction

### 3. Launch interactive web app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 4. Predict for a custom student

```python
from src.predict import predict

result = predict({
    "gender": "female",
    "parental_education": "bachelors",
    "internet_access": "yes",
    "tutoring": "no",
    "study_hours_per_day": 6.0,
    "attendance_pct": 82,
    "previous_marks": 72,
    "assignments_completed_pct": 80,
    "sleep_hours": 7.5,
    "extracurricular_activities": 2,
    "tutoring_hours_per_week": 0,
    "social_media_hours": 2.5,
})

print(result["grade"])           # → "B"
print(result["at_risk"])         # → False
print(result["recommendations"]) # → [...]
```

---

##  Interview Talking Points

**Q: Why synthetic data?**
> Real student data is privacy-sensitive (FERPA/GDPR). Synthetic data lets us build the same pipeline, validate the architecture, and demonstrate ML skills without privacy issues — a standard approach at EdTech companies.

**Q: Why Random Forest over deep learning?**
> Tabular data with ~20 features rarely benefits from neural networks. Tree ensembles are interpretable, require no normalization tuning, handle mixed types, and provide feature importance natively — exactly what a school administrator needs to trust predictions.

**Q: How would you deploy this?**
> Package the `predict()` function behind a FastAPI endpoint, containerize with Docker, and deploy to AWS Lambda or a cloud VM. The Streamlit app demonstrates the UI layer already.


---

## 📈 Project Roadmap

- [x] Synthetic data generation with realistic correlations
- [x] Feature engineering pipeline
- [x] 4-model comparison with cross-validation
- [x] Recommendation engine
- [x] Streamlit dashboard
- [ ] FastAPI REST endpoint
- [ ] Docker containerization
- [ ] SHAP explainability plots
- [ ] Time-series performance tracking

---

##  Author

Built by **Shruti Srivastava** as a portfolio project for Data Science / ML Engineer roles.

- GitHub:(https://github.com/Suru2005-shri)
- LinkedIn: www.linkedin.com/in/shruti-srivastava-36b26232a
<p align="center">
  <img src="https://raw.githubusercontent.com/Suru2005-shri/Student-Prediction-System/main/images/correlation_heatmap.png" width="48%" alt="Correlation_heatmap" />
  <img src="https://raw.githubusercontent.com/Suru2005-shri/Student-Prediction-System/main/images/feature_importance.png" width="48%" alt="feature_importance" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Suru2005-shri/Student-Prediction-System/main/images/model_comparison.png" width="48%" alt="comparsiion"/>
  <img src="https://raw.githubusercontent.com/Suru2005-shri/Student-Prediction-System/main/images/parental_education_effect.png" width="48%" alt="Eduaction" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Suru2005-shri/Student-Prediction-System/main/images/score_distribution.png" width="70%" alt="score_distribution" />
</p>


---

##  License

MIT License — free to use, modify, and share.
