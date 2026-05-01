"""
app.py — Streamlit Web Application
------------------------------------
Run:  streamlit run app.py

Provides a browser-based interface for:
  • EDA dashboard
  • Live grade prediction with sliders
  • Model comparison table
  • At-risk student identification
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = Path("data/student_performance.csv")
RESULTS_PATH = Path("outputs/model_results.json")
MODELS_DIR = Path("models")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    # generate on the fly if not present
    from src.generate_data import simulate_students
    df = simulate_students(1000)
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


@st.cache_resource
def load_model():
    import joblib
    if not (MODELS_DIR / "best_model.pkl").exists():
        return None
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    imputer = joblib.load(MODELS_DIR / "imputer.pkl")
    le = joblib.load(MODELS_DIR / "label_encoder.pkl")
    return model, scaler, imputer, le


df = load_data()
artefacts = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Performance\nPrediction System v2.0")
    st.markdown("---")
    page = st.radio("Navigate", ["📊 Dashboard", "🔮 Live Predictor",
                                  "📈 EDA", "🤖 Model Insights", "⚠️ At-Risk Students"])
    st.markdown("---")
    st.markdown(f"**Dataset:** {len(df):,} students")
    st.markdown(f"**Features:** 17 + 6 engineered")
    st.markdown(f"**Best model:** Random Forest")
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            res = json.load(f)
        st.markdown(f"**Accuracy:** {res['best_accuracy']:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Dashboard
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.title("📊 Analytics Dashboard")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Students", f"{len(df):,}")
    col2.metric("Avg Score", f"{df['final_score'].mean():.1f}")
    col3.metric("At Risk", f"{df['at_risk'].sum()}")
    col4.metric("Top Performers", f"{(df['final_score'] >= 85).sum()}")
    col5.metric("Pass Rate", f"{df['pass_fail'].mean():.1%}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Grade Distribution")
        grade_counts = df["grade"].value_counts().reindex(["A","B","C","D","F"])
        st.bar_chart(grade_counts)

    with c2:
        st.subheader("Score Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df["final_score"], bins=20, color="#378ADD", edgecolor="white", alpha=0.85)
        ax.axvline(df["final_score"].mean(), color="red", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Average Score by Attendance Level")
    att_bands = pd.cut(df["attendance_pct"], bins=[0, 55, 70, 85, 100],
                       labels=["40-55%", "56-70%", "71-85%", "86-100%"])
    st.bar_chart(df.groupby(att_bands)["final_score"].mean())


# ─────────────────────────────────────────────────────────────────────────────
# Page: Live Predictor
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Live Predictor":
    st.title("🔮 Live Grade Predictor")
    if artefacts is None:
        st.warning("⚠️ Models not found. Run `python main.py` first.")
    else:
        st.markdown("Adjust the student parameters below and get an instant grade prediction.")
        c1, c2 = st.columns(2)
        with c1:
            study = st.slider("Study hours / day", 0.0, 12.0, 5.0, 0.5)
            attendance = st.slider("Attendance (%)", 30, 100, 75)
            prev_marks = st.slider("Previous marks", 10, 100, 65)
            assignments = st.slider("Assignments completed (%)", 0, 100, 70)
        with c2:
            sleep = st.slider("Sleep hours / night", 3.0, 12.0, 7.0, 0.5)
            extra = st.slider("Extracurricular activities", 0, 5, 2)
            social = st.slider("Social media hours / day", 0.0, 10.0, 3.0, 0.5)
            gender = st.selectbox("Gender", ["male", "female"])
            parental_edu = st.selectbox("Parental education",
                ["no_education","high_school","some_college","bachelors","masters"], index=3)
            internet = st.selectbox("Internet access", ["yes", "no"])
            tutoring = st.selectbox("Tutoring", ["no", "yes"])

        student = {
            "gender": gender, "parental_education": parental_edu,
            "internet_access": internet, "tutoring": tutoring,
            "study_hours_per_day": study, "attendance_pct": attendance,
            "previous_marks": prev_marks, "assignments_completed_pct": assignments,
            "sleep_hours": sleep, "extracurricular_activities": extra,
            "tutoring_hours_per_week": 2 if tutoring == "yes" else 0,
            "social_media_hours": social,
        }

        from src.predict import predict
        result = predict(student)
        grade = result["grade"]
        grade_colors = {"A":"#3B6D11","B":"#185FA5","C":"#854F0B","D":"#A32D2D","F":"#501313"}
        color = grade_colors.get(grade, "#333")

        st.markdown("---")
        st.markdown(f"### Prediction")
        col_g, col_r = st.columns(2)
        col_g.markdown(f"<h1 style='color:{color};font-size:64px;'>{grade}</h1>", unsafe_allow_html=True)
        col_r.metric("At Risk", "Yes ⚠️" if result["at_risk"] else "No ✅")

        st.markdown("**Grade Probabilities**")
        prob_df = pd.DataFrame(result["probabilities"].items(), columns=["Grade","Probability"])
        prob_df = prob_df.sort_values("Grade")
        st.bar_chart(prob_df.set_index("Grade"))

        st.markdown("**Recommendations**")
        for tip in result["recommendations"]:
            st.markdown(f"- {tip}")


# ─────────────────────────────────────────────────────────────────────────────
# Page: EDA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")

    st.subheader("Correlation Heatmap")
    num_cols = ["study_hours_per_day","attendance_pct","previous_marks",
                "assignments_completed_pct","sleep_hours","final_score"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5)
    st.pyplot(fig)

    st.subheader("Score vs Study Hours")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    grade_colors = {"A":"#3B6D11","B":"#185FA5","C":"#854F0B","D":"#A32D2D","F":"#501313"}
    for grade, grp in df.groupby("grade"):
        ax2.scatter(grp["study_hours_per_day"], grp["final_score"],
                    label=grade, color=grade_colors[grade], alpha=0.5, s=10)
    ax2.set_xlabel("Study hours / day")
    ax2.set_ylabel("Final score")
    ax2.legend(title="Grade")
    st.pyplot(fig2)

    st.subheader("Raw Data Sample")
    st.dataframe(df.sample(50).reset_index(drop=True), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Model Insights
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Model Insights":
    st.title("🤖 Model Insights")
    if not RESULTS_PATH.exists():
        st.warning("Run `python main.py` first to generate model results.")
    else:
        with open(RESULTS_PATH) as f:
            res = json.load(f)

        st.subheader("Model Comparison")
        models_df = pd.DataFrame([
            {"Model": k, "Accuracy": v["accuracy"], "F1": v["f1"]}
            for k, v in res["all_models"].items()
        ]).sort_values("Accuracy", ascending=False)
        st.dataframe(models_df.style.format({"Accuracy": "{:.1%}", "F1": "{:.4f}"}))

        if res["feature_importance"]:
            st.subheader("Feature Importance")
            fi = res["feature_importance"]
            fi_df = pd.Series(fi).sort_values(ascending=False).head(12)
            st.bar_chart(fi_df)


# ─────────────────────────────────────────────────────────────────────────────
# Page: At-Risk Students
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚠️ At-Risk Students":
    st.title("⚠️ At-Risk Student Dashboard")

    at_risk_df = df[df["at_risk"] == 1].copy()
    st.markdown(f"**{len(at_risk_df)} at-risk students detected** (score < 50)")

    c1, c2, c3 = st.columns(3)
    c1.metric("At Risk Count", len(at_risk_df))
    c2.metric("Avg Score (at risk)", f"{at_risk_df['final_score'].mean():.1f}")
    c3.metric("Avg Attendance", f"{at_risk_df['attendance_pct'].mean():.1f}%")

    st.subheader("Top risk factors")
    risk_factors = {
        "Low attendance (<60%)": (at_risk_df["attendance_pct"] < 60).sum(),
        "Low study hours (<3h)": (at_risk_df["study_hours_per_day"] < 3).sum(),
        "Low prev marks (<50)": (at_risk_df["previous_marks"] < 50).sum(),
        "High social media (>5h)": (at_risk_df["social_media_hours"] > 5).sum(),
        "No tutoring": (at_risk_df["tutoring"] == "no").sum(),
    }
    st.bar_chart(pd.Series(risk_factors))

    st.subheader("At-Risk Student Records")
    display_cols = ["student_id", "attendance_pct", "study_hours_per_day",
                    "previous_marks", "final_score", "grade"]
    st.dataframe(at_risk_df[display_cols].sort_values("final_score"),
                 use_container_width=True)
