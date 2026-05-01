"""
app.py — Streamlit Web Application
------------------------------------
Run:  streamlit run app.py
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global style ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', 'Inter', sans-serif;
}
section[data-testid="stSidebar"] {
    background-color: #0f1117;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-size: 13px;
    letter-spacing: 0.02em;
    padding: 4px 0;
}
.main .block-container {
    background-color: #0d1117;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1 { font-size: 26px !important; font-weight: 600 !important;
     color: #e6edf3 !important; letter-spacing: -0.02em; }
h2 { font-size: 18px !important; font-weight: 500 !important;
     color: #c9d1d9 !important; }
h3 { font-size: 14px !important; font-weight: 500 !important;
     color: #8b949e !important; text-transform: uppercase;
     letter-spacing: 0.08em; }
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px 20px !important;
}
[data-testid="metric-container"] label {
    font-size: 11px !important;
    color: #8b949e !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 600 !important;
    color: #e6edf3 !important;
}
hr { border-color: #21262d !important; margin: 1.5rem 0; }
.stDataFrame { border: 1px solid #21262d !important; border-radius: 8px; }
.stAlert {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Palette ───────────────────────────────────────────────────────────────────
GRADE_COLORS = {"A": "#3fb950", "B": "#58a6ff", "C": "#d29922",
                "D": "#f85149", "F": "#8b1a1a"}
CHART_BG    = "#0d1117"
AXES_BG     = "#161b22"
TEXT_COLOR  = "#8b949e"
GRID_COLOR  = "#21262d"
ACCENT      = "#58a6ff"

matplotlib.rcParams.update({
    "figure.facecolor":  CHART_BG,
    "axes.facecolor":    AXES_BG,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlecolor":   "#c9d1d9",
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "text.color":        "#c9d1d9",
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.facecolor":  AXES_BG,
    "legend.edgecolor":  GRID_COLOR,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = Path("data/student_performance.csv")
RESULTS_PATH = Path("outputs/model_results.json")
MODELS_DIR   = Path("models")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
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
    return (
        joblib.load(MODELS_DIR / "best_model.pkl"),
        joblib.load(MODELS_DIR / "scaler.pkl"),
        joblib.load(MODELS_DIR / "imputer.pkl"),
        joblib.load(MODELS_DIR / "label_encoder.pkl"),
    )

df        = load_data()
artefacts = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<p style='font-size:11px;letter-spacing:0.1em;color:#8b949e;"
        "text-transform:uppercase;margin-bottom:4px;'>System</p>"
        "<p style='font-size:17px;font-weight:600;color:#e6edf3;margin:0 0 2px;'>"
        "Student Performance</p>"
        "<p style='font-size:13px;color:#58a6ff;margin:0 0 20px;'>Prediction v2.0</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "Navigation",
        ["Dashboard", "Live Predictor", "Exploratory Analysis",
         "Model Insights", "At-Risk Students"],
        label_visibility="collapsed",
    )
    st.divider()
    info = (
        f"<p style='font-size:12px;color:#8b949e;line-height:2.2;'>"
        f"Dataset&nbsp;&nbsp;<strong style='color:#c9d1d9'>{len(df):,} students</strong><br>"
        f"Features&nbsp;&nbsp;<strong style='color:#c9d1d9'>17 + 6 engineered</strong><br>"
        f"Best model&nbsp;&nbsp;<strong style='color:#c9d1d9'>Random Forest</strong><br>"
    )
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            _r = json.load(f)
        info += (
            f"Accuracy&nbsp;&nbsp;"
            f"<strong style='color:#3fb950'>{_r['best_accuracy']:.1%}</strong>"
        )
    info += "</p>"
    st.markdown(info, unsafe_allow_html=True)


def section(label):
    st.markdown(f"<h3>{label}</h3>", unsafe_allow_html=True)


# =============================================================================
# DASHBOARD
# =============================================================================
if page == "Dashboard":
    st.markdown("<h1>Analytics Overview</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e;font-size:14px;margin-top:-8px;margin-bottom:24px;'>"
        "Cohort summary across 1,000 simulated student records.</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Students", f"{len(df):,}")
    c2.metric("Average Score",  f"{df['final_score'].mean():.1f}")
    c3.metric("At Risk",        f"{df['at_risk'].sum()}")
    c4.metric("Top Performers", f"{(df['final_score'] >= 85).sum()}")
    c5.metric("Pass Rate",      f"{df['pass_fail'].mean():.1%}")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        section("Grade Distribution")
        grade_counts = df["grade"].value_counts().reindex(["A","B","C","D","F"])
        fig, ax = plt.subplots(figsize=(5, 3.2))
        bars = ax.bar(grade_counts.index, grade_counts.values,
                      color=[GRADE_COLORS[g] for g in grade_counts.index],
                      width=0.55, edgecolor="none")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    str(int(bar.get_height())), ha="center",
                    fontsize=10, color="#c9d1d9")
        ax.set_ylabel("Students")
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_r:
        section("Score Distribution")
        fig2, ax2 = plt.subplots(figsize=(5, 3.2))
        ax2.hist(df["final_score"], bins=20, color=ACCENT,
                 edgecolor="none", alpha=0.75)
        ax2.axvline(df["final_score"].mean(), color="#f85149",
                    linestyle="--", linewidth=1.4,
                    label=f"Mean {df['final_score'].mean():.1f}")
        ax2.set_xlabel("Final Score")
        ax2.set_ylabel("Count")
        ax2.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
        ax2.set_axisbelow(True)
        ax2.legend(fontsize=10)
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

    st.divider()
    section("Average Score by Attendance Band")
    att_bands = pd.cut(df["attendance_pct"], bins=[0,55,70,85,100],
                       labels=["40-55%","56-70%","71-85%","86-100%"])
    band_avg = df.groupby(att_bands, observed=True)["final_score"].mean()
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.bar(band_avg.index, band_avg.values,
            color=["#f85149","#d29922",ACCENT,"#3fb950"],
            width=0.45, edgecolor="none")
    for i, v in enumerate(band_avg.values):
        ax3.text(i, v + 0.4, f"{v:.1f}", ha="center",
                 fontsize=10, color="#c9d1d9")
    ax3.set_ylabel("Avg Score")
    ax3.set_ylim(0, 100)
    ax3.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
    ax3.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig3); plt.close()


# =============================================================================
# LIVE PREDICTOR
# =============================================================================
elif page == "Live Predictor":
    st.markdown("<h1>Live Grade Predictor</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e;font-size:14px;margin-top:-8px;margin-bottom:24px;'>"
        "Adjust parameters to predict a student's grade in real time.</p>",
        unsafe_allow_html=True,
    )

    if artefacts is None:
        st.warning("Models not found. Run `python main.py` first.")
    else:
        col_inputs, col_result = st.columns([1.1, 0.9])

        with col_inputs:
            section("Academic Parameters")
            study       = st.slider("Study hours / day",           0.0, 12.0, 5.0, 0.5)
            attendance  = st.slider("Attendance (%)",               30,  100,  75)
            prev_marks  = st.slider("Previous marks",               10,  100,  65)
            assignments = st.slider("Assignments completed (%)",     0,   100,  70)
            st.divider()
            section("Lifestyle Parameters")
            sleep  = st.slider("Sleep hours / night",           3.0, 12.0, 7.0, 0.5)
            extra  = st.slider("Extracurricular activities",    0, 5, 2)
            social = st.slider("Social media hours / day",      0.0, 10.0, 3.0, 0.5)
            st.divider()
            section("Background")
            gender      = st.selectbox("Gender", ["male", "female"])
            parental_edu= st.selectbox("Parental education",
                ["no_education","high_school","some_college","bachelors","masters"],
                index=3)
            internet = st.selectbox("Internet access", ["yes", "no"])
            tutoring = st.selectbox("Tutoring",        ["no",  "yes"])

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
        grade  = result["grade"]
        color  = GRADE_COLORS.get(grade, "#c9d1d9")

        with col_result:
            section("Prediction")
            st.markdown(
                f"<div style='background:#161b22;border:1px solid #21262d;"
                f"border-radius:10px;padding:28px 24px;margin-bottom:16px;'>"
                f"<p style='font-size:11px;color:#8b949e;text-transform:uppercase;"
                f"letter-spacing:0.08em;margin-bottom:6px;'>Predicted Grade</p>"
                f"<p style='font-size:72px;font-weight:700;color:{color};"
                f"line-height:1;margin:0 0 8px;'>{grade}</p>"
                f"<p style='font-size:13px;color:#8b949e;margin:0;'>Status: "
                f"<strong style='color:{'#f85149' if result['at_risk'] else '#3fb950'};'>"
                f"{'At Risk' if result['at_risk'] else 'On Track'}</strong></p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            section("Grade Probabilities")
            prob_df = (
                pd.DataFrame(result["probabilities"].items(),
                             columns=["Grade","Probability"])
                .sort_values("Grade")
            )
            fig_p, ax_p = plt.subplots(figsize=(4.5, 2.8))
            bars_p = ax_p.barh(prob_df["Grade"], prob_df["Probability"],
                               color=[GRADE_COLORS[g] for g in prob_df["Grade"]],
                               height=0.45, edgecolor="none")
            for bar in bars_p:
                w = bar.get_width()
                ax_p.text(w + 0.005, bar.get_y() + bar.get_height()/2,
                          f"{w:.0%}", va="center", fontsize=10, color="#c9d1d9")
            ax_p.set_xlim(0, 1.15)
            ax_p.xaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
            ax_p.set_axisbelow(True)
            ax_p.set_xlabel("Probability")
            plt.tight_layout()
            st.pyplot(fig_p); plt.close()

            section("Recommendations")
            for tip in result["recommendations"]:
                st.markdown(
                    f"<div style='background:#161b22;border-left:3px solid #58a6ff;"
                    f"border-radius:0 6px 6px 0;padding:10px 14px;"
                    f"margin-bottom:8px;font-size:13px;color:#c9d1d9;'>{tip}</div>",
                    unsafe_allow_html=True,
                )


# =============================================================================
# EXPLORATORY ANALYSIS
# =============================================================================
elif page == "Exploratory Analysis":
    st.markdown("<h1>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e;font-size:14px;margin-top:-8px;margin-bottom:24px;'>"
        "Feature relationships and distributions across the dataset.</p>",
        unsafe_allow_html=True,
    )

    section("Correlation Matrix")
    num_cols = [
        "study_hours_per_day","attendance_pct","previous_marks",
        "assignments_completed_pct","sleep_hours","social_media_hours","final_score",
    ]
    fig_h, ax_h = plt.subplots(figsize=(9, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, ax=ax_h,
                linewidths=0.5, linecolor=GRID_COLOR,
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 10})
    ax_h.tick_params(colors=TEXT_COLOR)
    plt.tight_layout()
    st.pyplot(fig_h); plt.close()

    st.divider()
    patches = [mpatches.Patch(color=GRADE_COLORS[g], label=g)
               for g in ["A","B","C","D","F"]]
    col_a, col_b = st.columns(2)

    with col_a:
        section("Score vs Study Hours")
        fig_s, ax_s = plt.subplots(figsize=(5, 3.5))
        for g in ["A","B","C","D","F"]:
            sub = df[df["grade"] == g]
            ax_s.scatter(sub["study_hours_per_day"], sub["final_score"],
                         color=GRADE_COLORS[g], alpha=0.5, s=12, edgecolors="none")
        m, b = np.polyfit(df["study_hours_per_day"], df["final_score"], 1)
        xs = np.linspace(0, 12, 100)
        ax_s.plot(xs, m*xs+b, color="#c9d1d9", linewidth=1.2, linestyle="--")
        ax_s.set_xlabel("Study hours / day")
        ax_s.set_ylabel("Final score")
        ax_s.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
        ax_s.set_axisbelow(True)
        ax_s.legend(handles=patches, fontsize=9, title="Grade",
                    title_fontsize=9, ncol=5, loc="upper left")
        plt.tight_layout()
        st.pyplot(fig_s); plt.close()

    with col_b:
        section("Score vs Attendance")
        fig_a, ax_a = plt.subplots(figsize=(5, 3.5))
        for g in ["A","B","C","D","F"]:
            sub = df[df["grade"] == g]
            ax_a.scatter(sub["attendance_pct"], sub["final_score"],
                         color=GRADE_COLORS[g], alpha=0.5, s=12, edgecolors="none")
        m2, b2 = np.polyfit(df["attendance_pct"], df["final_score"], 1)
        xs2 = np.linspace(30, 100, 100)
        ax_a.plot(xs2, m2*xs2+b2, color="#c9d1d9", linewidth=1.2, linestyle="--")
        ax_a.set_xlabel("Attendance (%)")
        ax_a.set_ylabel("Final score")
        ax_a.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
        ax_a.set_axisbelow(True)
        ax_a.legend(handles=patches, fontsize=9, title="Grade",
                    title_fontsize=9, ncol=5, loc="upper left")
        plt.tight_layout()
        st.pyplot(fig_a); plt.close()

    st.divider()
    section("Raw Data Sample — 50 rows")
    st.dataframe(df.sample(50).reset_index(drop=True), use_container_width=True)


# =============================================================================
# MODEL INSIGHTS
# =============================================================================
elif page == "Model Insights":
    st.markdown("<h1>Model Insights</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e;font-size:14px;margin-top:-8px;margin-bottom:24px;'>"
        "Accuracy, F1, and feature importance across all trained models.</p>",
        unsafe_allow_html=True,
    )

    if not RESULTS_PATH.exists():
        st.warning("Run `python main.py` first to generate model results.")
    else:
        with open(RESULTS_PATH) as f:
            res = json.load(f)

        section("Model Comparison Table")
        models_df = pd.DataFrame([
            {"Model": k, "Accuracy": v["accuracy"],
             "F1 (weighted)": v["f1"], "CV Score": v.get("cv_mean", 0)}
            for k, v in res["all_models"].items()
        ]).sort_values("Accuracy", ascending=False).reset_index(drop=True)
        st.dataframe(
            models_df.style
                .format({"Accuracy": "{:.1%}", "F1 (weighted)": "{:.4f}",
                         "CV Score": "{:.4f}"})
                .highlight_max(subset=["Accuracy","F1 (weighted)"],
                               color="#1a2e1a"),
            use_container_width=True,
        )

        st.divider()
        col_c, col_d = st.columns(2)

        with col_c:
            section("Accuracy vs F1 by Model")
            names = list(res["all_models"].keys())
            accs  = [res["all_models"][m]["accuracy"] for m in names]
            f1s   = [res["all_models"][m]["f1"]       for m in names]
            x, w  = np.arange(len(names)), 0.35
            fig_m, ax_m = plt.subplots(figsize=(5.5, 3.5))
            ax_m.bar(x-w/2, accs, w, label="Accuracy", color=ACCENT, edgecolor="none")
            ax_m.bar(x+w/2, f1s,  w, label="F1",       color="#3fb950", edgecolor="none")
            ax_m.set_xticks(x)
            ax_m.set_xticklabels([n.replace(" ","\n") for n in names], fontsize=9)
            ax_m.set_ylim(0.6, 1.0)
            ax_m.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
            ax_m.set_axisbelow(True)
            ax_m.legend(fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_m); plt.close()

        with col_d:
            if res.get("feature_importance"):
                section("Top Feature Importances")
                fi_s = pd.Series(res["feature_importance"]).sort_values(ascending=True).tail(12)
                fig_f, ax_f = plt.subplots(figsize=(5.5, 3.5))
                ax_f.barh(fi_s.index, fi_s.values,
                          color=ACCENT, edgecolor="none", height=0.55)
                ax_f.xaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
                ax_f.set_axisbelow(True)
                ax_f.set_xlabel("Importance (MDI)")
                plt.tight_layout()
                st.pyplot(fig_f); plt.close()


# =============================================================================
# AT-RISK STUDENTS
# =============================================================================
elif page == "At-Risk Students":
    st.markdown("<h1>At-Risk Student Report</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e;font-size:14px;margin-top:-8px;margin-bottom:24px;'>"
        "Students predicted to score below 50, flagged for early intervention.</p>",
        unsafe_allow_html=True,
    )

    at_risk_df = df[df["at_risk"] == 1].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("At-Risk Count",    len(at_risk_df))
    c2.metric("Avg Score",        f"{at_risk_df['final_score'].mean():.1f}")
    c3.metric("Avg Attendance",   f"{at_risk_df['attendance_pct'].mean():.1f}%")
    c4.metric("Avg Study Hours",  f"{at_risk_df['study_hours_per_day'].mean():.1f}h")

    st.divider()
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        section("Primary Risk Factors")
        risk_factors = {
            "Low attendance (<60%)":   (at_risk_df["attendance_pct"] < 60).sum(),
            "Low study time (<3h)":    (at_risk_df["study_hours_per_day"] < 3).sum(),
            "Low prev marks (<50)":    (at_risk_df["previous_marks"] < 50).sum(),
            "High social media (>5h)": (at_risk_df["social_media_hours"] > 5).sum(),
            "No tutoring":             (at_risk_df["tutoring"] == "no").sum(),
        }
        rf_s = pd.Series(risk_factors).sort_values()
        fig_r, ax_r = plt.subplots(figsize=(5, 3.2))
        ax_r.barh(rf_s.index, rf_s.values,
                  color="#f85149", edgecolor="none", height=0.5)
        for i, v in enumerate(rf_s.values):
            ax_r.text(v + 0.3, i, str(v), va="center",
                      fontsize=10, color="#c9d1d9")
        ax_r.xaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
        ax_r.set_axisbelow(True)
        ax_r.set_xlabel("Number of students")
        plt.tight_layout()
        st.pyplot(fig_r); plt.close()

    with col_r2:
        section("Score Distribution — At-Risk Cohort")
        fig_ar, ax_ar = plt.subplots(figsize=(5, 3.2))
        ax_ar.hist(at_risk_df["final_score"], bins=15,
                   color="#f85149", edgecolor="none", alpha=0.8)
        ax_ar.axvline(at_risk_df["final_score"].mean(),
                      color="#d29922", linestyle="--", linewidth=1.4,
                      label=f"Mean {at_risk_df['final_score'].mean():.1f}")
        ax_ar.set_xlabel("Final Score")
        ax_ar.set_ylabel("Count")
        ax_ar.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
        ax_ar.set_axisbelow(True)
        ax_ar.legend(fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_ar); plt.close()

    st.divider()
    section("At-Risk Student Records")
    display_cols = [
        "student_id","gender","attendance_pct","study_hours_per_day",
        "previous_marks","assignments_completed_pct","final_score","grade",
    ]
    st.dataframe(
        at_risk_df[display_cols].sort_values("final_score").reset_index(drop=True),
        use_container_width=True,
    )
