"""
visualize.py
------------
Generates publication-quality charts for EDA, model evaluation, and insights.
All plots are saved to /outputs/ and /images/.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

OUTPUTS = Path(__file__).parent.parent / "outputs"
IMAGES  = Path(__file__).parent.parent / "images"
DATA    = Path(__file__).parent.parent / "data"
OUTPUTS.mkdir(exist_ok=True)
IMAGES.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "A": "#3B6D11", "B": "#185FA5", "C": "#854F0B", "D": "#A32D2D", "F": "#501313"
}
GRADE_ORDER = ["A", "B", "C", "D", "F"]
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def load_data():
    df = pd.read_csv(DATA / "student_performance.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. Score distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_score_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # histogram
    ax = axes[0]
    ax.hist(df["final_score"], bins=20, color="#378ADD", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(df["final_score"].mean(), color="#E24B4A", linestyle="--", linewidth=1.5,
               label=f"Mean = {df['final_score'].mean():.1f}")
    ax.axvline(df["final_score"].median(), color="#639922", linestyle=":", linewidth=1.5,
               label=f"Median = {df['final_score'].median():.1f}")
    ax.set_title("Final Score Distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend(fontsize=10)

    # grade donut
    ax2 = axes[1]
    counts = df["grade"].value_counts().reindex(GRADE_ORDER).fillna(0)
    nonzero = counts[counts > 0]
    colors = [PALETTE[g] for g in nonzero.index]
    wedges, texts, autotexts = ax2.pie(
        nonzero, labels=nonzero.index, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for at in autotexts: at.set_fontsize(9)
    ax2.set_title("Grade Distribution")

    plt.tight_layout()
    fig.savefig(IMAGES / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: images/score_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df):
    num_cols = [
        "study_hours_per_day", "attendance_pct", "previous_marks",
        "assignments_completed_pct", "sleep_hours", "extracurricular_activities",
        "social_media_hours", "final_score"
    ]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1, ax=ax,
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=15, pad=12)
    plt.tight_layout()
    fig.savefig(IMAGES / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: images/correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Study hours vs score (scatter + regression line)
# ─────────────────────────────────────────────────────────────────────────────
def plot_study_vs_score(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # scatter
    ax = axes[0]
    colors = [PALETTE[g] for g in df["grade"]]
    ax.scatter(df["study_hours_per_day"], df["final_score"],
               c=colors, alpha=0.5, s=15, edgecolors="none")
    # regression
    m, b = np.polyfit(df["study_hours_per_day"], df["final_score"], 1)
    xs = np.linspace(0, 12, 100)
    ax.plot(xs, m*xs + b, color="black", linewidth=1.5, linestyle="--", label=f"y = {m:.1f}x + {b:.1f}")
    ax.set_title("Study Hours vs Final Score")
    ax.set_xlabel("Study hours per day")
    ax.set_ylabel("Final score")
    patches = [mpatches.Patch(color=PALETTE[g], label=g) for g in GRADE_ORDER]
    ax.legend(handles=patches, fontsize=9, title="Grade")

    # attendance vs score
    ax2 = axes[1]
    ax2.scatter(df["attendance_pct"], df["final_score"],
                c=colors, alpha=0.5, s=15, edgecolors="none")
    m2, b2 = np.polyfit(df["attendance_pct"], df["final_score"], 1)
    xs2 = np.linspace(30, 100, 100)
    ax2.plot(xs2, m2*xs2 + b2, color="black", linewidth=1.5, linestyle="--")
    ax2.set_title("Attendance % vs Final Score")
    ax2.set_xlabel("Attendance (%)")
    ax2.set_ylabel("Final score")
    ax2.legend(handles=patches, fontsize=9, title="Grade")

    plt.tight_layout()
    fig.savefig(IMAGES / "study_attendance_vs_score.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: images/study_attendance_vs_score.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature importance bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance():
    results_path = OUTPUTS / "model_results.json"
    if not results_path.exists():
        print("Run train_models.py first to generate model_results.json")
        return

    with open(results_path) as f:
        data = json.load(f)

    fi = data.get("feature_importance", {})
    if not fi:
        print("No feature importance available (SVM/Logistic Reg. don't expose it)")
        return

    fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
    fi_df = fi_df.sort_values("Importance", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(fi_df["Feature"], fi_df["Importance"],
                   color="#378ADD", edgecolor="white", height=0.6)
    ax.set_xlabel("Feature Importance (MDI)")
    ax.set_title(f"Top Feature Importances — {data['best_model']}")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height()/2,
                f"{w:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(IMAGES / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: images/feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_comparison():
    results_path = OUTPUTS / "model_results.json"
    if not results_path.exists():
        print("Run train_models.py first.")
        return

    with open(results_path) as f:
        data = json.load(f)

    models = list(data["all_models"].keys())
    accs = [data["all_models"][m]["accuracy"] for m in models]
    f1s = [data["all_models"][m]["f1"] for m in models]

    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#378ADD", edgecolor="white")
    b2 = ax.bar(x + w/2, f1s, w, label="Weighted F1", color="#639922", edgecolor="white")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0.7, 1.0)
    ax.set_title("ML Model Comparison — Accuracy & F1 Score")
    ax.legend()
    plt.tight_layout()
    fig.savefig(IMAGES / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: images/model_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Score by parental education
# ─────────────────────────────────────────────────────────────────────────────
def plot_parental_education_effect(df):
    edu_order = ["no_education", "high_school", "some_college", "bachelors", "masters"]
    means = df.groupby("parental_education")["final_score"].mean().reindex(edu_order)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(means.index, means.values, color="#7F77DD", edgecolor="white", width=0.55)
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", fontsize=10)
    ax.set_xticklabels(edu_order, rotation=20, ha="right")
    ax.set_ylabel("Average Final Score")
    ax.set_title("Impact of Parental Education on Student Score")
    plt.tight_layout()
    fig.savefig(IMAGES / "parental_education_effect.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: images/parental_education_effect.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Generating visualizations...")
    df = load_data()
    plot_score_distribution(df)
    plot_correlation_heatmap(df)
    plot_study_vs_score(df)
    plot_parental_education_effect(df)
    plot_feature_importance()
    plot_model_comparison()
    print("\nAll charts saved to /images/")


if __name__ == "__main__":
    main()
