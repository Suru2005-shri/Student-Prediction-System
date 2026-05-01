"""
generate_data.py
----------------
Synthesizes a realistic student performance dataset using domain-informed
feature relationships. Mimics patterns from UCI Student Performance dataset
(Cortez & Silva, 2008) while being entirely synthetic and free to use.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
np.random.seed(SEED)
N = 1000

PARENTAL_EDU = ["no_education", "high_school", "some_college", "bachelors", "masters"]
GENDER = ["male", "female"]
INTERNET = ["yes", "no"]
TUTORING = ["yes", "no"]


def simulate_students(n: int = N) -> pd.DataFrame:
    """Generate synthetic student records with realistic feature correlations."""

    # ── Demographic features ─────────────────────────────────────────────────
    gender = np.random.choice(GENDER, n)
    parental_education = np.random.choice(PARENTAL_EDU, n,
                                          p=[0.05, 0.20, 0.25, 0.35, 0.15])
    internet_access = np.random.choice(INTERNET, n, p=[0.80, 0.20])
    tutoring = np.random.choice(TUTORING, n, p=[0.35, 0.65])

    # Parental education → encoded numeric for score influence
    edu_score = np.vectorize({"no_education": 0, "high_school": 1,
                               "some_college": 2, "bachelors": 3,
                               "masters": 4}.get)(parental_education)

    # ── Academic behaviour ────────────────────────────────────────────────────
    study_hours = np.clip(np.random.normal(5, 2, n), 0, 12).round(1)
    attendance = np.clip(np.random.normal(75, 15, n), 30, 100).round(0).astype(int)
    previous_marks = np.clip(np.random.normal(65, 18, n), 10, 100).round(0).astype(int)
    assignments_completed = np.clip(
        np.random.normal(70, 20, n) + edu_score * 3, 0, 100
    ).round(0).astype(int)
    sleep_hours = np.clip(np.random.normal(7, 1.2, n), 3, 12).round(1)
    extracurricular = np.random.randint(0, 6, n)
    tutoring_hours = np.where(tutoring == "yes",
                              np.clip(np.random.normal(2, 1, n), 0, 6), 0).round(1)
    social_media_hours = np.clip(np.random.normal(3, 1.5, n), 0, 10).round(1)

    # ── Final score — weighted formula + noise ───────────────────────────────
    noise = np.random.normal(0, 5, n)
    internet_bonus = np.where(internet_access == "yes", 3, 0)
    sleep_penalty = np.abs(sleep_hours - 7.5) * 1.5   # penalty for over/under sleep

    raw_score = (
        0.30 * previous_marks
        + 0.22 * (attendance - 30) / 70 * 100
        + 0.20 * study_hours * 8
        + 0.10 * assignments_completed
        + 0.05 * edu_score * 18
        + 0.04 * tutoring_hours * 10
        + 0.03 * extracurricular * 8
        + internet_bonus
        - 0.02 * social_media_hours * 5
        - sleep_penalty
        + noise
    )

    final_score = np.clip(raw_score, 0, 100).round(0).astype(int)

    # ── Derived targets ───────────────────────────────────────────────────────
    def to_grade(s):
        if s >= 90: return "A"
        if s >= 75: return "B"
        if s >= 60: return "C"
        if s >= 50: return "D"
        return "F"

    grade = np.vectorize(to_grade)(final_score)
    at_risk = (final_score < 50).astype(int)
    pass_fail = (final_score >= 50).astype(int)

    df = pd.DataFrame({
        "student_id": [f"S{str(i+1).zfill(4)}" for i in range(n)],
        "gender": gender,
        "parental_education": parental_education,
        "internet_access": internet_access,
        "tutoring": tutoring,
        "study_hours_per_day": study_hours,
        "attendance_pct": attendance,
        "previous_marks": previous_marks,
        "assignments_completed_pct": assignments_completed,
        "sleep_hours": sleep_hours,
        "extracurricular_activities": extracurricular,
        "tutoring_hours_per_week": tutoring_hours,
        "social_media_hours": social_media_hours,
        "final_score": final_score,
        "grade": grade,
        "at_risk": at_risk,
        "pass_fail": pass_fail,
    })
    return df


if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    df = simulate_students(N)
    path = out_dir / "student_performance.csv"
    df.to_csv(path, index=False)

    print(f"Dataset generated  →  {path}")
    print(f"Shape              :  {df.shape}")
    print(f"Grade distribution :\n{df['grade'].value_counts().sort_index()}")
    print(f"At-risk students   :  {df['at_risk'].sum()} / {N}")
