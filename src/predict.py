"""
predict.py
----------
Loads serialized artefacts and runs inference on new student records.
Can be called from the CLI, imported as a module, or wired into an API.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"


def load_artefacts():
    """Load saved model, scaler, imputer, and label encoder."""
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    imputer = joblib.load(MODELS_DIR / "imputer.pkl")
    le = joblib.load(MODELS_DIR / "label_encoder.pkl")
    return model, scaler, imputer, le


# Match exact feature order used during training (after engineer_features + encode)
FEATURE_ORDER = [
    "gender", "parental_education", "internet_access", "tutoring",
    "study_hours_per_day", "attendance_pct", "previous_marks",
    "assignments_completed_pct", "sleep_hours", "extracurricular_activities",
    "tutoring_hours_per_week", "social_media_hours",
    "engagement_score", "learning_support", "healthy_sleep",
    "attendance_band", "consistency", "study_x_attend",
]

ORDINAL_MAP_EDU = {
    "no_education": 0, "high_school": 1,
    "some_college": 2, "bachelors": 3, "masters": 4
}


def preprocess_input(student: dict) -> np.ndarray:
    """
    Accepts a raw student dict (same schema as the original CSV, minus student_id)
    and returns a scaled feature vector ready for .predict().
    """
    row = student.copy()

    # ── encode categoricals ───────────────────────────────────────────────────
    row["parental_education"] = ORDINAL_MAP_EDU.get(
        row.get("parental_education", "high_school"), 1
    )
    row["gender"] = 1 if row.get("gender", "male") == "female" else 0
    row["internet_access"] = 1 if row.get("internet_access", "yes") == "yes" else 0
    row["tutoring"] = 1 if row.get("tutoring", "no") == "yes" else 0

    # ── derive engineered features ────────────────────────────────────────────
    attendance = row.get("attendance_pct", 75)
    assignments = row.get("assignments_completed_pct", 70)
    study = row.get("study_hours_per_day", 5)
    sleep = row.get("sleep_hours", 7)
    internet = row["internet_access"]
    tutoring_flag = row["tutoring"]

    row["engagement_score"] = round(
        attendance * 0.4 + assignments * 0.35 + study / 12 * 100 * 0.25, 2
    )
    row["learning_support"] = internet * 0.5 + tutoring_flag * 0.5
    row["healthy_sleep"] = 1 if 6 <= sleep <= 9 else 0
    row["attendance_band"] = (
        3 if attendance >= 90 else 2 if attendance >= 75 else 1 if attendance >= 60 else 0
    )
    row["consistency"] = 100 - abs(
        row.get("previous_marks", 65) - assignments
    )
    row["study_x_attend"] = round(study * attendance / 100, 3)

    # build ordered feature vector
    X = np.array([[row.get(f, 0) for f in FEATURE_ORDER]], dtype=float)
    return X


def predict(student: dict) -> dict:
    """
    Main inference function.

    Parameters
    ----------
    student : dict  — raw student record (see SAMPLE below)

    Returns
    -------
    dict with keys:
        grade          : predicted letter grade (A/B/C/D/F)
        probabilities  : {grade: probability} dict
        at_risk        : bool
        recommendation : list of action items
    """
    model, scaler, imputer, le = load_artefacts()

    X_raw = preprocess_input(student)
    X_imp = imputer.transform(X_raw)
    X_scaled = scaler.transform(X_imp)

    grade_encoded = model.predict(X_scaled)[0]
    grade = le.inverse_transform([grade_encoded])[0]

    proba = model.predict_proba(X_scaled)[0]
    proba_dict = {cls: round(float(p), 4) for cls, p in zip(le.classes_, proba)}

    at_risk = grade in ("D", "F")
    recommendations = _generate_recommendations(student, grade)

    return {
        "grade": grade,
        "probabilities": proba_dict,
        "at_risk": at_risk,
        "recommendations": recommendations,
    }


def _generate_recommendations(student: dict, grade: str) -> list[str]:
    """Rule-based recommendation engine layered on top of ML output."""
    tips = []
    attendance = student.get("attendance_pct", 75)
    study = student.get("study_hours_per_day", 5)
    assign = student.get("assignments_completed_pct", 70)
    sleep = student.get("sleep_hours", 7)
    social = student.get("social_media_hours", 3)
    prev = student.get("previous_marks", 65)
    tutoring = student.get("tutoring", "no")

    if attendance < 75:
        tips.append(f"Attendance is {attendance}%. Aim for above 85% to significantly boost performance.")
    if study < 3:
        tips.append(f"Study time ({study}h/day) is low. Even 1 extra hour daily has measurable impact.")
    if assign < 70:
        tips.append(f"Only {assign}% of assignments completed. Consistent submission builds long-term retention.")
    if not (6 <= sleep <= 9):
        tips.append(f"Sleep is {sleep}h/night. Optimal learning requires 6-9 hours.")
    if social > 4:
        tips.append(f"Social media usage ({social}h/day) may be impacting focus. Try time-blocking study sessions.")
    if tutoring == "no" and grade in ("C", "D", "F"):
        tips.append("Consider enrolling in tutoring or peer study groups — shown to improve grades by 1-2 letter grades.")
    if prev < 55:
        tips.append("Review fundamentals from previous modules — foundation gaps compound over time.")

    if not tips:
        tips.append("Great habits! Maintain consistency and challenge yourself with advanced problems.")
    return tips


# ── Sample usage ───────────────────────────────────────────────────────────────
SAMPLE_STUDENT = {
    "gender": "male",
    "parental_education": "bachelors",
    "internet_access": "yes",
    "tutoring": "no",
    "study_hours_per_day": 4.0,
    "attendance_pct": 68,
    "previous_marks": 58,
    "assignments_completed_pct": 65,
    "sleep_hours": 6.5,
    "extracurricular_activities": 2,
    "tutoring_hours_per_week": 0,
    "social_media_hours": 4.5,
}

if __name__ == "__main__":
    result = predict(SAMPLE_STUDENT)
    print("\n── Prediction Result ─────────────────────────────────")
    print(f"  Predicted grade  : {result['grade']}")
    print(f"  At risk          : {result['at_risk']}")
    print(f"  Grade probs      :")
    for g, p in sorted(result["probabilities"].items()):
        bar = "█" * int(p * 30)
        print(f"    {g}  {bar:<30}  {p:.2%}")
    print("\n  Recommendations:")
    for i, tip in enumerate(result["recommendations"], 1):
        print(f"    {i}. {tip}")
