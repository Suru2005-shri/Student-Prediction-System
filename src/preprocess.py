"""
preprocess.py
-------------
Cleaning, encoding, feature engineering, and train/test split.
Returns processed arrays ready for model training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

SEED = 42
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── categorical encoders (saved for inference) ────────────────────────────────
ORDINAL_MAP_EDU = {
    "no_education": 0, "high_school": 1,
    "some_college": 2, "bachelors": 3, "masters": 4
}


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = DATA_DIR / "student_performance.csv"
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: duplicates, range checks, and type enforcement."""
    df = df.drop_duplicates(subset="student_id")
    df = df.drop(columns=["student_id"])  # not a feature

    # enforce numeric ranges
    df["attendance_pct"] = df["attendance_pct"].clip(0, 100)
    df["previous_marks"] = df["previous_marks"].clip(0, 100)
    df["assignments_completed_pct"] = df["assignments_completed_pct"].clip(0, 100)
    df["study_hours_per_day"] = df["study_hours_per_day"].clip(0, 16)
    df["sleep_hours"] = df["sleep_hours"].clip(0, 12)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction and derived features."""
    # engagement score (0-100 composite)
    df["engagement_score"] = (
        df["attendance_pct"] * 0.4
        + df["assignments_completed_pct"] * 0.35
        + df["study_hours_per_day"] / 12 * 100 * 0.25
    ).round(2)

    # study efficiency: study hours weighted by whether student has internet + tutoring
    df["learning_support"] = (
        (df["internet_access"] == "yes").astype(int) * 0.5
        + (df["tutoring"] == "yes").astype(int) * 0.5
    )

    # sleep quality: 1 if in healthy range 6-9
    df["healthy_sleep"] = df["sleep_hours"].between(6, 9).astype(int)

    # attendance band
    def att_band(a):
        if a >= 90: return 3
        if a >= 75: return 2
        if a >= 60: return 1
        return 0
    df["attendance_band"] = df["attendance_pct"].apply(att_band)

    # academic consistency (how similar prev_marks and assign_pct are — proxy for consistency)
    df["consistency"] = 100 - np.abs(df["previous_marks"] - df["assignments_completed_pct"])

    # study × attendance interaction
    df["study_x_attend"] = (df["study_hours_per_day"] * df["attendance_pct"] / 100).round(3)

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals for ML models."""
    # ordinal encode parental education
    df["parental_education"] = df["parental_education"].map(ORDINAL_MAP_EDU)

    # binary encode gender, internet, tutoring
    df["gender"] = (df["gender"] == "female").astype(int)
    df["internet_access"] = (df["internet_access"] == "yes").astype(int)
    df["tutoring"] = (df["tutoring"] == "yes").astype(int)
    return df


def build_datasets(df: pd.DataFrame, target: str = "grade",
                   test_size: float = 0.2, random_state: int = SEED):
    """
    Returns X_train, X_test, y_train, y_test, scaler, feature_names.
    target can be 'grade' (classification), 'at_risk' (binary),
    or 'final_score' (regression).
    """
    drop_cols = {"final_score", "grade", "at_risk", "pass_fail"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df[target].copy()

    # encode grade labels
    le = None
    if target == "grade":
        le = LabelEncoder()
        y = le.fit_transform(y)
        joblib.dump(le, MODELS_DIR / "label_encoder.pkl")

    # impute (should be none with synthetic data, but good practice)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(imputer, MODELS_DIR / "imputer.pkl")

    return X_train, X_test, y_train, y_test, scaler, feature_cols, le


def get_processed(target: str = "grade"):
    """One-shot: load → clean → engineer → encode → split."""
    df = load_data()
    df = clean(df)
    df = engineer_features(df)
    df = encode(df)
    return build_datasets(df, target=target)


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, scaler, feats, le = get_processed("grade")
    print(f"Train shape : {X_tr.shape}")
    print(f"Test shape  : {X_te.shape}")
    print(f"Features    : {feats}")
