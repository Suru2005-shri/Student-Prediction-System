"""
train_models.py
---------------
Trains and evaluates four ML classifiers on the grade-prediction task.
Saves the best model + serialized artefacts to /models/.
"""

import json
import warnings
import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings("ignore")
SEED = 42
MODELS_DIR = Path(__file__).parent.parent / "models"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Import preprocessing ──────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import get_processed

# ── Model definitions ─────────────────────────────────────────────────────────
MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        random_state=SEED,
    ),
    "SVM": SVC(
        kernel="rbf",
        C=5,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=SEED,
    ),
    "Logistic Regression": LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
    ),
}


def train_and_evaluate(X_train, X_test, y_train, y_test, le, feature_cols):
    """Train all models, print metrics, return results dict."""
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for name, model in MODELS.items():
        print(f"\n{'─'*55}")
        print(f"  Training  →  {name}")
        print(f"{'─'*55}")

        # cross-val on training set
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=skf, scoring="accuracy", n_jobs=-1)
        print(f"  CV accuracy  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # fit on full training set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)

        print(f"  Test accuracy: {acc:.4f}")
        print(f"  Weighted F1  : {f1:.4f}")

        labels = le.classes_ if le else sorted(set(y_test))
        print("\n" + classification_report(y_test, y_pred, target_names=labels))

        results[name] = {
            "model": model,
            "accuracy": acc,
            "f1": f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "confusion_matrix": cm.tolist(),
        }

    return results


def save_best_model(results, feature_cols):
    """Persist the best model + feature importance if available."""
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best = results[best_name]
    best_model = best["model"]

    print(f"\n{'═'*55}")
    print(f"  Best model  →  {best_name}  (acc={best['accuracy']:.4f})")
    print(f"{'═'*55}\n")

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    # feature importance
    feat_imp = {}
    if hasattr(best_model, "feature_importances_"):
        feat_imp = dict(zip(feature_cols, best_model.feature_importances_.tolist()))
        feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    # save summary
    summary = {
        "best_model": best_name,
        "best_accuracy": best["accuracy"],
        "best_f1": best["f1"],
        "feature_importance": feat_imp,
        "all_models": {
            k: {"accuracy": v["accuracy"], "f1": v["f1"],
                "cv_mean": v["cv_mean"], "confusion_matrix": v["confusion_matrix"]}
            for k, v in results.items()
        }
    }
    with open(OUTPUTS_DIR / "model_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Artefacts saved:")
    print(f"  models/best_model.pkl")
    print(f"  outputs/model_results.json")
    return best_name, feat_imp


def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_cols, le = get_processed("grade")
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    results = train_and_evaluate(X_train, X_test, y_train, y_test, le, feature_cols)
    save_best_model(results, feature_cols)
    return results


if __name__ == "__main__":
    main()
