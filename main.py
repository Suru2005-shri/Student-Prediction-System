"""
main.py
-------
Orchestrates the full pipeline:
  generate data → preprocess → train → evaluate → visualize → predict

Run:  python main.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def banner(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def main():
    # ── 1. Generate dataset ───────────────────────────────────────────────────
    banner("Step 1 / 5  —  Generating synthetic dataset")
    from src.generate_data import simulate_students
    df = simulate_students(1000)
    path = Path("data/student_performance.csv")
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  {len(df)} students generated  →  {path}")
    print(f"  Grade counts:\n{df['grade'].value_counts().sort_index().to_string()}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    banner("Step 2 / 5  —  Preprocessing & feature engineering")
    from src.preprocess import get_processed
    X_train, X_test, y_train, y_test, scaler, feature_cols, le = get_processed("grade")
    print(f"  Train shape  : {X_train.shape}")
    print(f"  Test shape   : {X_test.shape}")
    print(f"  Features ({len(feature_cols)}) : {', '.join(feature_cols[:6])} ...")

    # ── 3. Train models ───────────────────────────────────────────────────────
    banner("Step 3 / 5  —  Training ML models")
    from src.train_models import train_and_evaluate, save_best_model
    results = train_and_evaluate(X_train, X_test, y_train, y_test, le, feature_cols)
    best_name, feat_imp = save_best_model(results, feature_cols)

    # ── 4. Visualize ──────────────────────────────────────────────────────────
    banner("Step 4 / 5  —  Generating visualizations")
    from src.visualize import main as vis_main
    vis_main()

    # ── 5. Sample prediction ──────────────────────────────────────────────────
    banner("Step 5 / 5  —  Running sample prediction")
    from src.predict import predict, SAMPLE_STUDENT

    result = predict(SAMPLE_STUDENT)
    print(f"  Sample student profile:")
    for k, v in SAMPLE_STUDENT.items():
        print(f"    {k:<30}: {v}")
    print(f"\n  ► Predicted Grade  : {result['grade']}")
    print(f"  ► At Risk          : {result['at_risk']}")
    print(f"  ► Grade probs      :")
    for g, p in sorted(result["probabilities"].items()):
        bar = "█" * int(p * 25)
        print(f"       {g}  {bar:<25}  {p:.1%}")
    print(f"\n  ► Recommendations  :")
    for i, tip in enumerate(result["recommendations"], 1):
        print(f"       {i}. {tip}")

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("Pipeline complete!")
    with open("outputs/model_results.json") as f:
        summary = json.load(f)
    print(f"  Best model         : {summary['best_model']}")
    print(f"  Test accuracy      : {summary['best_accuracy']:.4f}")
    print(f"  Weighted F1        : {summary['best_f1']:.4f}")
    print(f"\n  Outputs generated  :")
    print(f"    data/student_performance.csv")
    print(f"    models/best_model.pkl  +  scaler.pkl  +  label_encoder.pkl")
    print(f"    outputs/model_results.json")
    print(f"    images/ (5 charts)")
    print()


if __name__ == "__main__":
    main()
