import os
import sys

# ============================================================
# PIPELINE RUNNER
# ============================================================

STEPS = [
    "src/data_preparation.py",
    "src/feature_engineering.py",
    "src/model_training.py",
    "src/ab_test.py",
    "src/clustering.py",
    "src/propensity_model.py"
]

def run_step(step_path):
    print(f"\n=== RUNNING: {step_path} ===")
    result = os.system(f"python {step_path}")

    if result != 0:
        print(f"\n❌ ERROR: Step failed — {step_path}")
        sys.exit(1)

    print(f"✓ COMPLETED: {step_path}")


def run_pipeline():
    print("\n======================================")
    print("      FULL DATA & ML PIPELINE START   ")
    print("======================================\n")

    for step in STEPS:
        run_step(step)

    print("\n======================================")
    print("      PIPELINE FINISHED SUCCESSFULLY  ")
    print("======================================\n")


if __name__ == "__main__":
    run_pipeline()
