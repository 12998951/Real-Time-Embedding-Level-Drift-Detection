
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
WINDOW_SIZE = 20
DIMENSIONS = 16
NOISE_STD = 0.10

REFERENCE_TRIALS = 30
CALIBRATION_WINDOWS = 500
TEST_WINDOWS = 300
CALIBRATION_PERCENTILE = 99

SHIFT_VALUES = [0.00, 0.02, 0.04, 0.05, 0.07, 0.10, 0.15]

# Resolve output paths relative to the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
ASSETS_DIR = PROJECT_ROOT / "assets"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def run_evaluation():
    all_results = []
    thresholds = []

    print("Starting reproducible synthetic evaluation...")

    for ref_id in range(REFERENCE_TRIALS):
        rng = np.random.default_rng(10000 + ref_id)

        # Create an independent reference window.
        reference = rng.normal(
            0.0, NOISE_STD,
            size=(WINDOW_SIZE, DIMENSIONS)
        )
        reference_mean = np.mean(reference, axis=0)

        # Calibrate using independent stable windows.
        calibration_scores = []

        for _ in range(CALIBRATION_WINDOWS):
            window = rng.normal(
                0.0, NOISE_STD,
                size=(WINDOW_SIZE, DIMENSIONS)
            )

            score = np.linalg.norm(
                reference_mean - np.mean(window, axis=0)
            )
            calibration_scores.append(score)

        threshold = float(np.percentile(
            calibration_scores,
            CALIBRATION_PERCENTILE
        ))
        thresholds.append(threshold)

        # Test stable and shifted windows using the same reference.
        for shift in SHIFT_VALUES:
            scores = []

            for _ in range(TEST_WINDOWS):
                window = rng.normal(
                    shift, NOISE_STD,
                    size=(WINDOW_SIZE, DIMENSIONS)
                )

                score = np.linalg.norm(
                    reference_mean - np.mean(window, axis=0)
                )
                scores.append(score)

            scores = np.asarray(scores)
            detected = scores > threshold

            all_results.append({
                "reference_id": ref_id + 1,
                "shift": shift,
                "threshold": threshold,
                "test_windows": TEST_WINDOWS,
                "alarms": int(np.sum(detected)),
                "detection_rate_pct": 100 * np.mean(detected)
            })

    # ==========================================
    # SUMMARIZE RESULTS
    # ==========================================
    results = pd.DataFrame(all_results)

    summary = results.groupby("shift").agg(
        mean_detection_rate_pct=("detection_rate_pct", "mean"),
        min_detection_rate_pct=("detection_rate_pct", "min"),
        max_detection_rate_pct=("detection_rate_pct", "max"),
        total_alarms=("alarms", "sum"),
        total_windows=("test_windows", "sum")
    ).reset_index()

    summary["pooled_alarm_rate_pct"] = (
        100 * summary["total_alarms"] / summary["total_windows"]
    )

    print("\n=== EVALUATION SUMMARY ===")
    print("Independent reference windows:", REFERENCE_TRIALS)
    print("Mean calibrated threshold:", round(np.mean(thresholds), 4))
    print("\n", summary.round(2).to_string(index=False))

    # ==========================================
    # SAVE RESULTS
    # ==========================================
    reference_path = RESULTS_DIR / "drift_results_by_reference.csv"
    summary_path = RESULTS_DIR / "drift_results_summary.csv"
    chart_path = ASSETS_DIR / "drift_detection_curve.png"

    results.to_csv(reference_path, index=False)
    summary.to_csv(summary_path, index=False)

    # ==========================================
    # PLOT DETECTION RATE
    # ==========================================
    plt.figure(figsize=(8, 5))

    plt.plot(
        summary["shift"],
        summary["mean_detection_rate_pct"],
        marker="o"
    )

    plt.xlabel("Synthetic mean shift per dimension")
    plt.ylabel("Mean detection rate (%)")
    plt.title("Embedding Drift Detector: Synthetic Evaluation")
    plt.ylim(0, 105)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(chart_path, dpi=200)
    plt.close()

    print("\nFiles saved:")
    print(reference_path)
    print(summary_path)
    print(chart_path)
    print("\nEvaluation completed.")


if __name__ == "__main__":
    run_evaluation()
