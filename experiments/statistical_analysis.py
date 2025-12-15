# ============================================================
# Statistical Analysis: Multi-Seed Evaluation
# Author: Ghanta Krishna Murthy
# ============================================================

import os
import json
import numpy as np
from scipy import stats
import math

# -------------------------------
# Load raw results
# -------------------------------
with open("results/multi_seed_raw_results.json", "r") as f:
    data = json.load(f)

baseline = np.array(data["baseline"])
qinn = np.array(data["qinn"])
n = len(baseline)

# -------------------------------
# Helper functions
# -------------------------------
def summary_stats(x):
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    return mean, std, min_v, max_v


def confidence_interval(x, alpha=0.05):
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    t_val = stats.t.ppf(1 - alpha / 2, df=len(x) - 1)
    margin = t_val * std / math.sqrt(len(x))
    return float(mean - margin), float(mean + margin)


# -------------------------------
# Compute statistics
# -------------------------------
b_mean, b_std, b_min, b_max = summary_stats(baseline)
q_mean, q_std, q_min, q_max = summary_stats(qinn)

b_ci = confidence_interval(baseline)
q_ci = confidence_interval(qinn)

# Paired t-test
t_stat, p_value = stats.ttest_rel(qinn, baseline)

# -------------------------------
# Save machine-readable summary
# -------------------------------
summary = {
    "baseline": {
        "mean": b_mean,
        "std": b_std,
        "min": b_min,
        "max": b_max,
        "ci_95": b_ci,
        "n_seeds": n,
    },
    "qinn": {
        "mean": q_mean,
        "std": q_std,
        "min": q_min,
        "max": q_max,
        "ci_95": q_ci,
        "n_seeds": n,
    },
    "paired_t_test": {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
    },
}

os.makedirs("results", exist_ok=True)
with open("results/statistical_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# -------------------------------
# Save human-readable summary
# -------------------------------
with open("results/statistical_summary.txt", "w") as f:
    f.write("Multi-Seed Statistical Evaluation (MNIST)\n")
    f.write("=" * 45 + "\n\n")

    f.write("Baseline Model:\n")
    f.write(f"  Mean Accuracy: {b_mean:.4f}\n")
    f.write(f"  Std Dev      : {b_std:.4f}\n")
    f.write(f"  Min / Max    : {b_min:.4f} / {b_max:.4f}\n")
    f.write(f"  95% CI       : [{b_ci[0]:.4f}, {b_ci[1]:.4f}]\n\n")

    f.write("QINN Model:\n")
    f.write(f"  Mean Accuracy: {q_mean:.4f}\n")
    f.write(f"  Std Dev      : {q_std:.4f}\n")
    f.write(f"  Min / Max    : {q_min:.4f} / {q_max:.4f}\n")
    f.write(f"  95% CI       : [{q_ci[0]:.4f}, {q_ci[1]:.4f}]\n\n")

    f.write("Paired t-test (QINN vs Baseline):\n")
    f.write(f"  t-statistic : {t_stat:.4f}\n")
    f.write(f"  p-value     : {p_value:.6f}\n")

print("✅ Statistical summary generated.")
