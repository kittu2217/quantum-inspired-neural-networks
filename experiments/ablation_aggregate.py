# ============================================================
# Ablation Study: Aggregation & Visualization
# Author: Ghanta Krishna Murthy
# ============================================================

import json
import numpy as np
import os
import matplotlib.pyplot as plt

# -------------------------------
# Load raw results
# -------------------------------
with open("results/ablation_raw_results.json", "r") as f:
    raw = json.load(f)

# -------------------------------
# Aggregate statistics
# -------------------------------
summary = {}

for variant, accs in raw.items():
    accs = np.array(accs)
    summary[variant] = {
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs, ddof=1)),
        "min": float(np.min(accs)),
        "max": float(np.max(accs)),
        "n_seeds": len(accs),
    }

# -------------------------------
# Save JSON summary
# -------------------------------
os.makedirs("results", exist_ok=True)
with open("results/ablation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# -------------------------------
# Save human-readable table
# -------------------------------
with open("results/ablation_summary.txt", "w") as f:
    f.write("Ablation Study Summary (MNIST)\n")
    f.write("=" * 45 + "\n\n")
    f.write(f"{'Variant':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}\n")
    f.write("-" * 60 + "\n")

    for v, s in summary.items():
        f.write(
            f"{v:<20} "
            f"{s['mean']:>8.4f} "
            f"{s['std']:>8.4f} "
            f"{s['min']:>8.4f} "
            f"{s['max']:>8.4f}\n"
        )

# -------------------------------
# Create bar plot (publication-style)
# -------------------------------
variants = list(summary.keys())
means = [summary[v]["mean"] for v in variants]
stds = [summary[v]["std"] for v in variants]

plt.figure(figsize=(10, 5))
plt.bar(variants, means, yerr=stds, capsize=6)
plt.ylabel("Test Accuracy")
plt.title("Ablation Study: Component Contribution (MNIST)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig("results/ablation_barplot.png", dpi=200)
plt.close()

print("Ablation aggregation and visualization completed.")
