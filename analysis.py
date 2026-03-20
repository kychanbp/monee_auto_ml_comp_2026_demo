"""
analysis.py — Generate experiment progress chart from results.csv
Run: python analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")
    if not os.path.exists(results_path):
        print("No results.csv found. Run some experiments first.")
        return

    df = pd.read_csv(results_path)
    df = df[df["val_auc"] > 0]  # exclude crashes
    df = df.reset_index(drop=True)
    df["experiment"] = df.index + 1

    # Track running best
    df["best_auc"] = df["val_auc"].where(df["status"] == "keep").ffill()

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot all experiments
    colors = {"keep": "#2ecc71", "discard": "#e74c3c", "crash": "#95a5a6"}
    for status, group in df.groupby("status"):
        ax.scatter(
            group["experiment"], group["val_auc"],
            c=colors.get(status, "#95a5a6"),
            label=status, s=40, zorder=3,
        )

    # Plot improvement frontier
    keeps = df[df["status"] == "keep"]
    if not keeps.empty:
        ax.step(
            keeps["experiment"], keeps["val_auc"],
            where="post", color="#2ecc71", linewidth=2,
            alpha=0.5, label="best frontier",
        )

    ax.set_xlabel("Experiment #")
    ax.set_ylabel("OOF ROC AUC")
    ax.set_title("Experiment Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Stats
    total = len(df)
    n_keep = len(df[df["status"] == "keep"])
    n_discard = len(df[df["status"] == "discard"])
    best = df["val_auc"].max()
    ax.text(
        0.02, 0.98,
        f"Total: {total} | Keep: {n_keep} | Discard: {n_discard} | Best AUC: {best:.6f}",
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "progress.png")
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
