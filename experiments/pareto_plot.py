from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def nondominated_latency_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    data = df.dropna(subset=["latency_cycles", "accuracy_subset"]).copy()
    keep = []
    for i, row in data.iterrows():
        dominated = False
        for j, other in data.iterrows():
            if i == j:
                continue
            no_worse = (
                other["latency_cycles"] <= row["latency_cycles"]
                and other["accuracy_subset"] >= row["accuracy_subset"]
            )
            strictly_better = (
                other["latency_cycles"] < row["latency_cycles"]
                or other["accuracy_subset"] > row["accuracy_subset"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        keep.append(not dominated)
    return data.loc[keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("results.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    for col in ("accuracy_subset", "latency_cycles", "throughput_inf_per_s", "luts"):
        df[col] = numeric(df, col)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pareto = nondominated_latency_accuracy(df)
    ax = df.plot.scatter(x="latency_cycles", y="accuracy_subset", label="all")
    if not pareto.empty:
        pareto.plot.scatter(x="latency_cycles", y="accuracy_subset", color="red", label="pareto", ax=ax)
    for _, row in df.dropna(subset=["latency_cycles", "accuracy_subset"]).iterrows():
        ax.annotate(row["experiment_id"], (row["latency_cycles"], row["accuracy_subset"]))
    ax.set_title("MNIST CNN: latency vs accuracy")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(args.out_dir / "pareto_latency_accuracy.png", dpi=160)

    if df["luts"].notna().any() and df["throughput_inf_per_s"].notna().any():
        ax = df.plot.scatter(x="luts", y="throughput_inf_per_s")
        ax.set_title("Throughput vs LUTs")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(args.out_dir / "throughput_luts.png", dpi=160)

    if df["luts"].notna().any() and df["latency_cycles"].notna().any():
        ax = df.plot.scatter(x="luts", y="latency_cycles")
        ax.set_title("Latency vs LUTs")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(args.out_dir / "latency_luts.png", dpi=160)

    pareto.to_csv(args.out_dir / "pareto_points.csv", index=False)
    print(f"wrote={args.out_dir}")


if __name__ == "__main__":
    main()
