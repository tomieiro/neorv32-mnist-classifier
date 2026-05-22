from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--clock-hz", type=float, default=27_000_000)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    cycles = df["cycles"]
    cycles_mean = cycles.mean()
    accuracy = (df["label"] == df["pred"]).mean()

    print(f"file={args.csv}")
    print(f"samples={len(df)}")
    print(f"accuracy_subset={accuracy:.4f}")
    print(f"latency_cycles_mean={int(cycles_mean)}")
    print(f"latency_cycles_min={int(cycles.min())}")
    print(f"latency_cycles_max={int(cycles.max())}")
    print(f"latency_ms_mean={cycles_mean / args.clock_hz * 1000.0:.3f}")
    print(f"throughput_inf_per_s={args.clock_hz / cycles_mean:.3f}")
    print(f"roundtrip_ms_mean={df['roundtrip_ms'].mean():.3f}")


if __name__ == "__main__":
    main()
