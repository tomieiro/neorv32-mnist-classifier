from __future__ import annotations

import argparse
import csv
import re
import struct
import time
from pathlib import Path

import numpy as np
import serial


def load_mnist_uint8():
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train.astype(np.uint8), y_train.astype(np.uint8)), (
        x_test.astype(np.uint8),
        y_test.astype(np.uint8),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("../experiments/uart_results.csv"))
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    (_, _), (x_test, y_test) = load_mnist_uint8()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    correct = 0
    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser:
        time.sleep(0.2)
        ser.reset_input_buffer()
        for idx in range(args.start, args.start + args.count):
            image = x_test[idx].reshape(784).tobytes()
            frame = b"MN" + struct.pack("<H", len(image)) + image
            t0 = time.perf_counter()
            ser.write(frame)
            ser.flush()
            line = ser.readline().decode("ascii", errors="replace").strip()
            t1 = time.perf_counter()
            match = re.search(r"PRED=(\d+)\s+CYCLES=(\d+)", line)
            pred = int(match.group(1)) if match else -1
            cycles = int(match.group(2)) if match else -1
            rows.append(
                {
                    "index": idx,
                    "label": int(y_test[idx]),
                    "pred": pred,
                    "cycles": cycles,
                    "roundtrip_ms": (t1 - t0) * 1000.0,
                    "raw": line,
                }
            )
            correct += int(pred == int(y_test[idx]))
            print(
                f"{idx:04d} label={int(y_test[idx])} pred={pred} "
                f"cycles={cycles} rt_ms={(t1 - t0) * 1000.0:.1f}"
            )

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"accuracy={correct / len(rows):.4f} samples={len(rows)}")
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
