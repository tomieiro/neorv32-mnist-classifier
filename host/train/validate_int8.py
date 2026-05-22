from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from utils.dataset import load_mnist_uint8


def parse_array(text: str, name: str, dtype) -> np.ndarray:
    match = re.search(rf"{name}[^=]*=\s*(.*?)\s*;", text, re.S)
    if not match:
        raise RuntimeError(f"missing array: {name}")
    return np.array([int(x) for x in re.findall(r"-?\d+", match.group(1))], dtype=dtype)


def load_params(path: Path) -> dict:
    weights = (path / "weights.h").read_text()
    meta = (path / "model_meta.h").read_text()

    def define(name: str) -> int:
        return int(re.search(rf"#define\s+{name}\s+(\d+)", meta).group(1))

    return {
        "conv1_w": parse_array(weights, "mnist_conv1_weights", np.int8).reshape(4, 1, 3, 3).astype(np.int32),
        "conv1_b": parse_array(weights, "mnist_conv1_bias", np.int32),
        "conv2_w": parse_array(weights, "mnist_conv2_weights", np.int8).reshape(8, 4, 3, 3).astype(np.int32),
        "conv2_b": parse_array(weights, "mnist_conv2_bias", np.int32),
        "dense_w": parse_array(weights, "mnist_dense_weights", np.int8).reshape(10, 200).astype(np.int32),
        "dense_b": parse_array(weights, "mnist_dense_bias", np.int32),
        "zp": define("MNIST_INPUT_ZERO_POINT"),
        "s1": define("MNIST_CONV1_SHIFT"),
        "s2": define("MNIST_CONV2_SHIFT"),
        "sd": define("MNIST_DENSE_SHIFT"),
    }


def relu_shift(x: np.ndarray, shift: int) -> np.ndarray:
    x = np.maximum(x, 0)
    if shift:
        x >>= shift
    return np.clip(x, 0, 32767).astype(np.int16)


def maxpool2x2(x: np.ndarray) -> np.ndarray:
    return np.maximum.reduce(
        (
            x[:, 0::2, 0::2, :],
            x[:, 0::2, 1::2, :],
            x[:, 1::2, 0::2, :],
            x[:, 1::2, 1::2, :],
        )
    ).astype(np.int16)


def predict(images: np.ndarray, p: dict) -> np.ndarray:
    x = images.astype(np.int32) - p["zp"]

    win1 = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(1, 2))
    c1 = np.einsum("nyxkl,okl->nyxo", win1, p["conv1_w"][:, 0], optimize=True)
    c1 = relu_shift(c1 + p["conv1_b"].reshape(1, 1, 1, 4), p["s1"])
    p1 = maxpool2x2(c1)

    win2 = np.lib.stride_tricks.sliding_window_view(p1.astype(np.int32), (3, 3), axis=(1, 2))
    c2 = np.einsum("nyxckl,ockl->nyxo", win2, p["conv2_w"], optimize=True)
    c2 = relu_shift(c2 + p["conv2_b"].reshape(1, 1, 1, 8), p["s2"])
    p2 = maxpool2x2(c2).reshape(images.shape[0], 200).astype(np.int32)

    logits = p2 @ p["dense_w"].T
    logits += p["dense_b"].reshape(1, 10)
    if p["sd"]:
        logits >>= p["sd"]
    return np.argmax(logits, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    params = load_params(Path("generated"))
    (_, _), (x_test, y_test) = load_mnist_uint8()
    pred = predict(x_test[: args.limit], params)
    acc = np.mean(pred == y_test[: args.limit])
    print(f"int8_accuracy={acc:.4f} samples={args.limit}")


if __name__ == "__main__":
    main()
