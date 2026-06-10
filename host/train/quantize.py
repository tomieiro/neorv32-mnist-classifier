from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from train.config import (
    CONV1_SHIFT,
    CONV2_SHIFT,
    DENSE_SHIFT,
    FW_DIR,
    INPUT_ZERO_POINT,
    MODEL_PATH,
    OUT_DIR,
)


def quantize_symmetric(values: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(values)))
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    values_q = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
    return values_q, scale


def c_array(values: np.ndarray, indent: int = 0) -> str:
    pad = " " * indent
    if values.ndim == 1:
        return "{" + ", ".join(str(int(x)) for x in values) + "}"
    body = ",\n".join(pad + "  " + c_array(v, indent + 2) for v in values)
    return "{\n" + body + "\n" + pad + "}"


def write_headers(arrays: dict[str, np.ndarray]) -> None:
    weights_h = f"""#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <stdint.h>

static const int8_t mnist_conv1_weights[4][1][3][3] = {c_array(arrays["conv1_w"])};
static const int32_t mnist_conv1_bias[4] = {c_array(arrays["conv1_b"])};
static const int8_t mnist_conv2_weights[8][4][3][3] = {c_array(arrays["conv2_w"])};
static const int32_t mnist_conv2_bias[8] = {c_array(arrays["conv2_b"])};
static const int8_t mnist_dense_weights[10][200] = {c_array(arrays["dense_w"])};
static const int32_t mnist_dense_bias[10] = {c_array(arrays["dense_b"])};

#endif
"""
    meta_h = f"""#ifndef MODEL_META_H
#define MODEL_META_H

#define MNIST_MODEL_NAME "mnist_cnn_m1_4_8_int8"
#define MNIST_INPUT_ZERO_POINT {INPUT_ZERO_POINT}
#define MNIST_CONV1_SHIFT {CONV1_SHIFT}
#define MNIST_CONV2_SHIFT {CONV2_SHIFT}
#define MNIST_DENSE_SHIFT {DENSE_SHIFT}

#endif
"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FW_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "weights.h").write_text(weights_h)
    (OUT_DIR / "model_meta.h").write_text(meta_h)
    for name in ("weights.h", "model_meta.h"):
        shutil.copy2(OUT_DIR / name, FW_DIR / name)
    for stale_path in (
        OUT_DIR / "test_images.h",
        FW_DIR / "test_images.h",
        FW_DIR / MODEL_PATH.name,
    ):
        stale_path.unlink(missing_ok=True)


def run_export() -> None:
    model = tf.keras.models.load_model(MODEL_PATH)
    conv1_w, conv1_b = model.layers[0].get_weights()
    conv2_w, conv2_b = model.layers[2].get_weights()
    dense_w, dense_b = model.layers[5].get_weights()

    q_conv1_w, conv1_scale = quantize_symmetric(np.transpose(conv1_w, (3, 2, 0, 1)))
    q_conv2_w, conv2_scale = quantize_symmetric(np.transpose(conv2_w, (3, 2, 0, 1)))
    q_dense_w, dense_scale = quantize_symmetric(np.transpose(dense_w, (1, 0)))

    input_scale = 1.0 / 255.0
    conv1_acc_scale = input_scale * conv1_scale
    conv1_act_scale = conv1_acc_scale * (1 << CONV1_SHIFT)
    conv2_acc_scale = conv1_act_scale * conv2_scale
    conv2_act_scale = conv2_acc_scale * (1 << CONV2_SHIFT)
    dense_acc_scale = conv2_act_scale * dense_scale

    arrays = {
        "conv1_w": q_conv1_w,
        "conv1_b": np.rint(conv1_b / conv1_acc_scale).astype(np.int32),
        "conv2_w": q_conv2_w,
        "conv2_b": np.rint(conv2_b / conv2_acc_scale).astype(np.int32),
        "dense_w": q_dense_w,
        "dense_b": np.rint(dense_b / dense_acc_scale).astype(np.int32),
    }
    write_headers(arrays)
    print("exported=generated -> ../firmware/generated")
