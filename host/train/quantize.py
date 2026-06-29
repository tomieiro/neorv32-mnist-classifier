from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from train.config import (
    CONV1_SHIFT,
    CONV2_SHIFT,
    DENSE_SHIFT,
    FPGA_SRC_DIR,
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


def vhdl_int_list(values: list[int], per_line: int = 16, indent: str = "    ") -> str:
    lines = []
    for i in range(0, len(values), per_line):
        chunk = ", ".join(str(int(x)) for x in values[i : i + per_line])
        if i + per_line < len(values):
            chunk += ","
        lines.append(indent + chunk)
    return "\n".join(lines)


def conv2_vhdl_package(arrays: dict[str, np.ndarray]) -> str:
    conv2_w = arrays["conv2_w"]
    conv2_b = arrays["conv2_b"]
    flat_w: list[int] = []
    for ky in range(3):
        for kx in range(3):
            for oc in range(8):
                for ic in range(4):
                    flat_w.append(int(conv2_w[oc, ic, ky, kx]))
    flat_b = [int(x) for x in conv2_b]
    return f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package conv2_weights_pkg is

  type conv2_weight_mem_t is array (0 to 287) of integer range -128 to 127;
  type conv2_bias_mem_t is array (0 to 7) of integer range -2147483647 to 2147483647;

  -- Layout: weights[(kpos * 8 + oc) * 4 + ic], where kpos = ky * 3 + kx.
  constant conv2_weights_c : conv2_weight_mem_t := (
{vhdl_int_list(flat_w)}
  );

  constant conv2_bias_c : conv2_bias_mem_t := (
{vhdl_int_list(flat_b, per_line=8)}
  );

  function conv2_w_index(kpos : natural; oc : natural; ic : natural) return natural;

end package;

package body conv2_weights_pkg is

  function conv2_w_index(kpos : natural; oc : natural; ic : natural) return natural is
  begin
    return (kpos * 32) + (oc * 4) + ic;
  end function;

end package body;
"""


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
    FPGA_SRC_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "weights.h").write_text(weights_h)
    (OUT_DIR / "model_meta.h").write_text(meta_h)
    (OUT_DIR / "conv2_weights_pkg.vhd").write_text(conv2_vhdl_package(arrays))
    for name in ("weights.h", "model_meta.h"):
        shutil.copy2(OUT_DIR / name, FW_DIR / name)
    shutil.copy2(OUT_DIR / "conv2_weights_pkg.vhd", FPGA_SRC_DIR / "conv2_weights_pkg.vhd")
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
