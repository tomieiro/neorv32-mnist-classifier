from __future__ import annotations

from pathlib import Path

TRAIN_BATCH_SIZE = 128
TRAIN_VALIDATION_SPLIT = 0.1

MODEL_PATH = Path("generated/mnist_cnn_4_8.keras")
OUT_DIR = Path("generated")
FW_DIR = Path("../firmware/generated")

INPUT_SHAPE = (28, 28, 1)
CONV1_FILTERS = 4
CONV1_KERNEL = 3
CONV2_FILTERS = 8
CONV2_KERNEL = 3
DENSE_UNITS = 10

CONV1_SHIFT = 7
CONV2_SHIFT = 7
DENSE_SHIFT = 7
INPUT_ZERO_POINT = 0
