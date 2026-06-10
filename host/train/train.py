from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf

from train.config import (
    CONV1_FILTERS,
    CONV1_KERNEL,
    CONV2_FILTERS,
    CONV2_KERNEL,
    DENSE_UNITS,
    INPUT_SHAPE,
    MODEL_PATH,
    TRAIN_BATCH_SIZE,
    TRAIN_VALIDATION_SPLIT,
)
from train.quantize import run_export


def load_mnist():
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    return (x_train, y_train.astype(np.int64)), (x_test, y_test.astype(np.int64))


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=INPUT_SHAPE),
            tf.keras.layers.Conv2D(CONV1_FILTERS, CONV1_KERNEL, activation="relu"),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(CONV2_FILTERS, CONV2_KERNEL, activation="relu"),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(DENSE_UNITS),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def run_train(epochs: int) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    (x_train, y_train), (x_test, y_test) = load_mnist()

    model = build_model()
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=TRAIN_BATCH_SIZE,
        validation_split=TRAIN_VALIDATION_SPLIT,
        verbose=2,
    )
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    model.save(MODEL_PATH)

    print(f"float32_accuracy={acc:.4f}")
    print(f"saved={MODEL_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export the MNIST host pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save the float32 model.")
    train_parser.add_argument("--epochs", type=int, default=4)

    subparsers.add_parser("export", help="Quantize the trained model and export firmware headers.")

    args = parser.parse_args()

    if args.command == "train":
        run_train(args.epochs)
    else:
        run_export()


if __name__ == "__main__":
    main()
