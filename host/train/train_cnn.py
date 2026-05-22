from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from utils.dataset import load_mnist


MODEL_PATH = Path("generated/mnist_cnn_4_8.keras")


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(4, 3, activation="relu"),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(8, 3, activation="relu"),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    (x_train, y_train), (x_test, y_test) = load_mnist()

    model = build_model()
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=128, validation_split=0.1, verbose=2)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    model.save(MODEL_PATH)

    print(f"float32_accuracy={acc:.4f}")
    print(f"saved={MODEL_PATH}")


if __name__ == "__main__":
    main()
