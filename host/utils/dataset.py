from __future__ import annotations

import numpy as np


def load_mnist():
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    return (x_train, y_train.astype(np.int64)), (x_test, y_test.astype(np.int64))


def load_mnist_uint8():
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train.astype(np.uint8), y_train.astype(np.uint8)), (
        x_test.astype(np.uint8),
        y_test.astype(np.uint8),
    )
