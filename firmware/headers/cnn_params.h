#ifndef CNN_PARAMS_H
#define CNN_PARAMS_H

#define MNIST_IMAGE_W 28
#define MNIST_IMAGE_H 28
#define MNIST_IMAGE_SIZE (MNIST_IMAGE_W * MNIST_IMAGE_H)

#define CONV1_IN_CH 1
#define CONV1_OUT_CH 4
#define CONV1_OUT_W 26
#define CONV1_OUT_H 26

#define POOL1_OUT_W 13
#define POOL1_OUT_H 13

#define CONV2_IN_CH 4
#define CONV2_OUT_CH 8
#define CONV2_OUT_W 11
#define CONV2_OUT_H 11

#define POOL2_OUT_W 5
#define POOL2_OUT_H 5

#define DENSE_IN_SIZE (POOL2_OUT_W * POOL2_OUT_H * CONV2_OUT_CH)
#define DENSE_OUT_SIZE 10

#endif
