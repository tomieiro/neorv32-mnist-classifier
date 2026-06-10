#include "infer.h"

#include <stdint.h>

#include "cnn_params.h"
#include "generated/model_meta.h"
#include "generated/weights.h"

static int16_t pool1_out[POOL1_OUT_W * POOL1_OUT_H * CONV1_OUT_CH];
static int16_t conv2_out[CONV2_OUT_W * CONV2_OUT_H * CONV2_OUT_CH];
static int16_t pool2_out[POOL2_OUT_W * POOL2_OUT_H * CONV2_OUT_CH];
static int32_t logits[DENSE_OUT_SIZE];

static inline int32_t mac_i8(int32_t acc, int16_t a, int8_t b) {
  return acc + ((int32_t)a * (int32_t)b);
}

static inline int16_t relu_shift_clip(int32_t x, uint8_t shift) {
  if (x <= 0) {
    return 0;
  }
  if (shift != 0) {
    x >>= shift;
  }
  if (x > 32767) {
    return 32767;
  }
  return (int16_t)x;
}

static inline uint32_t idx3(uint32_t y, uint32_t x, uint32_t c, uint32_t w, uint32_t ch) {
  return ((y * w) + x) * ch + c;
}

static int16_t conv1_pixel_i8(const uint8_t input[MNIST_IMAGE_SIZE],
                              uint32_t oy, uint32_t ox, uint32_t oc) {
  int32_t acc = mnist_conv1_bias[oc];
  for (uint32_t ky = 0; ky < 3; ky++) {
    for (uint32_t kx = 0; kx < 3; kx++) {
      uint32_t in_idx = ((oy + ky) * MNIST_IMAGE_W) + ox + kx;
      int16_t pix = (int16_t)input[in_idx] - (int16_t)MNIST_INPUT_ZERO_POINT;
      acc = mac_i8(acc, pix, mnist_conv1_weights[oc][0][ky][kx]);
    }
  }
  return relu_shift_clip(acc, MNIST_CONV1_SHIFT);
}

static void conv1_pool1_i8(const uint8_t input[MNIST_IMAGE_SIZE]) {
  for (uint32_t oy = 0; oy < POOL1_OUT_H; oy++) {
    for (uint32_t ox = 0; ox < POOL1_OUT_W; ox++) {
      for (uint32_t oc = 0; oc < CONV1_OUT_CH; oc++) {
        uint32_t cy = oy << 1;
        uint32_t cx = ox << 1;
        int16_t m = conv1_pixel_i8(input, cy, cx, oc);
        int16_t v = conv1_pixel_i8(input, cy, cx + 1, oc);
        if (v > m) {
          m = v;
        }
        v = conv1_pixel_i8(input, cy + 1, cx, oc);
        if (v > m) {
          m = v;
        }
        v = conv1_pixel_i8(input, cy + 1, cx + 1, oc);
        if (v > m) {
          m = v;
        }
        pool1_out[idx3(oy, ox, oc, POOL1_OUT_W, CONV1_OUT_CH)] = m;
      }
    }
  }
}

static void maxpool2x2_i16(const int16_t *input, int16_t *output,
                           uint32_t in_w, uint32_t out_w,
                           uint32_t out_h, uint32_t channels) {
  for (uint32_t oy = 0; oy < out_h; oy++) {
    for (uint32_t ox = 0; ox < out_w; ox++) {
      for (uint32_t c = 0; c < channels; c++) {
        uint32_t iy = oy << 1;
        uint32_t ix = ox << 1;
        int16_t m = input[idx3(iy, ix, c, in_w, channels)];
        int16_t v = input[idx3(iy, ix + 1, c, in_w, channels)];
        if (v > m) {
          m = v;
        }
        v = input[idx3(iy + 1, ix, c, in_w, channels)];
        if (v > m) {
          m = v;
        }
        v = input[idx3(iy + 1, ix + 1, c, in_w, channels)];
        if (v > m) {
          m = v;
        }
        output[idx3(oy, ox, c, out_w, channels)] = m;
      }
    }
  }
}

static void conv2_3x3_i8(void) {
  for (uint32_t oy = 0; oy < CONV2_OUT_H; oy++) {
    for (uint32_t ox = 0; ox < CONV2_OUT_W; ox++) {
      for (uint32_t oc = 0; oc < CONV2_OUT_CH; oc++) {
        int32_t acc = mnist_conv2_bias[oc];
        for (uint32_t ic = 0; ic < CONV2_IN_CH; ic++) {
          for (uint32_t ky = 0; ky < 3; ky++) {
            for (uint32_t kx = 0; kx < 3; kx++) {
              int16_t a = pool1_out[idx3(oy + ky, ox + kx, ic, POOL1_OUT_W, CONV2_IN_CH)];
              acc = mac_i8(acc, a, mnist_conv2_weights[oc][ic][ky][kx]);
            }
          }
        }
        conv2_out[idx3(oy, ox, oc, CONV2_OUT_W, CONV2_OUT_CH)] =
            relu_shift_clip(acc, MNIST_CONV2_SHIFT);
      }
    }
  }
}

static void dense_i8(void) {
  for (uint32_t o = 0; o < DENSE_OUT_SIZE; o++) {
    int32_t acc = mnist_dense_bias[o];
    for (uint32_t i = 0; i < DENSE_IN_SIZE; i++) {
      acc = mac_i8(acc, pool2_out[i], mnist_dense_weights[o][i]);
    }
    if (MNIST_DENSE_SHIFT != 0) {
      acc >>= MNIST_DENSE_SHIFT;
    }
    logits[o] = acc;
  }
}

static int argmax_i32(const int32_t *values, uint32_t n) {
  int best = 0;
  int32_t best_value = values[0];
  for (uint32_t i = 1; i < n; i++) {
    if (values[i] > best_value) {
      best_value = values[i];
      best = (int)i;
    }
  }
  return best;
}

static int mnist_predict(const uint8_t input[MNIST_IMAGE_SIZE]) {
  conv1_pool1_i8(input);
  conv2_3x3_i8();
  maxpool2x2_i16(conv2_out, pool2_out, CONV2_OUT_W, POOL2_OUT_W, POOL2_OUT_H, CONV2_OUT_CH);
  dense_i8();
  return argmax_i32(logits, DENSE_OUT_SIZE);
}

void mnist_model_init(void) {
}

void mnist_model_run(const uint8_t input[MNIST_IMAGE_SIZE], mnist_inference_result_t *result) {
  result->pred = mnist_predict(input);
  result->logits = logits;
}
