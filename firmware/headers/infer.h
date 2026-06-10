#ifndef INFER_H
#define INFER_H

#include <stdint.h>

typedef struct {
  int pred;
  const int32_t *logits;
} mnist_inference_result_t;

void mnist_model_init(void);
void mnist_model_run(const uint8_t input[784], mnist_inference_result_t *result);

#endif
