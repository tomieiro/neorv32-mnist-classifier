#ifndef INFER_H
#define INFER_H

#include <stdint.h>

int mnist_predict(const uint8_t input[784]);
const int32_t *mnist_get_last_logits(void);

#endif
