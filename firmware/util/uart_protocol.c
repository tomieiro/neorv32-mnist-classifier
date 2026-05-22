#include "uart_protocol.h"

#include "cnn_params.h"
#include "platform_neorv32.h"

int uart_recv_mnist_frame(uint8_t image[MNIST_IMAGE_SIZE]) {
  uint8_t m0 = platform_get_u8_blocking();
  uint8_t m1 = platform_get_u8_blocking();
  uint8_t len0 = platform_get_u8_blocking();
  uint8_t len1 = platform_get_u8_blocking();
  uint16_t len = (uint16_t)len0 | ((uint16_t)len1 << 8);

  if ((m0 != (uint8_t)'M') || (m1 != (uint8_t)'N')) {
    return -1;
  }
  if (len != MNIST_IMAGE_SIZE) {
    return -2;
  }

  for (uint32_t i = 0; i < MNIST_IMAGE_SIZE; i++) {
    image[i] = platform_get_u8_blocking();
  }
  return 0;
}

void uart_send_result(int pred, uint64_t cycles) {
  platform_puts("PRED=");
  platform_put_u32((uint32_t)pred);
  platform_puts(" CYCLES=");
  platform_put_u64(cycles);
  platform_puts("\n");
}

void uart_send_error(const char *reason) {
  platform_puts("ERR=");
  platform_puts(reason);
  platform_puts("\n");
}
