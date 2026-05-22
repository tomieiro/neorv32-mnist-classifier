#include <stdint.h>

#include "cnn_params.h"
#include "cycle_count.h"
#include "infer.h"
#include "platform_neorv32.h"
#include "uart_protocol.h"

static uint8_t image[MNIST_IMAGE_SIZE];

int main(void) {
  platform_init();
  platform_puts("MNIST_NEORV32_READY\n");

  while (1) {
    int status = uart_recv_mnist_frame(image);
    if (status != 0) {
      uart_send_error(status == -2 ? "BAD_LEN" : "BAD_MAGIC");
      continue;
    }

    uint64_t start = get_cycles64();
    int pred = mnist_predict(image);
    uint64_t end = get_cycles64();
    uart_send_result(pred, end - start);
  }
}
