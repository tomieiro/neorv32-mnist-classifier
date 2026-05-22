#include "platform_neorv32.h"

#include <neorv32.h>

void platform_init(void) {
  neorv32_rte_setup();
  neorv32_uart0_setup(MNIST_UART_BAUD, 0);
}

void platform_puts(const char *s) {
  neorv32_uart0_puts(s);
}

void platform_put_u32(uint32_t value) {
  neorv32_uart0_printf("%u", value);
}

void platform_put_u64(uint64_t value) {
  if (value > 4294967295ULL) {
    uint32_t hi = (uint32_t)(value / 1000000000ULL);
    uint32_t lo = (uint32_t)(value % 1000000000ULL);
    neorv32_uart0_printf("%u%09u", hi, lo);
  } else {
    neorv32_uart0_printf("%u", (uint32_t)value);
  }
}

uint8_t platform_get_u8_blocking(void) {
  return (uint8_t)neorv32_uart0_getc();
}
