#ifndef PLATFORM_NEORV32_H
#define PLATFORM_NEORV32_H

#include <stdint.h>

#ifndef MNIST_UART_BAUD
#define MNIST_UART_BAUD 115200
#endif

void platform_init(void);
void platform_puts(const char *s);
void platform_put_u32(uint32_t value);
void platform_put_u64(uint64_t value);
uint8_t platform_get_u8_blocking(void);

#endif
