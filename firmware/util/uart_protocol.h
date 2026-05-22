#ifndef UART_PROTOCOL_H
#define UART_PROTOCOL_H

#include <stdint.h>

int uart_recv_mnist_frame(uint8_t image[784]);
void uart_send_result(int pred, uint64_t cycles);
void uart_send_error(const char *reason);

#endif
