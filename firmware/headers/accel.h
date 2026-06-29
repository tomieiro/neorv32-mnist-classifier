#ifndef ACCEL_H
#define ACCEL_H

// Driver for the model-specific conv2 accelerator mapped on the NEORV32 XBUS
// of the Tang Nano 9K SoC. See tang_nano_9k_neorv32/ACELLERATOR.md.
//
// Build with -DMNIST_USE_ACCEL=1 only for bitstreams that implement the
// accelerator window at 0x90000000: on older bitstreams any access to that
// region raises a bus-fault exception.

#include <stdint.h>

#ifndef MNIST_USE_ACCEL
#define MNIST_USE_ACCEL 0
#endif

#define ACCEL_BASE 0x90000000u
#define ACCEL_CTRL (*(volatile uint32_t *)(ACCEL_BASE + 0x00u))
#define ACCEL_ID (*(volatile uint32_t *)(ACCEL_BASE + 0x00u))
#define ACCEL_STATUS (*(volatile uint32_t *)(ACCEL_BASE + 0x04u))
#define ACCEL_ACT01 (*(volatile uint32_t *)(ACCEL_BASE + 0x08u))
#define ACCEL_ACT23 (*(volatile uint32_t *)(ACCEL_BASE + 0x0Cu))
#define ACCEL_OUT01 (*(volatile uint32_t *)(ACCEL_BASE + 0x10u))
#define ACCEL_OUT23 (*(volatile uint32_t *)(ACCEL_BASE + 0x14u))
#define ACCEL_OUT45 (*(volatile uint32_t *)(ACCEL_BASE + 0x18u))
#define ACCEL_OUT67 (*(volatile uint32_t *)(ACCEL_BASE + 0x1Cu))

#define ACCEL_ID_WORD 0x43563241u /* "CV2A" */

static inline int accel_present(void) {
  return ACCEL_ID == ACCEL_ID_WORD;
}

static inline uint32_t accel_pack_i16x2(int16_t lo, int16_t hi) {
  return ((uint32_t)(uint16_t)lo) | ((uint32_t)(uint16_t)hi << 16);
}

static inline int16_t accel_unpack_lo_i16(uint32_t value) {
  return (int16_t)(uint16_t)value;
}

static inline int16_t accel_unpack_hi_i16(uint32_t value) {
  return (int16_t)(uint16_t)(value >> 16);
}

static inline void accel_conv2_start(void) {
  ACCEL_CTRL = 1u;
}

static inline void accel_conv2_push_kpos(uint32_t act01, uint32_t act23) {
  ACCEL_ACT01 = act01;
  ACCEL_ACT23 = act23;
}

#endif
