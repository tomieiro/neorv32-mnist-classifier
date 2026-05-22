#include "cycle_count.h"

#include <neorv32.h>

uint64_t get_cycles64(void) {
  return neorv32_cpu_get_cycle();
}
