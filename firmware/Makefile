RISCV_PREFIX ?= riscv64-elf-
MARCH ?= rv32i_zicsr_zifencei
EFFORT ?= -Os
PORT ?= /dev/ttyUSB2


USER_FLAGS += -ggdb -gdwarf-3
USER_FLAGS += -Wl,--defsym,__neorv32_rom_size=64k
USER_FLAGS += -Wl,--defsym,__neorv32_ram_size=32k

APP_SRC += $(wildcard src/*.c) $(wildcard util/*.c)
APP_INC += -I headers -I util -I generated -I .

NEORV32_HOME ?= ../../tang_nano_9k_neorv32/sdk

include $(NEORV32_HOME)/sw/common/common.mk

all: legacy
legacy: exe
	python3 ../scripts/make_legacy_exe.py $(APP_FLT) neorv32_legacy_exe.bin

upload-legacy: legacy
	python3 ../scripts/upload.py --port $(PORT) --bin neorv32_legacy_exe.bin

clean: clean-legacy

clean-legacy:
	rm -rf neorv32_legacy_exe.bin
