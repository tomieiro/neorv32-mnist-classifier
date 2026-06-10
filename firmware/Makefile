RISCV_PREFIX ?= riscv64-elf-
MARCH ?= rv32i_zicsr_zifencei
EFFORT ?= -O3
PORT ?= /dev/ttyUSB1

HOST_GENERATED_DIR ?= ../host/generated
MODEL_FILE ?= $(HOST_GENERATED_DIR)/mnist_cnn_4_8.keras
MODEL_META ?= generated/model_meta.h
MODEL_WEIGHTS ?= generated/weights.h


USER_FLAGS += -ggdb -gdwarf-3
USER_FLAGS += -Wl,--defsym,__neorv32_rom_size=64k
USER_FLAGS += -Wl,--defsym,__neorv32_ram_size=32k

APP_SRC += $(wildcard src/*.c) $(wildcard util/*.c)
APP_INC += -I headers -I util -I generated -I .

NEORV32_HOME ?= ../../tang_nano_9k_neorv32/sdk

include $(NEORV32_HOME)/sw/common/common.mk

.PHONY: all check-generated-fresh legacy bupload upload-legacy bupload-legacy clean-legacy

all: legacy

check-generated-fresh:
	@if [ ! -f "$(HOST_GENERATED_DIR)/model_meta.h" ] || [ ! -f "$(HOST_GENERATED_DIR)/weights.h" ]; then \
		echo "ERROR: missing exported headers in $(HOST_GENERATED_DIR). Run: cd ../host && make export"; \
		exit 1; \
	fi
	@if [ -f "$(MODEL_FILE)" ] && { [ "$(MODEL_FILE)" -nt "$(HOST_GENERATED_DIR)/model_meta.h" ] || [ "$(MODEL_FILE)" -nt "$(HOST_GENERATED_DIR)/weights.h" ]; }; then \
		echo "ERROR: trained model is newer than exported headers. Run: cd ../host && make export"; \
		echo "model=$(MODEL_FILE)"; \
		echo "headers=$(HOST_GENERATED_DIR)/model_meta.h $(HOST_GENERATED_DIR)/weights.h"; \
		exit 1; \
	fi

$(MODEL_META): $(HOST_GENERATED_DIR)/model_meta.h | check-generated-fresh
	mkdir -p generated
	cp $< $@

$(MODEL_WEIGHTS): $(HOST_GENERATED_DIR)/weights.h | check-generated-fresh
	mkdir -p generated
	cp $< $@

build/infer.c.o: $(MODEL_META) $(MODEL_WEIGHTS) | check-generated-fresh

exe: $(MODEL_META) $(MODEL_WEIGHTS) | check-generated-fresh
bin: exe legacy

legacy: exe
	python3 deploy/make_legacy_exe.py $(APP_FLT) neorv32_legacy_exe.bin

bupload: upload-legacy

upload-legacy: legacy
	python3 deploy/upload.py --port $(PORT) --bin neorv32_legacy_exe.bin

bupload-legacy: upload-legacy

clean: clean-legacy

clean-legacy:
	rm -rf neorv32_legacy_exe.bin __pycache__ generated
