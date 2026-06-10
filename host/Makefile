PY ?= python3
PORT ?= /dev/ttyUSB2
COUNT ?= 20
OUT ?= ../experiments/uart_results.csv

.PHONY: deps train export validate send summary all clean

deps:
	$(PY) -m pip install -r requirements.txt

train:
	$(PY) -m train.train train --epochs 5

export:
	$(PY) -m train.train export

validate:
	@echo "host-side int8 validation was removed"
	@echo "use firmware build/upload plus UART inference to validate the exported model"

send:
	$(PY) -m deploy.send_uart --port $(PORT) --count $(COUNT) --out $(OUT)

summary:
	$(PY) -m deploy.summarize_uart $(OUT)

all: train export

clean:
	rm -rf __pycache__
	rm -f generated/golden_vectors.json generated/quantization.json
