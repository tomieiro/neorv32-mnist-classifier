PY ?= python3
PORT ?= /dev/ttyUSB2
COUNT ?= 20
OUT ?= ../experiments/uart_results.csv

.PHONY: deps train export validate send summary all clean

deps:
	$(PY) -m pip install -r requirements.txt

train:
	$(PY) -m train.train_cnn --epochs 4

export:
	$(PY) -m train.quantize_export

validate:
	$(PY) -m train.validate_int8 --limit 1000

send:
	$(PY) -m deploy.send_uart --port $(PORT) --count $(COUNT) --out $(OUT)

summary:
	$(PY) -m deploy.summarize_uart $(OUT)

all: train export validate

clean:
	rm -rf __pycache__
	rm -f generated/golden_vectors.json generated/quantization.json
