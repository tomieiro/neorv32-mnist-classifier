# Fluxo Curto

## 1. Gerar pesos

```sh
cd mnist_neorv32/host
. .venv/bin/activate
make all
```

## 2. Compilar firmware

```sh
cd ../firmware
make clean
make exe USER_FLAGS="-Ofast"
```

Saida principal:

```text
firmware/neorv32_exe.bin
```

## 3. Subir para a placa

```sh
cd ..
python3 scripts/upload.py --port /dev/ttyUSB2
```

Quando o script pedir, aperte reset na Tang Nano 9K.

## 4. Medir

```sh
cd host
make send summary OUT=../experiments/uart_m1_ofast.csv
```

## 5. Repetir variacoes

Troque apenas a flag:

```sh
make exe USER_FLAGS="-O2"
make exe USER_FLAGS="-O3"
make exe USER_FLAGS="-Ofast"
```

Para cada build: upload, medir, salvar CSV e copiar o resumo para `experiments/results.csv`.
