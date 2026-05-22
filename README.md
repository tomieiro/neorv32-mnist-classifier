# MNIST CNN no NEORV32

Projeto bare-metal para rodar uma micro-CNN int8 de MNIST na Tang Nano 9K usando NEORV32. O fluxo evita TFLM no firmware: o host treina, quantiza e exporta headers C; o firmware recebe imagens pela UART, executa a inferencia e retorna classe e ciclos.

Para a sequencia curta de uso, veja `FLOW.md`.

## Layout

- `host/train/`: treino, exportacao e validacao int8.
- `host/deploy/`: envio UART e resumo de resultados.
- `host/utils/`: utilitarios compartilhados.
- `firmware/`: C bare-metal para NEORV32, sem malloc, filesystem, SO ou bibliotecas de host.
- `experiments/`: CSV de resultados, notas e graficos de Pareto.
- `scripts/`: comandos auxiliares para build, upload e coleta.

## Fluxo rapido

```sh
cd mnist_neorv32/host
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
make all
```

Depois compile o firmware:

```sh
cd ../firmware
make clean
make exe USER_FLAGS="-Ofast"
```

Envio de imagens para a placa:

```sh
cd ..
python3 scripts/upload.py --port /dev/ttyUSB2
cd host
make send summary PORT=/dev/ttyUSB2 COUNT=100
```

## Protocolo UART

PC envia:

- 2 bytes: `M` `N`
- 2 bytes: tamanho little-endian, valor `784`
- 784 bytes: imagem MNIST `uint8`

Placa responde:

```text
PRED=<digit> CYCLES=<cycles>
```

## Modelo inicial

```text
28x28x1
Conv2D 3x3, 4 filtros, valid + ReLU -> 26x26x4
MaxPool 2x2 -> 13x13x4
Conv2D 3x3, 8 filtros, valid + ReLU -> 11x11x8
MaxPool 2x2 -> 5x5x8
Flatten -> 200
Dense -> 10
Argmax
```

## Medicoes

Preencha `experiments/results.csv` com latencia, throughput, acuracia, tamanho do binario e recursos FPGA. Gere os graficos:

```sh
cd mnist_neorv32/experiments
python3 pareto_plot.py
```

`latency_cycles` deve medir apenas `mnist_predict()`. O round-trip UART fica separado no CSV gerado por `host/deploy/send_uart.py`.
