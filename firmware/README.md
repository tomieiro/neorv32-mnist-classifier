# Firmware MNIST

## Arquitetura da Rede

Rede: `M1_4_8`, CNN int8 para MNIST.

Fluxo:

```text
Input 28x28x1 uint8
-> Conv2D 3x3, 4 filtros, ReLU, shift
-> MaxPool 2x2
-> Conv2D 3x3, 8 filtros, ReLU, shift
-> MaxPool 2x2
-> Flatten 5x5x8 = 200
-> Dense 200 -> 10
-> Argmax
```

Dimensoes:

```text
28x28x1
26x26x4
13x13x4
11x11x8
5x5x8
200
10 logits
```

Pesos usados pelo firmware:

```text
generated/weights.h
generated/model_meta.h
```

O firmware nao usa TensorFlow Lite. A inferencia e C bare-metal em `src/infer.c`, com buffers estaticos e aritmetica inteira:

```text
int8 weights
int32 bias/acumulacao
int16 ativacoes intermediarias
```

Se treinar de novo, exporte antes de compilar:

```bash
cd ../host
make export
cd ../firmware
make legacy
```
