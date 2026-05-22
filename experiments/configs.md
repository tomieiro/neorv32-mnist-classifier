# Configuracoes Experimentais

## Hardware

P0: RV32IC, sem M.
P1: RV32IMC, com M.
P2: RV32IMC com multiplicador rapido, se a configuracao VHDL couber.
P3: P2 com cache, se couber.
P4: float/Zfinx/FPU, opcional.

## Software

S0: int8 simples, `-O0`.
S1: int8 simples, `-O2`.
S2: int8 simples, `-O3`.
S3: int8 com ponteiros/buffers revisados, `-O3`.
S4: float32 opcional para contraste.

## Modelos

M0: CNN 2/4 filtros.
M1: CNN 4/8 filtros.
M2: CNN 6/12 filtros, opcional.
