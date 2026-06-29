#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-/dev/ttyUSB1}"
COUNT="${COUNT:-10}"
START="${START:-0}"
CLOCK_HZ="${CLOCK_HZ:-27000000}"
ACCEL="${ACCEL:-0}"
OUT="${OUT:-experiments/p2_fastmul_fastshift_uart_results.csv}"
MARCH="${MARCH:-rv32imc_zicsr_zifencei}"
USER_FLAGS="${USER_FLAGS:--O3}"
BUILD_FPGA="${BUILD_FPGA:-1}"
FLASH_FPGA="${FLASH_FPGA:-1}"
REPORT_DIR="${REPORT_DIR:-experiments/hw_reports/p2_fastmul_fastshift}"

echo "[p2] Configurando processador: M=true, fast_mul=true, fast_shift=true"
python3 set_neorv32_config.py --enable-m true --fast-mul true --fast-shift true

if [[ "${BUILD_FPGA}" == "1" ]]; then
  echo "[p2] Sintetizando FPGA"
  make -C tang_nano_9k_neorv32 clean
  make -C tang_nano_9k_neorv32 build
  python3 experiments/save_gowin_snapshot.py "$(basename "${REPORT_DIR}")"
else
  echo "[p2] AVISO: BUILD_FPGA=0; a configuracao em VHDL foi atualizada, mas o bitstream existente pode estar stale"
fi

if [[ "${FLASH_FPGA}" == "1" ]]; then
  echo "[p2] Gravando bitstream na FPGA"
  make -C tang_nano_9k_neorv32 flash
fi

echo "[p2] Compilando firmware com MARCH=${MARCH} USER_FLAGS=${USER_FLAGS} ACCEL=${ACCEL}"
mkdir -p neorv32-mnist-classifier/firmware/generated
cp neorv32-mnist-classifier/host/generated/model_meta.h neorv32-mnist-classifier/firmware/generated/model_meta.h
cp neorv32-mnist-classifier/host/generated/weights.h neorv32-mnist-classifier/firmware/generated/weights.h
make -C neorv32-mnist-classifier/firmware clean
mkdir -p neorv32-mnist-classifier/firmware/generated
cp neorv32-mnist-classifier/host/generated/model_meta.h neorv32-mnist-classifier/firmware/generated/model_meta.h
cp neorv32-mnist-classifier/host/generated/weights.h neorv32-mnist-classifier/firmware/generated/weights.h
make -C neorv32-mnist-classifier/firmware bin MARCH="${MARCH}" USER_FLAGS="${USER_FLAGS}" ACCEL="${ACCEL}" PORT="${PORT}"

echo "[p2] Upload do firmware"
python3 neorv32-mnist-classifier/firmware/deploy/upload.py --port "${PORT}" --bin neorv32-mnist-classifier/firmware/neorv32_legacy_exe.bin

echo "[p2] Coletando resultados UART em ${OUT}"
python3 neorv32-mnist-classifier/host/deploy/send_uart.py --port "${PORT}" --count "${COUNT}" --start "${START}" --out "${OUT}"

echo "[p2] Resumo"
python3 neorv32-mnist-classifier/host/deploy/summarize_uart.py "${OUT}" --clock-hz "${CLOCK_HZ}"
