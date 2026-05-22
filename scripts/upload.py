#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path


def read_until(ser, token: bytes, timeout: float, poke: bytes | None = None) -> None:
    end = time.monotonic() + timeout
    next_poke = time.monotonic()
    data = bytearray()
    while time.monotonic() < end:
        if poke and time.monotonic() >= next_poke:
            ser.write(poke)
            next_poke = time.monotonic() + 0.25
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            print(chunk.decode("ascii", errors="ignore"), end="", flush=True)
            data.extend(chunk)
            if token in data:
                print("", flush=True)
                return
    raise TimeoutError(f"timeout waiting for {token.decode(errors='ignore')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB2")
    parser.add_argument("--bin", type=Path, default=Path("firmware/neorv32_exe.bin"))
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    import serial

    with serial.Serial(args.port, args.baud, timeout=0.1, write_timeout=2.0) as ser:
        print("Press reset on the board.")
        read_until(ser, b"Autoboot", 20)
        ser.write(b" ")
        read_until(ser, b"CMD:>", 12, poke=b" ")

        ser.write(b"e")
        read_until(ser, b"CMD:>", 20)

        ser.write(b"u")
        read_until(ser, b"Awaiting neorv32_exe.bin", 8)

        data = args.bin.read_bytes()
        ser.write(data)
        ser.flush()
        print(f"sent={len(data)} bytes")
        read_until(ser, b"OK", 20)

        ser.write(b"x")
        time.sleep(0.5)
        print(ser.read_all().decode("ascii", errors="ignore"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
