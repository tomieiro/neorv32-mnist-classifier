#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
from pathlib import Path

LEGACY_SIGNATURE = 0x4788CAFE


def read_words_le(data: bytes) -> list[int]:
    if len(data) % 4:
        data += b"\x00" * (4 - (len(data) % 4))
    return [struct.unpack_from("<I", data, i)[0] for i in range(0, len(data), 4)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Tang Nano legacy NEORV32 executable format.")
    parser.add_argument("raw_image", type=Path, help="Raw binary image, usually firmware/elf.bin")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    raw = args.raw_image.read_bytes()
    if len(raw) % 4:
        raw += b"\x00" * (4 - (len(raw) % 4))

    checksum = sum(read_words_le(raw)) & 0xFFFFFFFF
    checksum = (-checksum) & 0xFFFFFFFF

    with args.output.open("wb") as f:
        f.write(struct.pack("<III", LEGACY_SIGNATURE, len(raw), checksum))
        f.write(raw)

    print(f"legacy_exe={args.output} size={len(raw)} checksum=0x{checksum:08x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
