#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import serial
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify a custom image on the Tang Nano 9K NEORV32.")
    parser.add_argument("image_path", type=Path, help="Path to the JPG/PNG image")
    parser.add_argument("--port", default="/dev/ttyUSB1", help="Serial port of the board")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--invert", choices=["auto", "yes", "no"], default="auto",
                        help="Invert colors (MNIST needs white digit on black background)")
    args = parser.parse_args()

    if not args.image_path.exists():
        print(f"Error: Image not found at {args.image_path}", file=sys.stderr)
        return 1

    # Load and preprocess image
    try:
        img = Image.open(args.image_path).convert("L")
    except Exception as e:
        print(f"Error opening image: {e}", file=sys.stderr)
        return 1

    # Resize to 28x28 (MNIST size) using LANCZOS
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_np = np.array(img, dtype=np.uint8)

    # Handle inversion
    mean_val = img_np.mean()
    should_invert = False
    if args.invert == "yes":
        should_invert = True
    elif args.invert == "no":
        should_invert = False
    else:
        # Fundo branco
        if mean_val > 127:
            should_invert = True

    if should_invert:
        img_np = 255 - img_np
        print("Note: Image was auto-inverted to match MNIST white-on-black convention.")

    # Serialize image
    image_bytes = img_np.tobytes()
    frame = b"MN" + struct.pack("<H", len(image_bytes)) + image_bytes

    print(f"Connecting to Tang Nano 9K on {args.port}...")
    try:
        with serial.Serial(args.port, args.baud, timeout=5.0) as ser:
            ser.write(frame)
            ser.flush()
            print("Image sent! Waiting for prediction...")

            response = ser.readline().decode("ascii", errors="replace").strip()
            print(f"\nResponse from Board: {response}\n")
    except Exception as e:
        print(f"UART communication error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
