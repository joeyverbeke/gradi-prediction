#!/usr/bin/env python3
"""
Convert a little-endian 16 kHz mono PCM .raw file into a PROGMEM header.

Usage:
    python gen_prompt_header.py --input tts-prompts/EN-0_Politics.raw \
        --output firmware/esp-daf/en_prompt_politics.h \
        --symbol en_prompt_politics
"""

import argparse
import pathlib
import sys
from array import array
from textwrap import wrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Arduino prompt header from PCM raw audio")
    parser.add_argument("--input", required=True, type=pathlib.Path, help="Path to PCM .raw input (16-bit LE, mono)")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Header file to write")
    parser.add_argument("--symbol", required=True, help="Base symbol name (e.g., en_prompt_politics)")
    parser.add_argument("--samples-per-line", type=int, default=10, help="How many samples per line in the array")
    return parser.parse_args()


def load_pcm_samples(path: pathlib.Path) -> array:
    data = path.read_bytes()
    if len(data) % 2:
        raise ValueError(f"Input {path} length must be divisible by 2 bytes, got {len(data)}")
    samples = array("h")
    samples.frombytes(data)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def emit_header(output: pathlib.Path, symbol: str, samples: array, samples_per_line: int) -> None:
    guard = symbol.upper() + "_H"
    lines = []
    lines.append("#pragma once")
    lines.append("#include <stddef.h>")
    lines.append("#include <stdint.h>")
    lines.append("#include <pgmspace.h>")
    lines.append("")
    lines.append(f"static const int16_t {symbol}[] PROGMEM = {{")
    for chunk_start in range(0, len(samples), samples_per_line):
        chunk = samples[chunk_start:chunk_start + samples_per_line]
        line = ", ".join(str(value) for value in chunk)
        lines.append(f"    {line},")
    if len(samples):
        lines[-1] = lines[-1].rstrip(",")
    lines.append("};")
    lines.append(f"static const size_t {symbol}_length = {len(samples)};")
    lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    samples = load_pcm_samples(args.input)
    emit_header(args.output, args.symbol, samples, args.samples_per_line)


if __name__ == "__main__":
    main()
