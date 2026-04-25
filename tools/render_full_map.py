#!/usr/bin/env python3

import argparse
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path


OBJECT_TABLE = 62000
OBJECT_RECORD_SIZE = 20
OBJECT_TABLE_TERMINATOR = 255
TILE_BITMAP_BASE = 0xFA00
TILE_ATTRIBUTE_BASE = 0xFE00
SEA_TILE = 0
CELL_SIZE = 8
WORLD_SIZE = 256
MAP_X_OFFSET = 0x4000
MAP_Y_OFFSET = 0x8000
MAP_Z_OFFSET = 0xC000

SPECTRUM_PALETTE = [
    (0, 0, 0),
    (0, 0, 205),
    (205, 0, 0),
    (205, 0, 205),
    (0, 205, 0),
    (0, 205, 205),
    (205, 205, 0),
    (205, 205, 205),
    (0, 0, 0),
    (0, 0, 255),
    (255, 0, 0),
    (255, 0, 255),
    (0, 255, 0),
    (0, 255, 255),
    (255, 255, 0),
    (255, 255, 255),
]


@dataclass(frozen=True)
class MapChunk:
    left: int
    right: int
    top: int
    bottom: int
    source: int

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1


def _decompress_z80_block(block: bytes) -> bytes:
    output = bytearray()
    index = 0
    while index < len(block):
        if index + 4 <= len(block) and block[index] == 0xED and block[index + 1] == 0xED:
            output.extend([block[index + 3]] * block[index + 2])
            index += 4
            continue
        output.append(block[index])
        index += 1
    return bytes(output)


def load_snapshot_memory(snapshot_path: Path) -> bytes:
    data = snapshot_path.read_bytes()
    if len(data) < 30:
        raise ValueError(f"{snapshot_path} is too short to be a Z80 snapshot")

    header_pc = struct.unpack_from("<H", data, 6)[0]
    memory = bytearray(0x10000)

    if header_pc:
        compressed = bool(data[12] & 0x20)
        ram = _decompress_z80_block(data[30:]) if compressed else data[30:]
        if len(ram) != 0xC000:
            raise ValueError(f"Expected 48K RAM in v1 snapshot, found {len(ram)} bytes")
        memory[MAP_X_OFFSET:] = ram
        return bytes(memory)

    extension_length = struct.unpack_from("<H", data, 30)[0]
    position = 32 + extension_length
    pages: dict[int, bytes] = {}
    while position < len(data):
        compressed_length, page = struct.unpack_from("<HB", data, position)
        position += 3
        block = data[position : position + compressed_length]
        position += compressed_length
        pages[page] = _decompress_z80_block(block)

    page_map = {8: MAP_X_OFFSET, 4: MAP_Y_OFFSET, 5: MAP_Z_OFFSET}
    for page, address in page_map.items():
        block = pages.get(page)
        if block is None:
            raise ValueError(f"Missing RAM page {page} in {snapshot_path}")
        if len(block) != 0x4000:
            raise ValueError(f"Expected 16K page for RAM page {page}, found {len(block)} bytes")
        memory[address : address + 0x4000] = block
    return bytes(memory)


def parse_map_chunks(memory: bytes) -> list[MapChunk]:
    chunks: list[MapChunk] = []
    offset = OBJECT_TABLE
    while offset + OBJECT_RECORD_SIZE <= len(memory):
        record = memory[offset : offset + OBJECT_RECORD_SIZE]
        if record[0] == OBJECT_TABLE_TERMINATOR:
            break
        chunks.append(
            MapChunk(
                left=record[2],
                right=record[3],
                top=record[4],
                bottom=record[5],
                source=record[10] | (record[11] << 8),
            )
        )
        offset += OBJECT_RECORD_SIZE
    if not chunks:
        raise ValueError("No gameplay map chunks were found in the object table")
    return chunks


def build_world_map(memory: bytes, chunks: list[MapChunk]) -> tuple[list[list[int]], tuple[int, int, int, int]]:
    world = [[None for _ in range(WORLD_SIZE)] for _ in range(WORLD_SIZE)]
    min_x = min(chunk.left for chunk in chunks)
    max_x = max(chunk.right for chunk in chunks)
    min_y = min(chunk.top for chunk in chunks)
    max_y = max(chunk.bottom for chunk in chunks)

    for chunk in chunks:
        for y in range(chunk.top, chunk.bottom + 1):
            source_row = chunk.source + (y - chunk.top) * 128
            row = world[y]
            for x in range(chunk.left, chunk.right + 1):
                if row[x] is None:
                    row[x] = memory[source_row + (x - chunk.left)]

    for y in range(WORLD_SIZE):
        for x in range(WORLD_SIZE):
            if world[y][x] is None:
                world[y][x] = SEA_TILE

    return world, (min_x, min_y, max_x, max_y)


def _tile_bitmap(memory: bytes, tile: int) -> bytes:
    tile_index = tile & 0x7F
    start = TILE_BITMAP_BASE + tile_index * CELL_SIZE
    return memory[start : start + CELL_SIZE]


def _tile_attribute(memory: bytes, tile: int) -> int:
    return memory[TILE_ATTRIBUTE_BASE + tile]


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def write_png(path: Path, rgb_rows: list[bytes], width: int, height: int) -> None:
    raw = b"".join(b"\x00" + row for row in rgb_rows)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr)
    png += _chunk(b"IDAT", zlib.compress(raw, level=9))
    png += _chunk(b"IEND", b"")
    path.write_bytes(png)


def render_map(memory: bytes, world: list[list[int]], bounds: tuple[int, int, int, int], scale: int, crop: bool) -> tuple[list[bytes], int, int]:
    min_x, min_y, max_x, max_y = bounds if crop else (0, 0, WORLD_SIZE - 1, WORLD_SIZE - 1)
    width_cells = max_x - min_x + 1
    height_cells = max_y - min_y + 1
    width = width_cells * CELL_SIZE * scale
    height = height_cells * CELL_SIZE * scale
    rows: list[bytes] = []

    for cell_y in range(min_y, max_y + 1):
        scanlines = [bytearray() for _ in range(CELL_SIZE * scale)]
        for cell_x in range(min_x, max_x + 1):
            tile = world[cell_y][cell_x]
            attr = _tile_attribute(memory, tile)
            palette_offset = 8 if attr & 0x40 else 0
            ink = SPECTRUM_PALETTE[palette_offset + (attr & 0x07)]
            paper = SPECTRUM_PALETTE[palette_offset + ((attr >> 3) & 0x07)]
            for bitmap_row_index, bitmap_row in enumerate(_tile_bitmap(memory, tile)):
                pixel_row = bytearray()
                for bit in range(7, -1, -1):
                    colour = ink if bitmap_row & (1 << bit) else paper
                    pixel_row.extend(bytes(colour) * scale)
                for scale_row in range(scale):
                    scanlines[bitmap_row_index * scale + scale_row].extend(pixel_row)
        rows.extend(bytes(scanline) for scanline in scanlines)

    return rows, width, height


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("scale must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Cyclone's reverse engineered full gameplay map to a PNG")
    parser.add_argument("snapshot", type=Path, help="Path to the .z80 snapshot")
    parser.add_argument("output", type=Path, help="Path to the PNG to write")
    parser.add_argument(
        "--full-world",
        action="store_true",
        help="Render the full 256x256 world instead of cropping to the occupied gameplay bounds",
    )
    parser.add_argument("--scale", type=positive_int, default=1, help="Nearest-neighbour scale factor for the output image")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    memory = load_snapshot_memory(args.snapshot)
    chunks = parse_map_chunks(memory)
    world, bounds = build_world_map(memory, chunks)
    rows, width, height = render_map(memory, world, bounds, scale=args.scale, crop=not args.full_world)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_png(args.output, rows, width, height)

    min_x, min_y, max_x, max_y = bounds
    rendered_bounds = f"{min_x},{min_y} -> {max_x},{max_y}" if not args.full_world else "0,0 -> 255,255"
    print(
        f"Rendered {len(chunks)} map chunks to {args.output} "
        f"({width}x{height} pixels, world bounds {rendered_bounds})"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, struct.error) as exc:  # pragma: no cover - CLI surface
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
