"""Build independent combat practice and validation layouts."""

import argparse
from pathlib import Path

from keycard_map import write_keycard_map

# Walls block sight and create chokepoints. Each exit is on an east-facing wall.
PRACTICE = {
    "open_room": (
        "#########",
        "#H....Z.#",
        "#.......#",
        "#P.S...E#",
        "#.......#",
        "#..I..A.#",
        "#########",
    ),
    "pillar": (
        "##########",
        "#.......Z#",
        "#..###...#",
        "#P.###..E#",
        "#..###...#",
        "#S...I..H#",
        "##########",
    ),
    "choke": (
        "###########",
        "#....#..I.#",
        "#.H..#....#",
        "#P.S...Z.E#",
        "#....#....#",
        "#....#A...#",
        "###########",
    ),
    "two_lanes": (
        "##########",
        "#..Z.....#",
        "#P..##...#",
        "#S..##..E#",
        "#...##...#",
        "#H..I...A#",
        "##########",
    ),
}

VALIDATION = {
    "staggered": (
        "############",
        "#.H.#..I...#",
        "#...#..##..#",
        "#P..Z.....E#",
        "#......##..#",
        "#...#G..A..#",
        "#.S.#......#",
        "############",
    ),
    "split_cover": (
        "############",
        "#...Z#.....#",
        "#H...#..I..#",
        "#...##.....#",
        "#P.S......E#",
        "#....###...#",
        "#..G.....A.#",
        "############",
    ),
}


def build_arenas(directory: Path, split: str) -> None:
    layouts = PRACTICE if split == "train" else VALIDATION
    starts = ((0, (0, 0)), (90, (12, -16)), (270, (-16, 12)), (180, (0, 15)))
    for name, layout in layouts.items():
        for variant, (angle, offset) in enumerate(starts):
            rows = layout[::-1] if variant % 2 else layout
            write_keycard_map(directory / f"{split}_{name}_{variant}.wad", rows, angle, offset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    args = parser.parse_args()
    build_arenas(args.output, args.split)
