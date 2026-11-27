"""Build independent combat practice and validation layouts."""

import argparse
from collections import deque
from pathlib import Path

from keycard_map import CELL, cell_center, write_keycard_map

# Walls block sight and create chokepoints. Each exit is on an east-facing wall.
PRACTICE = {
    "open_room": (
        "#########",
        "#H....Z.#",
        "#.......#",
        "#P.W...E#",
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
        "#W...I..H#",
        "##########",
    ),
    "choke": (
        "###########",
        "#....#..I.#",
        "#.H..#....#",
        "#P.W...Z.E#",
        "#....#....#",
        "#....#A...#",
        "###########",
    ),
    "two_lanes": (
        "##########",
        "#..Z.....#",
        "#P..##...#",
        "#W..##..E#",
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
        "#.W.#......#",
        "############",
    ),
    "split_cover": (
        "############",
        "#...Z#.....#",
        "#H...#..I..#",
        "#...##.....#",
        "#P.W......E#",
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
            write_keycard_map(directory / f"{split}_{name}_{variant}.wad", rows, angle, offset, monster_angle=0)


def build_exit_lessons(directory: Path):
    """Generate successful post-combat routes on cleared practice layouts."""
    for name, original in PRACTICE.items():
        for mirror in (False, True):
            rows = original[::-1] if mirror else original
            floor = {(x, y) for y, row in enumerate(rows) for x, tile in enumerate(row) if tile != "#"}
            exit_tile = next((x, y) for x, y in floor if rows[y][x] == "E")
            previous = {exit_tile: None}
            distance = {exit_tile: 0}
            queue = deque([exit_tile])
            while queue:
                x, y = point = queue.popleft()
                for adjacent in ((x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)):
                    if adjacent in floor and adjacent not in previous:
                        previous[adjacent] = point
                        distance[adjacent] = distance[point] + 1
                        queue.append(adjacent)
            starts = sorted((point for point in previous if point != exit_tile), key=lambda p: (distance[p], p))
            for index, spawn in enumerate((starts[len(starts) // 4], starts[len(starts) // 2], starts[-1])):
                layout = [["#" if tile == "#" else "." for tile in row] for row in rows]
                layout[exit_tile[1]][exit_tile[0]] = "E"
                layout[spawn[1]][spawn[0]] = "P"
                path = directory / f"exit_{name}_{int(mirror)}_{index}.wad"
                write_keycard_map(path, layout, spawn_angle=(90, 180, 270)[index])
                points = [spawn]
                while points[-1] != exit_tile:
                    points.append(previous[points[-1]])
                # Keep turn waypoints and the final approach point.
                corners = [middle for a, middle, b in zip(points, points[1:], points[2:])
                           if (middle[0] - a[0], middle[1] - a[1]) != (b[0] - middle[0], b[1] - middle[1])]
                route = [(cell_center(point), "walk") for point in corners]
                exit_x, exit_y = cell_center(exit_tile)
                route.extend([((exit_x, exit_y), "walk"), ((exit_x + CELL / 2, exit_y), "exit")])
                yield path, route


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    args = parser.parse_args()
    build_arenas(args.output, args.split)
