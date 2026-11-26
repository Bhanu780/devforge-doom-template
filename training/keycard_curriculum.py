"""Generate side-room practice maps and their training-only waypoint routes."""

import argparse
from pathlib import Path

from keycard_map import CELL, cell_center, write_keycard_map

# Branch column, direction, depth, locked-door column, and horizontal bend length.
PRACTICE_LAYOUTS = (
    (3, -1, 1, 5, 0),
    (3, 1, 1, 5, 0),
    (4, -1, 2, 5, 0),
    (4, 1, 2, 6, 0),
    (5, -1, 3, 7, 0),
    (5, 1, 3, 7, 0),
    (3, -1, 2, 6, 1),
    (4, 1, 2, 6, -2),
    (5, -1, 2, 8, -2),
    (4, 1, 3, 7, 2),
    (6, -1, 3, 8, 2),
    (5, 1, 2, 8, -3),
)


def build_practice_map(directory: Path, layout_index: int, variation: int, *, combat: bool = False):
    branch, direction, depth, door, bend = PRACTICE_LAYOUTS[layout_index]
    main_row = 4
    exit_column = door + 2
    grid = [["#"] * (exit_column + 2) for _ in range(9)]
    for x in range(1, exit_column + 1):
        grid[main_row][x] = "."
    for offset in range(1, depth + 1):
        grid[main_row + direction * offset][branch] = "."
    key_row = main_row + direction * depth
    key_column = branch + bend
    for x in range(min(branch, key_column), max(branch, key_column) + 1):
        grid[key_row][x] = "."
    grid[key_row][key_column] = "K"
    grid[main_row][1] = "P"
    grid[main_row][door] = "+"
    grid[main_row][exit_column] = "E"

    if combat:
        # Vary whether the encounter is on the way to the key or behind the door.
        if variation < 3:
            grid[main_row][branch - 1] = "Z"
        if variation >= 2:
            grid[main_row][door + 1] = "I"

    spawn_angles = (0, 90, 270, 180)
    spawn_offsets = ((0, 0), (15, -12), (-16, 15), (20, 4))
    name = f"combat_{layout_index:02d}_{variation}" if combat else f"side_room_{layout_index}_{variation}"
    path = directory / f"{name}.wad"
    write_keycard_map(path, grid, spawn_angles[variation], spawn_offsets[variation])
    junction = cell_center((branch, main_row))
    key = cell_center((key_column, key_row))
    corner = ((cell_center((branch, key_row)), "walk"),) if bend else ()
    door_x, corridor_y = cell_center((door, main_row))
    exit_wall = (exit_column + 1) * CELL
    route = (
        (junction, "walk"),
        *corner,
        (key, "walk"),
        *corner,
        (junction, "walk"),
        ((door_x - CELL / 2 - 26, corridor_y), "walk"),
        ((door_x - CELL / 2, corridor_y), "door"),
        ((exit_wall - 26, corridor_y), "walk"),
        ((exit_wall, corridor_y), "exit"),
    )
    return path, route


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate keycard practice maps populated with enemies.")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for layout in range(len(PRACTICE_LAYOUTS)):
        for variation in range(4):
            build_practice_map(args.output, layout, variation, combat=True)
