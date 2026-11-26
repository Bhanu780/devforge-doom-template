"""Generate side-room practice maps and their training-only waypoint routes."""

from pathlib import Path

from keycard_map import CELL, cell_center, write_keycard_map

# Branch column, branch direction (row offset), branch depth, and door column.
PRACTICE_LAYOUTS = (
    (3, -1, 1, 5),
    (3, 1, 1, 5),
    (4, -1, 2, 5),
    (4, 1, 2, 6),
    (5, -1, 3, 7),
    (5, 1, 3, 7),
)


def build_practice_map(directory: Path, layout_index: int, variation: int):
    branch, direction, depth, door = PRACTICE_LAYOUTS[layout_index]
    main_row = 4
    exit_column = door + 2
    grid = [["#"] * (exit_column + 2) for _ in range(9)]
    for x in range(1, exit_column + 1):
        grid[main_row][x] = "."
    for offset in range(1, depth + 1):
        grid[main_row + direction * offset][branch] = "."
    key_row = main_row + direction * depth
    grid[key_row][branch] = "K"
    grid[main_row][1] = "P"
    grid[main_row][door] = "+"
    grid[main_row][exit_column] = "E"

    spawn_angles = (0, 90, 270, 180)
    spawn_offsets = ((0, 0), (15, -12), (-16, 15), (20, 4))
    path = directory / f"side_room_{layout_index}_{variation}.wad"
    write_keycard_map(path, grid, spawn_angles[variation], spawn_offsets[variation])
    junction = cell_center((branch, main_row))
    key = cell_center((branch, key_row))
    door_x, corridor_y = cell_center((door, main_row))
    exit_wall = (exit_column + 1) * CELL
    route = (
        (junction, "walk"),
        (key, "walk"),
        (junction, "walk"),
        ((door_x - CELL / 2 - 26, corridor_y), "walk"),
        ((door_x - CELL / 2, corridor_y), "door"),
        ((exit_wall - 26, corridor_y), "walk"),
        ((exit_wall, corridor_y), "exit"),
    )
    return path, route
