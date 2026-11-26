"""Procedurally generated, harder progression levels (training only).

    python training/hard_levels.py training/runs/hard_maps --count 36          # write WADs for --practice-maps
    python training/hard_levels.py training/runs/hard_maps --count 36 --check  # also play each one with the
                                                                               # demonstration controller

Every level is a corridor with side branches built from keycard_map tiles. Difficulty tiers:

1. Decoys: 2-3 dead-end branches, one of them hides the red key; locked door before the exit.
2. Plus a key room behind a plain door, longer/bent branches, a guard near the exit, more decoys.
3. Two-key chain (blue door, then red door) with decoys in both halves and ambushes in dead ends.

The demonstration route is found automatically by breadth-first search over (tile, keys held).
That search is training-time scaffolding for imitate_keycard.py; the evaluated agent never sees
it (docs/RULES.md section 2).
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from keycard_map import CELL, DOOR_LOCKS, DOORS, KEY_THINGS, MONSTER_THINGS, cell_center, sector_index, write_keycard_map

ROWS = 13
MAIN = ROWS // 2
KEY_OPENS = {"K": "+r", "R": "+r", "B": "b", "Y": "y"}  # key tile -> door tiles it opens
SPAWN_ANGLES = (0, 90, 180, 270)


# ------------------------------------------------------------------------------ generation
def _branch(grid, rng, column, direction, depth, bend):
    """Carve a dead-end branch off the main corridor; returns its end tile (x, y)."""
    for offset in range(1, depth + 1):
        grid[MAIN + direction * offset][column] = "."
    end_y = MAIN + direction * depth
    end_x = column + bend
    for x in range(min(column, end_x), max(column, end_x) + 1):
        grid[end_y][x] = "."
    return end_x, end_y


def _segment(grid, rng, start, n_branches, key, tier, monsters):
    """Branches in columns [start, start + 3 * n_branches); one holds `key`. Returns next free column."""
    columns = [start + 1 + 3 * i for i in range(n_branches)]
    key_branch = int(rng.integers(n_branches))
    for i, column in enumerate(columns):
        direction = int(rng.choice((-1, 1)))
        depth = int(rng.integers(1, 4 if tier == 1 else 6))
        bend = int(rng.integers(-1, 2)) if depth >= 2 else 0
        end_x, end_y = _branch(grid, rng, column, direction, depth, bend)
        if i == key_branch:
            grid[end_y][end_x] = key
            if tier >= 2 and depth >= 3 and rng.random() < 0.6:
                grid[MAIN + direction * 2][column] = "D"  # key room behind a plain door
        elif monsters and rng.random() < 0.5:
            grid[end_y][end_x] = str(rng.choice(("Z", "S") if tier < 3 else ("Z", "S", "I")))
    return columns[-1] + 2


def generate(rng: np.random.Generator, tier: int) -> list[str]:
    width = 40
    grid = [["#"] * width for _ in range(ROWS)]
    monsters = tier >= 2
    x = 1
    if tier < 3:
        end = _segment(grid, rng, x, int(rng.integers(2, 4)) + (tier - 1), "R", tier, monsters)
        doors = [(end, "r")]
    else:
        mid = _segment(grid, rng, x, int(rng.integers(2, 4)), "B", tier, monsters)
        end = _segment(grid, rng, mid + 1, int(rng.integers(2, 4)), "R", tier, monsters)
        doors = [(mid, "b"), (end, "r")]
    exit_x = end + 2
    for column in range(1, exit_x + 1):
        if grid[MAIN][column] == "#":
            grid[MAIN][column] = "."
    for column, tile in doors:
        grid[MAIN][column] = tile
    grid[MAIN][1] = "P"
    grid[MAIN][exit_x] = "E"
    if monsters and rng.random() < 0.7:
        grid[MAIN][exit_x - 1] = "Z" if tier < 3 else "S"  # guard between the last door and the exit
    # crop unused columns, keep the wall east of the exit (the exit switch sits on it)
    return ["".join(row[: exit_x + 2]) for row in grid]


def valid(layout) -> bool:
    tiles = {(x, y): t for y, row in enumerate(layout) for x, t in enumerate(row) if t != "#"}
    for (x, y), t in tiles.items():
        if t in DOORS:  # a door must join exactly two opposite tiles
            n = [(x + dx, y + dy) in tiles for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            if n not in ([True, True, False, False], [False, False, True, True]):
                return False
        if t == "E" and (x + 1, y) in tiles:
            return False
    return route_tiles(layout) is not None


# ------------------------------------------------------------------------------ demonstration route
def route_tiles(layout):
    """Shortest tile path from P through the needed keys to E: BFS over (tile, keys held)."""
    tiles = {(x, y): t for y, row in enumerate(layout) for x, t in enumerate(row) if t != "#"}
    start = next(p for p, t in tiles.items() if t == "P")
    key_ids = sorted({t for t in tiles.values() if t in KEY_THINGS})

    def opened(tile, held):
        if tile not in DOORS or DOOR_LOCKS[tile] == 0:
            return True
        return any(tile in KEY_OPENS[k] for k in held)

    s0 = (start, frozenset())
    prev = {s0: None}
    queue = deque([s0])
    while queue:
        state = queue.popleft()
        (x, y), held = state
        if tiles[(x, y)] == "E":
            path = []
            while state is not None:
                path.append(state[0])
                state = prev[state]
            return path[::-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            t = tiles.get(nxt)
            if t is None or not opened(t, held):
                continue
            nheld = held | {t} if t in key_ids else held
            ns = (nxt, frozenset(nheld))
            if ns not in prev:
                prev[ns] = state
                queue.append(ns)
    return None


def route_waypoints(layout):
    """Waypoints in imitate_keycard's format: ((x, y), purpose[, door sector])."""
    path = route_tiles(layout)
    tiles = {(x, y): t for y, row in enumerate(layout) for x, t in enumerate(row) if t != "#"}
    sectors = sector_index(layout)
    out = []
    for i, point in enumerate(path[1:], 1):
        tile = tiles[point]
        before = path[i - 1]
        if tile in DOORS and tiles[before] not in DOORS:
            bx, by = cell_center(before)
            px, py = cell_center(point)
            out.append(((bx, by), "walk"))
            out.append((((bx + px) / 2, (by + py) / 2), "door", sectors[point]))
        nxt = path[i + 1] if i + 1 < len(path) else None
        turning = nxt is not None and (nxt[0] - point[0], nxt[1] - point[1]) != (point[0] - before[0], point[1] - before[1])
        if turning or tile in KEY_THINGS or nxt is None or tile in DOORS:
            out.append((cell_center(point), "walk"))
    ex, ey = cell_center(path[-1])
    out.append((((path[-1][0] + 1) * CELL, ey), "exit"))
    return tuple(out)


def build_hard_map(directory: Path, index: int, tier: int, seed: int = 0):
    """Write hard_t{tier}_{index}.wad; returns (path, route, layout)."""
    rng = np.random.default_rng([seed, tier, index])
    for _ in range(200):
        layout = generate(rng, tier)
        if valid(layout):
            break
    else:
        raise RuntimeError("could not generate a valid layout")
    angle = int(rng.choice(SPAWN_ANGLES))
    offset = (int(rng.integers(-20, 21)), int(rng.integers(-20, 21)))
    path = directory / f"hard_t{tier}_{index:02d}.wad"
    write_keycard_map(path, layout, angle, offset)
    return path, route_waypoints(layout), layout


def build_hard_set(directory: Path, count: int, seed: int = 0):
    """count levels split evenly across tiers 1-3."""
    return [build_hard_map(directory, i, 1 + i % 3, seed) for i in range(count)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", type=Path)
    ap.add_argument("--count", type=int, default=36)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="play every level with the demonstration controller")
    ap.add_argument("--show", action="store_true", help="print the layouts")
    args = ap.parse_args()
    levels = build_hard_set(args.output, args.count, args.seed)
    print(f"wrote {len(levels)} levels to {args.output}")
    if args.show:
        for path, _, layout in levels:
            print(path.name, *layout, sep="\n  ")
    if args.check:
        from imitate_keycard import collect_episode

        passed = 0
        for path, route, layout in levels:
            ok = collect_episode(path, route, args.seed, use_rgb=False, ticks=4200) is not None
            passed += ok
            print(f"  {path.name}: {'PASS' if ok else 'FAIL'}", flush=True)
        print(f"{passed}/{len(levels)} levels completed by the demonstration controller")


if __name__ == "__main__":
    main()
