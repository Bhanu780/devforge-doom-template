"""Generate the held-out side-room keycard map for local tests."""

import argparse
import json
from pathlib import Path

from doom_env.wad import Lump, write_wad

# Spawn faces east. The player must leave the corridor to collect the key.
LAYOUT = (
    "##K####",
    "##.####",
    "#P.+.E#",
    "#######",
)
BENT_LAYOUT = (
    "##########",
    "###..K####",
    "###.######",
    "###.######",
    "#P...+..E#",
    "##########",
)
CELL = 128
# Tile codes. "K" / "+" are the original red key / red locked door.
KEY_THINGS = {"K": 13, "R": 13, "B": 5, "Y": 6}  # red, red, blue, yellow keycards
DOOR_LOCKS = {"+": 1, "r": 1, "b": 2, "y": 3, "D": 0}  # ZDoom lock numbers; 0 = plain door
MONSTER_THINGS = {"Z": 3004, "S": 9, "I": 3001, "G": 9}  # zombieman, shotgun guy, imp (all face west)
DOORS = set(DOOR_LOCKS)
THING_TYPES = {"P": 1, "W": 2001, "A": 2008, "H": 2012, **KEY_THINGS, **MONSTER_THINGS}


def sector_index(layout) -> dict[tuple[int, int], int]:
    """(x, y) tile -> sector number, in the order write_keycard_map creates sectors."""
    tiles = [(x, y) for y, row in enumerate(layout) for x, tile in enumerate(row) if tile != "#"]
    return {point: index for index, point in enumerate(tiles)}


def cell_center(point: tuple[int, int]) -> tuple[float, float]:
    x, y = point
    return (x + 0.5) * CELL, -(y + 0.5) * CELL


def write_keycard_map(path: Path, layout=LAYOUT, spawn_angle: int = 0,
                      spawn_offset: tuple[int, int] = (0, 0), *, monster_angle: int = 180) -> None:
    blocks = ['namespace = "zdoom";']
    vertices = {}

    def add(kind, **fields):
        blocks.append(kind + " { " + " ".join(f"{key} = {json.dumps(value)};" for key, value in fields.items()) + " }")

    def vertex(point):
        if point not in vertices:
            vertices[point] = len(vertices)
            add("vertex", x=point[0] * CELL, y=-point[1] * CELL)
        return vertices[point]

    tiles = {(x, y): tile for y, row in enumerate(layout) for x, tile in enumerate(row) if tile != "#"}
    sectors = {point: index for index, point in enumerate(tiles)}
    sides = 0

    def side(point, texture="-"):
        nonlocal sides
        index = sides
        sides += 1
        add("sidedef", sector=sectors[point], texturemiddle=texture, texturetop="BIGDOOR2")
        return index

    for (x, y), tile in tiles.items():
        add("sector", heightfloor=0, heightceiling=0 if tile in DOORS else 128,
            texturefloor="FLOOR4_8", textureceiling="CEIL3_5", lightlevel=192)
        if tile in THING_TYPES:
            px, py = cell_center((x, y))
            if tile == "P":
                px += spawn_offset[0]
                py += spawn_offset[1]
                kind, angle = 1, spawn_angle
            else:
                kind = THING_TYPES[tile]
                angle = monster_angle if tile in MONSTER_THINGS else 0
            add("thing", x=px, y=py, angle=angle, type=kind,
                skill1=True, skill2=True, skill3=True, skill4=True, skill5=True, single=True)

    for point, tile in tiles.items():
        x, y = point
        edges = (
            ((x, y - 1), (x, y), (x + 1, y)),
            ((x + 1, y), (x + 1, y), (x + 1, y + 1)),
            ((x, y + 1), (x + 1, y + 1), (x, y + 1)),
            ((x - 1, y), (x, y + 1), (x, y)),
        )
        for neighbor, start, end in edges:
            other = tiles.get(neighbor)
            if other is not None and (tile in DOORS or (other not in DOORS and point > neighbor)):
                continue
            exit_switch = tile == "E" and neighbor == (x + 1, y)
            fields = dict(v1=vertex(start), v2=vertex(end))
            if other is None:
                fields.update(sidefront=side(point, "SW1EXIT" if exit_switch else "STARTAN3"), blocking=True)
            else:
                fields.update(sidefront=side(point), sideback=side(neighbor), twosided=True)
            if other in DOORS:
                lock = DOOR_LOCKS[other]
                if lock:  # Door_LockedRaise(tag 0 = back sector, speed, delay, lock)
                    fields.update(special=13, arg1=32, arg2=150, arg3=lock, playeruse=True, repeatspecial=True)
                else:  # Door_Open(tag 0 = back sector, speed): stays open, so routes that come back through it work
                    fields.update(special=11, arg1=32, playeruse=True)
            if exit_switch:
                fields.update(special=243, playeruse=True)
            add("linedef", **fields)

    path.parent.mkdir(parents=True, exist_ok=True)
    write_wad(path, [Lump("MAP01", b""), Lump("TEXTMAP", "\n".join(blocks).encode()), Lump("ENDMAP", b"")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layout", choices=("straight", "bent"), default="straight")
    args = parser.parse_args()
    write_keycard_map(args.output, BENT_LAYOUT if args.layout == "bent" else LAYOUT)
