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
CELL = 128


def write_keycard_map(path: Path) -> None:
    blocks = ['namespace = "zdoom";']
    vertices = {}

    def add(kind, **fields):
        blocks.append(kind + " { " + " ".join(f"{key} = {json.dumps(value)};" for key, value in fields.items()) + " }")

    def vertex(point):
        if point not in vertices:
            vertices[point] = len(vertices)
            add("vertex", x=point[0] * CELL, y=-point[1] * CELL)
        return vertices[point]

    tiles = {(x, y): tile for y, row in enumerate(LAYOUT) for x, tile in enumerate(row) if tile != "#"}
    sectors = {point: index for index, point in enumerate(tiles)}
    sides = 0

    def side(point, texture="-"):
        nonlocal sides
        index = sides
        sides += 1
        add("sidedef", sector=sectors[point], texturemiddle=texture, texturetop="BIGDOOR2")
        return index

    for (x, y), tile in tiles.items():
        add("sector", heightfloor=0, heightceiling=0 if tile == "+" else 128,
            texturefloor="FLOOR4_8", textureceiling="CEIL3_5", lightlevel=192)
        if tile in ("P", "K"):
            add("thing", x=(x + 0.5) * CELL, y=-(y + 0.5) * CELL,
                angle=0, type=1 if tile == "P" else 13,
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
            if other is not None and (tile == "+" or (other != "+" and point > neighbor)):
                continue
            exit_switch = tile == "E" and neighbor == (x + 1, y)
            fields = dict(v1=vertex(start), v2=vertex(end))
            if other is None:
                fields.update(sidefront=side(point, "SW1EXIT" if exit_switch else "STARTAN3"), blocking=True)
            else:
                fields.update(sidefront=side(point), sideback=side(neighbor), twosided=True)
            if other == "+":
                fields.update(special=13, arg1=32, arg2=150, arg3=1, playeruse=True, repeatspecial=True)
            if exit_switch:
                fields.update(special=243, playeruse=True)
            add("linedef", **fields)

    path.parent.mkdir(parents=True, exist_ok=True)
    write_wad(path, [Lump("MAP01", b""), Lump("TEXTMAP", "\n".join(blocks).encode()), Lump("ENDMAP", b"")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    write_keycard_map(parser.parse_args().output)
