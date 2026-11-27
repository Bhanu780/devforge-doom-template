"""Geodesic distance-to-exit fields for reward shaping (training only; never seen by the agent).

The shaper turns this into a potential  Phi(s) = -k * d(s)  where d is the walking distance from the
player to the exit, and pays  Phi(s') - Phi(s)  per step: moving toward the exit pays, moving away
costs the same, standing still pays nothing, and the total over an episode telescopes to
Phi(end) - Phi(start), so loops cannot be farmed.

* Walls: one-sided and "impassable" linedefs, rasterised onto a CELL-unit grid. Doors and other
  two-sided lines are open (the agent can open doors; floor heights are ignored).
* Locked doors: the boundary lines of a locked door sector are walls unless the held keys open it.
* Keys: when the exit is unreachable with the keys held, d = distance to a missing key + that key's
  own distance-to-exit with the key added (recursively). At the key's position both values agree,
  so picking the key up does not make the potential jump.
* Distances come from a 4-connected BFS (Manhattan inside open rooms: good enough for shaping).

Not modelled: teleporters, lifts, one-way drops and switch-opened bars. The shaper clips each
step's shaping so the rare mismatch (e.g. a teleport) cannot dominate the return.
"""

from __future__ import annotations

import math
import struct
from collections import OrderedDict

import numpy as np

from doom_env.mapinfo import parse_udmf, udmf_namespace, DOOM_NAMESPACES
from doom_env.wad import map_lumps, read_wad

CELL = 16.0
MAX_STEP_UP = 24.0  # vanilla walking limit, in map units
GRID_OFFSET = 5.3  # map units; see NavField.__init__
DOOM_EXITS = {11, 51, 52, 124, 197, 198}  # S1/W1/G1 normal and secret exits
HEXEN_EXITS = {74, 75, 243, 244}  # Teleport_NewMap, Teleport_EndGame, Exit_Normal, Exit_Secret
KEY_COLOR = {"BlueCard": 0, "YellowCard": 1, "RedCard": 2, "BlueSkull": 0, "YellowSkull": 1, "RedSkull": 2}
# vitals / privileged "keys" order: BlueCard, YellowCard, RedCard, BlueSkull, YellowSkull, RedSkull
COLOR_OF_SLOT = (0, 1, 2, 0, 1, 2)
# doom_env lock numbers (1 red, 2 blue, 3 yellow, 100 any, 101 all) plus raw ZDoom numbers seen in
# Hexen/UDMF maps (4-6 skulls, 129-134 "card or skull").
_RED, _BLUE, _YELLOW = 2, 0, 1
LOCK_COLORS = {1: {_RED}, 2: {_BLUE}, 3: {_YELLOW}, 4: {_RED}, 5: {_BLUE}, 6: {_YELLOW},
               129: {_RED}, 130: {_BLUE}, 131: {_YELLOW}, 132: {_RED}, 133: {_BLUE}, 134: {_YELLOW}}


def lock_open(lock: int, colors: frozenset) -> bool:
    if lock <= 0:
        return True
    if lock in LOCK_COLORS:
        return bool(LOCK_COLORS[lock] & colors)
    if lock in (101, 229):
        return len(colors) == 3
    return len(colors) > 0  # 100 "any key" and anything unknown


def exit_segments(wad: str, map_name: str) -> np.ndarray:
    """(K, 4) [x1, y1, x2, y2] of every exit linedef."""
    lumps = map_lumps(read_wad(wad), map_name)
    out = []
    if "TEXTMAP" in lumps:
        text = lumps["TEXTMAP"].decode("utf-8", "replace")
        exits = DOOM_EXITS if udmf_namespace(text) in DOOM_NAMESPACES else HEXEN_EXITS
        b = parse_udmf(text)
        verts = [(float(v.get("x", 0.0)), float(v.get("y", 0.0))) for v in b.get("vertex", [])]
        for line in b.get("linedef", []):
            try:
                if int(line.get("special", 0) or 0) in exits:
                    out.append(verts[int(line["v1"])] + verts[int(line["v2"])])
            except (KeyError, IndexError, ValueError, TypeError):
                continue
    else:
        hexen = "BEHAVIOR" in lumps
        vd = lumps.get("VERTEXES", b"")
        verts = [struct.unpack_from("<hh", vd, i * 4) for i in range(len(vd) // 4)]
        data = lumps.get("LINEDEFS", b"")
        size = 16 if hexen else 14
        for i in range(len(data) // size):
            v1, v2 = struct.unpack_from("<HH", data, i * size)
            special = data[i * size + 6] if hexen else struct.unpack_from("<H", data, i * size + 6)[0]
            if special in (HEXEN_EXITS if hexen else DOOM_EXITS) and v1 < len(verts) and v2 < len(verts):
                out.append(tuple(map(float, verts[v1] + verts[v2])))
    return np.array(out, np.float64).reshape(-1, 4)


def supports_planar_shaping(mapinfo, floor: np.ndarray | None) -> bool:
    """Reject planar shaping if it could route the player up a steep ledge."""
    lines = mapinfo.lines
    if lines is None or floor is None:
        return False
    two_sided = (lines.front >= 0) & (lines.front < len(floor)) & (lines.back >= 0) & (lines.back < len(floor))
    openings = two_sided & ~lines.impassable
    steps = floor[lines.front[openings]] - floor[lines.back[openings]]
    return bool(np.isfinite(steps).all() and (np.abs(steps) <= MAX_STEP_UP).all())


class NavField:
    def __init__(self, mapinfo, exits: np.ndarray):
        g = mapinfo.lines
        self.exits = exits
        pts = np.concatenate([g.p1, g.p2])
        # The fractional offset keeps integer map coordinates off cell boundaries: a wall lying exactly
        # on a boundary marks cells on one side only, and an L-corner of two such walls can leave a
        # one-cell gap that lets the search leak outside the map and back in.
        self.x0, self.y0 = pts.min(0) - 4 * CELL - GRID_OFFSET
        x1, y1 = pts.max(0) + 4 * CELL
        self.W = int(math.ceil((x1 - self.x0) / CELL)) + 1
        self.H = int(math.ceil((y1 - self.y0) / CELL)) + 1
        seg = np.concatenate([g.p1, g.p2], 1)
        wall = (g.back < 0) | g.impassable
        self.base = np.zeros((self.H, self.W), bool)
        self._mark(self.base, seg[wall])
        # locked door sector -> its boundary segments
        locked = {s: l for s, l in mapinfo.door_sectors.items() if l > 0}
        self.locks = []  # (lock, segments)
        if locked:
            for sec, segs in mapinfo.sector_lines().items():
                if sec in locked:
                    self.locks.append((locked[sec], segs))
        self.exit_cells = self._cells_on(exits)
        self.keys: list[tuple[int, int, int]] = []  # (color, row, col), set per episode
        self._blocked: dict = {}
        self._fields: dict = {}
        self._via_key: dict = {}
        self.inside: np.ndarray | None = None  # flat bool: cells reachable from the spawn, all doors open

    def set_start(self, x: float, y: float) -> None:
        """Restrict every search to the level's interior: the cells the player can reach from the
        spawn with all doors open. Without it, the search spreads from an exit switch (which sits on
        an outer wall) into the void around the map."""
        if self.inside is not None:
            return
        free = ~self.blocked(ALL_COLORS).ravel()
        r, c = self.cell(x, y)
        seeds = np.array([(r + dr) * self.W + c + dc for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                          if 0 <= r + dr < self.H and 0 <= c + dc < self.W])
        seeds = seeds[free[seeds]]
        if seeds.size == 0:
            return
        self.inside = np.isfinite(self._flood(free, seeds))
        self._fields, self._via_key = {}, {}

    # ---------------------------------------------------------------- grid helpers
    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int((y - self.y0) // CELL), int((x - self.x0) // CELL)

    def _samples(self, segs: np.ndarray):
        if len(segs) == 0:
            return np.zeros(0, int), np.zeros(0, int)
        p, q = segs[:, :2], segs[:, 2:]
        n = np.maximum(2, np.ceil(np.linalg.norm(q - p, axis=1) / (CELL / 3)).astype(int) + 1)
        idx = np.repeat(np.arange(len(segs)), n)
        t = np.concatenate([np.linspace(0, 1, k) for k in n])
        xy = p[idx] + (q - p)[idx] * t[:, None]
        r = ((xy[:, 1] - self.y0) // CELL).astype(int).clip(0, self.H - 1)
        c = ((xy[:, 0] - self.x0) // CELL).astype(int).clip(0, self.W - 1)
        return r, c

    def _mark(self, grid, segs):
        r, c = self._samples(segs)
        grid[r, c] = True

    def _cells_on(self, segs):
        r, c = self._samples(segs)
        return np.unique(r * self.W + c)

    def blocked(self, colors: frozenset) -> np.ndarray:
        if colors not in self._blocked:
            b = self.base.copy()
            for lock, segs in self.locks:
                if not lock_open(lock, colors):
                    self._mark(b, segs)
            b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = True  # keeps flat-index neighbours from wrapping
            self._blocked[colors] = b
        return self._blocked[colors]

    def _bfs(self, colors: frozenset, sources: np.ndarray) -> np.ndarray:
        free = ~self.blocked(colors).ravel()
        if self.inside is not None:
            free &= self.inside
        return self._flood(free, sources).reshape(self.H, self.W)

    def _flood(self, free: np.ndarray, sources: np.ndarray) -> np.ndarray:
        """Flat BFS distances (map units) from `sources` over `free` cells. Sources may be wall cells
        (exit switches sit on walls); the search only expands into free cells."""
        dist = np.full(self.H * self.W, np.inf, np.float32)
        dist[sources] = 0.0
        frontier, d = sources, 0
        steps = np.array([1, -1, self.W, -self.W])
        while frontier.size:
            d += 1
            nb = (frontier[:, None] + steps[None, :]).ravel()
            nb = nb[(nb >= 0) & (nb < dist.size)]
            nb = np.unique(nb[free[nb] & np.isinf(dist[nb])])
            dist[nb] = d * CELL
            frontier = nb
        return dist

    def field(self, colors: frozenset, source) -> np.ndarray:
        """source: "exit" or a key index into self.keys."""
        k = (colors, source)
        if k not in self._fields:
            if source == "exit":
                src = self.exit_cells
            else:
                _, r, c = self.keys[source]
                src = np.array([r * self.W + c])
            self._fields[k] = self._bfs(colors, src)
        return self._fields[k]

    def _lookup(self, colors: frozenset, f: np.ndarray, r: int, c: int) -> float:
        """Distance at (r, c). Only if that cell is a wall cell (the player brushing a wall) does it
        fall back to the nearest neighbour; a free but unreached cell is genuinely unreachable."""
        H, W = f.shape
        if not (0 <= r < H and 0 <= c < W):
            return math.inf
        if np.isfinite(f[r, c]) or not self.blocked(colors)[r, c]:
            return float(f[r, c])
        best = math.inf
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and np.isfinite(f[rr, cc]):
                    best = min(best, float(f[rr, cc]) + CELL * (abs(dr) + abs(dc)))
        return best

    # ---------------------------------------------------------------- public
    def set_keys(self, keys: list[tuple[int, float, float]]) -> None:
        """Key items present at episode start: (color, x, y). Same map -> same keys, so cached
        fields stay valid; a different layout resets the key-dependent caches."""
        new = [(col, *self.cell(x, y)) for col, x, y in keys]
        if new != self.keys:
            self.keys = new
            self._fields = {k: v for k, v in self._fields.items() if k[1] == "exit"}
            self._via_key = {}

    def distance(self, x: float, y: float, colors: frozenset, _depth: int = 0) -> float:
        """Walking distance to the exit, through missing keys when the exit is locked away."""
        r, c = self.cell(x, y)
        d = self._lookup(colors, self.field(colors, "exit"), r, c)
        if math.isfinite(d) or _depth > 3:
            return d
        best = math.inf
        for i, (col, kr, kc) in enumerate(self.keys):
            if col in colors:
                continue
            to_key = self._lookup(colors, self.field(colors, i), r, c)
            if not math.isfinite(to_key):
                continue
            vk = (colors, i)
            if vk not in self._via_key:
                kx, ky = self.x0 + (kc + 0.5) * CELL, self.y0 + (kr + 0.5) * CELL
                self._via_key[vk] = self.distance(kx, ky, colors | {col}, _depth + 1)
            best = min(best, to_key + self._via_key[vk])
        return best


_CACHE: OrderedDict = OrderedDict()
_CACHE_SIZE = 4


def for_env(env) -> NavField | None:
    """NavField for env's map (cached per worker), or None if the map has no exit linedef or no
    usable geometry. Call at reset: it also reads this episode's key positions."""
    key = (str(env.wad), env.map_name)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        nav = _CACHE[key]
    else:
        nav = None
        try:
            ex = exit_segments(env.wad, env.map_name)
            if len(ex) and env.mapinfo.lines is not None and len(env.mapinfo.lines.p1):
                nav = NavField(env.mapinfo, ex)
        except Exception as e:  # never let shaping break training
            print(f"[navfield] {env.map_name}: {e}", flush=True)
        _CACHE[key] = nav
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    if nav is not None:
        s = env.privileged_state()
        nav.set_start(s["x"], s["y"])
        keys = []
        state = env._game.get_state()
        for o in (state.objects if state is not None and state.objects is not None else []):
            if o.name in KEY_COLOR:
                keys.append((KEY_COLOR[o.name], float(o.position_x), float(o.position_y)))
        nav.set_keys(sorted(keys))
    return nav


ALL_COLORS = frozenset((0, 1, 2))


def colors_held(keys: np.ndarray) -> frozenset:
    return frozenset(COLOR_OF_SLOT[i] for i in np.flatnonzero(np.asarray(keys) > 0.5))
