"""Watch one episode of the real DoomAgent and explain what it did (training/diagnostics only).

    python training/watch.py --weights training/runs/x/candidate.pt --wad training/runs/hard_maps_val/hard_t3_02.wad
    python training/watch.py --weights ... --wad <freedoom2.wad> --map MAP04 --ticks 4200 --out trace.png

Prints a timeline (keys, doors, kills, damage, death/exit), button usage, where the agent stalled,
and writes a top-down PNG: walls grey, doors yellow, locked doors in their key colour, exit lines
green, keys as dots, monster spawns as orange crosses, the path from blue (start) to red (end), and stalls as white rings.
Ground truth comes from privileged_state(); none of it reaches the agent.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "submission"))

BUTTONS = ("fwd", "back", "strafeL", "strafeR", "turnL", "turnR", "attack", "use", "nextW")
KEY_RGB = {0: (60, 110, 255), 1: (255, 220, 0), 2: (255, 40, 40)}  # blue, yellow, red
LOCK_RGB = {1: KEY_RGB[2], 2: KEY_RGB[0], 3: KEY_RGB[1]}


def write_png(path: Path, img: np.ndarray) -> None:
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].astype(np.uint8).tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


class Canvas:
    def __init__(self, lines_xy: np.ndarray, size: int = 900):
        lo, hi = lines_xy.min(0) - 64, lines_xy.max(0) + 64
        self.lo, self.s = lo, (size - 1) / float(max(hi - lo))
        w, h = (np.ceil((hi - lo) * self.s) + 1).astype(int)
        self.img = np.full((h, w, 3), 18, np.uint8)

    def px(self, x, y):
        c = ((np.asarray(x) - self.lo[0]) * self.s).astype(int)
        r = ((self.lo[1] + self.img.shape[0] / self.s - np.asarray(y)) * self.s).astype(int)  # +y up
        return r.clip(0, self.img.shape[0] - 1), c.clip(0, self.img.shape[1] - 1)

    def segment(self, x1, y1, x2, y2, rgb, width=1):
        n = int(max(2, math.hypot(x2 - x1, y2 - y1) * self.s * 2))
        t = np.linspace(0, 1, n)
        r, c = self.px(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
        for dr in range(-(width // 2), width // 2 + 1):
            for dc in range(-(width // 2), width // 2 + 1):
                self.img[(r + dr).clip(0, self.img.shape[0] - 1), (c + dc).clip(0, self.img.shape[1] - 1)] = rgb

    def dot(self, x, y, rgb, radius=3, ring=False):
        r0, c0 = self.px(x, y)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                d = dr * dr + dc * dc
                if d <= radius * radius and (not ring or d >= (radius - 1.5) ** 2):
                    self.img[min(max(r0 + dr, 0), self.img.shape[0] - 1), min(max(c0 + dc, 0), self.img.shape[1] - 1)] = rgb


def stalls(xy: np.ndarray, window: int = 40, radius: float = 64.0):
    """(start step, length, x, y) of stretches where the agent stayed within `radius` units."""
    out, i, n = [], 0, len(xy)
    while i < n - window:
        if np.max(np.linalg.norm(xy[i:i + window] - xy[i], axis=1)) < radius:
            j = i + window
            while j < n and np.linalg.norm(xy[j] - xy[i]) < radius:
                j += 1
            out.append((i, j - i, *xy[i]))
            i = j
        else:
            i += 1
    return out


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--wad", required=True)
    ap.add_argument("--map", default="MAP01")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ticks", type=int, default=2100)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--out", type=Path, help="PNG path (default: next to the weights)")
    args = ap.parse_args()

    from doom_env import make_env
    from navfield import KEY_COLOR, exit_segments
    from rewards import Shaper
    import agent as agent_mod

    agent_mod.WEIGHTS = Path(args.weights)
    env = make_env(args.wad, args.map, episode_ticks=args.ticks, render_buffers=True)
    agent = agent_mod.DoomAgent(env.observation_space, env.action_space, "cpu")
    agent.rng = np.random.default_rng(args.seed)
    if args.temperature is not None:
        agent.temperature = args.temperature
    obs, _ = env.reset(seed=args.seed)
    agent.reset()
    sh = Shaper()
    sh.reset(env)
    state = env._game.get_state()
    keys = [(KEY_COLOR[o.name], o.position_x, o.position_y) for o in state.objects if o.name in KEY_COLOR]
    monsters = [(o.position_x, o.position_y) for o in state.objects if o.category == "Monster"]

    xy, acts, events = [], [], []
    prev = env.privileged_state()
    step, weapon = 0, int(obs["vitals"][2])
    while True:
        a = agent.act(obs)
        obs, _, te, tu, _ = env.step(a)
        sh.step(env, a, obs, te or tu)
        s = env.privileged_state()
        xy.append((s["x"], s["y"]))
        acts.append(a)
        if np.sum(s["keys"]) > np.sum(prev["keys"]):
            events.append((step, "picked up a key"))
        if s["kills"] > prev["kills"]:
            events.append((step, f"kill #{s['kills']}"))
        if s["damage_taken"] - prev["damage_taken"] >= 10:
            events.append((step, f"took {s['damage_taken'] - prev['damage_taken']:.0f} damage (hp {s['health']:.0f})"))
        slot = int(obs["vitals"][2])
        if step and slot != weapon:
            events.append((step, f"weapon slot {weapon} -> {slot}" + (" (fists)" if slot == 1 else "")))
        weapon = slot
        if te or tu:
            events.append((step, "EXIT REACHED" if s["exit_reached"] else ("DIED" if s["dead"] else "time limit")))
            break
        prev = s
        step += 1
    st = sh.stats
    xy, acts = np.array(xy), np.array(acts)

    print(f"{Path(args.wad).name}:{args.map} seed {args.seed}: {len(acts)} steps, exit {st['exit']} death {st['death']} "
          f"keys {st['keys']} doors {st['doors']} locked {st['locked']} cells {st['cells']} kills {st['kills']} dmg {st['dmg']:.0f}")
    print("timeline:")
    for t, e in events:
        print(f"  step {t:4d}: {e}")
    print("buttons held (% of steps): " + "  ".join(f"{b} {100 * acts[:, i].mean():.0f}" for i, b in enumerate(BUTTONS)))
    moving = np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    print(f"moving >2 units/step: {100 * np.mean(moving > 2):.0f}% of steps; distance walked {moving.sum():.0f} units; "
          f"unique 64-unit cells {len({(int(x // 64), int(y // 64)) for x, y in xy})}")
    st_list = stalls(xy)
    print(f"stalls (>=40 steps within 64 units): {len(st_list)}, {sum(s[1] for s in st_list)} steps in total")
    for t, n, x, y in st_list[:8]:
        print(f"  from step {t} for {n} steps at ({x:.0f}, {y:.0f})")

    g = env.mapinfo.lines
    cv = Canvas(np.concatenate([g.p1, g.p2]))
    door_lock = env.mapinfo.door_sectors
    for i in range(len(g.p1)):
        (x1, y1), (x2, y2) = g.p1[i], g.p2[i]
        locks = [door_lock[s] for s in (int(g.front[i]), int(g.back[i])) if s in door_lock]
        if locks and max(locks) > 0:
            cv.segment(x1, y1, x2, y2, LOCK_RGB.get(max(locks), (255, 0, 255)), 3)
        elif locks:
            cv.segment(x1, y1, x2, y2, (230, 200, 40), 2)
        elif g.back[i] < 0 or g.impassable[i]:
            cv.segment(x1, y1, x2, y2, (150, 150, 150))
        else:
            cv.segment(x1, y1, x2, y2, (55, 55, 55))
    for x1, y1, x2, y2 in exit_segments(args.wad, args.map):
        cv.segment(x1, y1, x2, y2, (40, 230, 80), 4)
    for col, x, y in keys:
        cv.dot(x, y, KEY_RGB[col], 5)
    for x, y in monsters:  # spawn positions, orange crosses
        cv.segment(x - 24, y - 24, x + 24, y + 24, (255, 140, 0), 2)
        cv.segment(x - 24, y + 24, x + 24, y - 24, (255, 140, 0), 2)
    for i in range(1, len(xy)):
        f = i / max(1, len(xy) - 1)
        cv.segment(*xy[i - 1], *xy[i], (int(60 + 195 * f), 80, int(255 - 205 * f)))
    for _, _, x, y in st_list:
        cv.dot(x, y, (255, 255, 255), 7, ring=True)
    cv.dot(*xy[0], (0, 200, 255), 4)
    cv.dot(*xy[-1], (255, 60, 60), 4)
    out = args.out or Path(args.weights).with_name(f"watch_{Path(args.wad).stem}_{args.map}_{args.seed}.png")
    write_png(out, cv.img)
    print(f"trace image: {out}")
    env.close()


if __name__ == "__main__":
    main()
