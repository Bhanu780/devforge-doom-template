"""Training / validation map pool. Everything here ships with vizdoom or this repo: no downloads.

Pool entries: (wad path, map name, group). Groups drive the curriculum weights in sample().
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import vizdoom

from doom_env.wad import list_maps, read_wad

ROOT = Path(__file__).resolve().parents[1]
VZ = Path(os.path.dirname(vizdoom.__file__))
FREEDOOM2 = VZ / "freedoom2.wad"
SCENARIOS = VZ / "scenarios"

# Held out for the benchmark only (never trained on).
HELDOUT_FREEDOOM = {"MAP04", "MAP09", "MAP14", "MAP19", "MAP24", "MAP29"}
HELDOUT_SCENARIOS = {"defend_the_line"}
SCENARIO_NAMES = ["defend_the_center", "defend_the_line", "deadly_corridor", "health_gathering",
                  "health_gathering_supreme", "my_way_home", "take_cover", "basic", "rocket_basic"]

# Extra maps you add (any Doom/UDMF WAD): drop them in training/extra_maps/; all their maps are used.
EXTRA_DIR = ROOT / "training" / "extra_maps"

# Generated hard progression levels (training/hard_levels.py). Train and validation sets use
# different seeds, so the validation layouts are never trained on. Written on first use by
# ensure_hard_levels() (called by train.py and bench.py before any worker starts).
HARD_DIR = ROOT / "training" / "runs" / "hard_maps"
HARD_VAL_DIR = ROOT / "training" / "runs" / "hard_maps_val"
HARD_TRAIN_COUNT, HARD_TRAIN_SEED = 60, 0
HARD_VAL_COUNT, HARD_VAL_SEED = 9, 1000


def ensure_hard_levels() -> None:
    from hard_levels import build_hard_set

    for directory, count, seed in ((HARD_DIR, HARD_TRAIN_COUNT, HARD_TRAIN_SEED), (HARD_VAL_DIR, HARD_VAL_COUNT, HARD_VAL_SEED)):
        if len(list(directory.glob("hard_*.wad"))) < count:
            build_hard_set(directory, count, seed)
            print(f"generated {count} hard levels in {directory}", flush=True)


def _hard_pool(directory: Path) -> list[tuple[str, str, str]]:
    return [(str(wad), "MAP01", "hard") for wad in sorted(directory.glob("hard_*.wad"))]


def _maps_in(wad: Path) -> list[str]:
    try:
        return list_maps(read_wad(wad))
    except Exception:
        return []


def build_pool(split: str = "train", practice_maps: Path | None = None) -> list[tuple[str, str, str]]:
    if split == "progression":
        return [entry for entry in build_pool("train", practice_maps) if entry[2] in ("progression", "hard")]
    if split == "hard":  # held-out generated levels only
        return _hard_pool(HARD_VAL_DIR)
    pool: list[tuple[str, str, str]] = []
    for m in _maps_in(FREEDOOM2):
        held = m in HELDOUT_FREEDOOM
        if (split == "val") == held:
            num = int(m[3:]) if m[3:].isdigit() else 99
            pool.append((str(FREEDOOM2), m, "fd_early" if num <= 8 else "fd"))
    for name in SCENARIO_NAMES:
        wad = SCENARIOS / f"{name}.wad"
        if wad.exists() and (split == "val") == (name in HELDOUT_SCENARIOS):
            for m in _maps_in(wad)[:1]:
                pool.append((str(wad), m, "small"))
    if split == "train":
        for wad in sorted((ROOT / "tests" / "fixtures").glob("*.wad")) + [ROOT / "maps" / "two_rooms.wad"]:
            for m in _maps_in(wad):
                # The hashed public fixture is the red-key / locked-door exit corridor.
                group = "progression" if wad.name in {"two_rooms.wad", "e7ab54fb6834ffa3.wad"} else "small"
                pool.append((str(wad), m, group))
        if practice_maps is not None:
            for wad in sorted(practice_maps.glob("*.wad")):
                pool.extend((str(wad), m, "progression") for m in _maps_in(wad))
        pool.extend(_hard_pool(HARD_DIR))
        if EXTRA_DIR.exists():
            for wad in sorted(EXTRA_DIR.glob("*.wad")):
                for m in _maps_in(wad):
                    pool.append((str(wad), m, "fd"))
    if split == "val":
        pool.extend(_hard_pool(HARD_VAL_DIR))
    return pool


# Curriculum: group weights as a function of training progress (env steps).
def group_weights(env_steps: int) -> dict[str, float]:
    override = os.environ.get("DOOM_GROUPS")  # e.g. DOOM_GROUPS="small=1,fd_early=1"
    if override:
        return {k: float(v) for k, v in (kv.split("=") for kv in override.split(","))}
    if env_steps < 3_000_000:
        return {"progression": 0.15, "hard": 0.15, "small": 0.35, "fd_early": 0.35, "fd": 0.0}
    if env_steps < 8_000_000:
        return {"progression": 0.08, "hard": 0.17, "small": 0.20, "fd_early": 0.30, "fd": 0.25}
    return {"progression": 0.05, "hard": 0.15, "small": 0.15, "fd_early": 0.20, "fd": 0.45}


def sample(pool, env_steps: int, rng: np.random.Generator):
    w = group_weights(env_steps)
    weights = np.array([w.get(g, 0.0) for _, _, g in pool], np.float64)
    # normalise per group so a group's share doesn't depend on how many maps it has
    for g in set(x[2] for x in pool):
        idx = [i for i, x in enumerate(pool) if x[2] == g]
        if weights[idx].sum() > 0:
            weights[idx] /= len(idx)
    if weights.sum() <= 0:
        weights[:] = 1.0
    i = rng.choice(len(pool), p=weights / weights.sum())
    return pool[i]
