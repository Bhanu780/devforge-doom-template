"""Held-out benchmark that runs the *submission* agent (submission/agent.py) exactly as evaluated.

    python training/bench.py                                  # submission/weights/policy.pt
    python training/bench.py --weights training/runs/main/policy_0020M.pt --procs 6

Prints per-map metrics and a proxy score (the real weights are hidden; exit dominates).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def proxy_score(st) -> float:
    return 100 * st["exit"] + 15 * st["keys"] + 5 * st["doors"] + 10 * st["locked"] + 0.5 * st["cells"] \
        + 2 * st["kills"] - 0.05 * st["dmg"] - 20 * st["death"]


def run_one(job):
    wad, mapname, seed, ticks, weights, temperature = job
    import os
    import warnings

    warnings.filterwarnings("ignore")
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.path.insert(0, str(ROOT / "training"))
    from doom_env import make_env

    from rewards import Shaper

    sys.path.insert(0, str(ROOT / "submission"))
    import agent as agent_mod  # the real evaluated entry point

    agent_mod.WEIGHTS = Path(weights) if weights else ROOT / "submission" / "weights" / "policy.pt"
    env = make_env(wad, mapname, episode_ticks=ticks)
    agent = agent_mod.DoomAgent(env.observation_space, env.action_space, "cpu")
    agent.rng = np.random.default_rng(seed)
    if temperature is not None:
        agent.temperature = temperature
    obs, _ = env.reset(seed=seed)
    agent.reset()
    sh = Shaper()
    sh.reset(env)
    seen = gun = 0  # steps with an enemy in view, and how many of those had a gun selected
    while True:
        a = agent.act(obs)
        if sh._enemy_visible(obs):
            seen += 1
            gun += int(obs["vitals"][2]) >= 2
        obs, _, te, tu, _ = env.step(a)
        sh.step(env, a, obs, te or tu)
        if te or tu:
            break
    env.close()
    st = dict(sh.stats)
    st["fists"] = st.get("fists", 0)  # steps holding fists vs an out-of-reach enemy while ammo remained
    st["gun%"] = 100.0 * gun / seen if seen else 100.0
    st["seed"] = seed
    st["map"] = f"{Path(wad).stem}:{mapname}"
    st["score"] = proxy_score(st)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--wad", type=Path, help="evaluate one local WAD instead of a preset split")
    ap.add_argument("--map", default="MAP01", help="map name for --wad")
    ap.add_argument("--split", choices=("val", "progression", "hard"), default="val")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--temperature", type=float, help="override policy sampling temperature for diagnostics")
    ap.add_argument("--ticks", type=int, default=4200)
    args = ap.parse_args()
    if args.temperature is not None and not (np.isfinite(args.temperature) and args.temperature > 0):
        ap.error("--temperature must be positive and finite")
    sys.path.insert(0, str(ROOT / "training"))
    import maps

    weights = str(Path(args.weights).resolve()) if args.weights else ""
    if not args.wad:
        maps.ensure_hard_levels()
    layouts = [(str(args.wad.resolve()), args.map.upper(), "custom")] if args.wad else maps.build_pool(args.split)
    jobs = [(wad, name, seed, args.ticks, weights, args.temperature)
            for wad, name, _ in layouts for seed in range(args.seeds)]
    with mp.get_context("spawn").Pool(args.procs) as pool:
        res = pool.map(run_one, jobs)
    keys = ("exit", "death", "keys", "doors", "locked", "cells", "kills", "dmg", "prog", "fists", "gun%", "score")
    print(f"{'map':32s} {'seed':>4s} " + " ".join(f"{k:>6s}" for k in keys))
    for r in res:
        print(f"{r['map']:32s} {r['seed']:4d} " + " ".join(f"{r[k]:6.1f}" for k in keys))
    print(f"{'MEAN':37s} " + " ".join(f"{np.mean([r[k] for r in res]):6.1f}" for k in keys))


if __name__ == "__main__":
    main()
