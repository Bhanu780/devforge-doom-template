"""Held-out benchmark that runs the *submission* agent (submission/agent.py) exactly as evaluated.

    python training/bench.py                                  # submission/weights/policy.pt
    python training/bench.py --weights training/runs/main/policy_0020M.pt --procs 6

Prints per-map metrics and a proxy score (the real weights are hidden; exit dominates).
"""

from __future__ import annotations

import argparse
from contextlib import closing
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRONT_RAYS = np.array([62, 63, 0, 1, 2])


def proxy_score(st) -> float:
    return 100 * st["exit"] + 15 * st["keys"] + 5 * st["doors"] + 10 * st["locked"] + 0.5 * st["cells"] \
        + 2 * st["kills"] - 0.05 * st["dmg"] - 20 * st["death"]


def run_one(job):
    wad, mapname, seed, ticks, weights, temperature, combat_stats, progress_stats = job
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
    with closing(make_env(wad, mapname, episode_ticks=ticks)) as env:
        agent = agent_mod.DoomAgent(env.observation_space, env.action_space, "cpu")
        agent.rng = np.random.default_rng(seed)
        if temperature is not None:
            agent.temperature = temperature
        obs, _ = env.reset(seed=seed)
        agent.reset()
        sh = Shaper()
        sh.reset(env)
        combat = {"fist": 0, "far_fist": 0, "switch": 0, "loaded": 0}
        seen = gun = 0
        previous_switch = False
        progress = {"near_gate": 0, "gate_use": 0, "last_new": 0, "barrel_atk": 0}
        previous_use = False
        while True:
            action = agent.act(obs)
            if sh._enemy_visible(obs):
                seen += 1
                gun += int(obs["vitals"][2]) >= 2
            if combat_stats:
                entities = obs["visible_entities"]
                alive = entities["mask"].astype(bool)
                enemy = alive & np.isin(entities["entity_type"], (1, 2, 3))
                target = enemy | (alive & (entities["entity_type"] == 10))
                weapon, ammo = obs["vitals"][2:4]
                if weapon == 1:
                    combat["fist"] += 1
                    if action[6] and target.any() and np.all(entities["features"][target, 0] > 128 / 1024):
                        combat["far_fist"] += 1
                pressing_switch = bool(action[8])
                if pressing_switch and not previous_switch:
                    combat["switch"] += 1
                    combat["loaded"] += int(ammo > 0 and enemy.any())
                previous_switch = pressing_switch
            if progress_stats:
                rays = obs["rays"]
                close = rays["distance"][FRONT_RAYS] <= 96 / 1024
                door = np.isin(rays["hit_type"][FRONT_RAYS], (2, 3))
                near_gate = bool(np.any(close & door))
                use = bool(action[7])
                progress["near_gate"] += near_gate
                progress["gate_use"] += bool(near_gate and use and not previous_use)
                previous_use = use
                entities = obs["visible_entities"]
                visible = entities["entity_type"][entities["mask"].astype(bool)]
                progress["barrel_atk"] += bool(action[6] and np.any(visible == 10) and not np.any(np.isin(visible, (1, 2, 3))))
            previous_cells = sh.stats["cells"]
            obs, _, terminated, truncated, _ = env.step(action)
            sh.step(env, action, obs, terminated or truncated)
            if progress_stats and sh.stats["cells"] > previous_cells:
                progress["last_new"] = sh.stats["steps"]
            if terminated or truncated:
                break
        st = dict(sh.stats)
        st["fists"] = st.get("fists", 0)
        st["gun%"] = 100.0 * gun / seen if seen else 100.0
        if combat_stats:
            from vizdoom import GameVariable

            combat["hits"] = env._game.get_game_variable(GameVariable.HITCOUNT)
            st.update(combat)
        if progress_stats:
            progress["stale_tail"] = st["steps"] - progress.pop("last_new")
            st.update(progress)
    st["seed"] = seed
    st["map"] = f"{Path(wad).stem}:{mapname}"
    st["score"] = proxy_score(st)
    return st


def try_episode(job):
    try:
        return run_one(job)
    except Exception as exc:
        wad, name, seed, *_ = job
        return {"map": f"{Path(wad).stem}:{name}", "seed": seed, "error": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--wad", type=Path, action="append", help="evaluate this WAD; repeat to compare layouts")
    ap.add_argument("--map", default="MAP01", help="map name for --wad")
    ap.add_argument("--split", choices=("val", "progression", "public", "hard"), default="val")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--first-seed", type=int, default=0, help="first seed in the benchmark range")
    ap.add_argument("--temperature", type=float, help="override policy sampling temperature for diagnostics")
    ap.add_argument("--combat-stats", action="store_true", help="show hits, fist time, and weapon switch presses")
    ap.add_argument("--progress-stats", action="store_true", help="show close-gate observations, Use presses, and stagnant tail")
    ap.add_argument("--ticks", type=int, default=4200)
    args = ap.parse_args()
    if min(args.procs, args.seeds, args.ticks) < 1 or args.first_seed < 0:
        ap.error("--procs, --seeds, and --ticks must be positive and --first-seed must be nonnegative")
    if args.temperature is not None and not (np.isfinite(args.temperature) and args.temperature > 0):
        ap.error("--temperature must be positive and finite")
    sys.path.insert(0, str(ROOT / "training"))
    import maps

    weights = str(Path(args.weights).resolve()) if args.weights else ""
    if not args.wad and args.split != "public":
        maps.ensure_hard_levels()
    layouts = ([(str(wad.resolve()), args.map.upper(), "custom") for wad in args.wad]
               if args.wad else maps.build_pool(args.split))
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    jobs = [(wad, name, seed, args.ticks, weights, args.temperature, args.combat_stats, args.progress_stats)
            for wad, name, _ in layouts for seed in seeds]
    keys = ("exit", "death", "keys", "doors", "locked", "cells", "kills", "dmg", "score", "prog", "fists", "gun%")
    if args.combat_stats:
        keys += ("steps", "fist", "far_fist", "switch", "loaded", "hits")
    if args.progress_stats:
        keys += ("near_gate", "gate_use", "stale_tail", "barrel_atk")
    print(f"{'map':32s} {'seed':>4s} " + " ".join(f"{k:>6s}" for k in keys), flush=True)
    completed = []
    with mp.get_context("spawn").Pool(args.procs) as pool:
        for r in pool.imap(try_episode, jobs):
            if "error" in r:
                print(f"{r['map']:32s} {r['seed']:4d} ERROR {r['error']}", flush=True)
                continue
            completed.append(r)
            metrics = " ".join(f"{r[k]:6.1f}" for k in keys)
            print(f"{r['map']:32s} {r['seed']:4d} {metrics}", flush=True)
    if completed:
        print(f"{'MEAN':37s} " + " ".join(f"{np.mean([r[k] for r in completed]):6.1f}" for k in keys))
    print(f"Completed {len(completed)}/{len(jobs)} episodes")
    return int(len(completed) != len(jobs))


if __name__ == "__main__":
    raise SystemExit(main())
