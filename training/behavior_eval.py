"""Measure how a checkpoint actually plays: the real submission DoomAgent on fixed Freedoom maps and seeds.

    python training/behavior_eval.py --weights training/runs/main/policy_0002M.pt --tag 2M
    python training/behavior_eval.py --weights training/runs/main/policy_0002M.pt --temp 0.7 --tag 2M_t07

Per episode it records exits, deaths, kills, coverage (128-unit areas), revisits (3rd+ entries), stall steps
(< 48 units moved in 16 steps, no enemy in view), turn jitter, shots with no enemy visible, switches to fists
while a gun has ammo, damage taken (and while strafing), medkits seen below 60 HP and actually taken.
Ground truth comes from privileged_state(); the agent only ever sees the official observation.
Compare runs with --compare tagA tagB ... (reads training/runs/eval_<tag>.json).
"""
import argparse, importlib.util, json, multiprocessing as mp, os, sys, time
from pathlib import Path
import numpy as np

ENEMY = (1, 2, 3)
HEALTH = 5


def load_agent(repo, weights, temp):
    sys.path.insert(0, str(Path(repo) / "submission"))
    spec = importlib.util.spec_from_file_location("agent_mod", Path(repo) / "submission" / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.WEIGHTS = Path(weights)
    if temp is not None:
        mod.TEMPERATURE = temp
    return mod


def run(job):
    repo, weights, temp, wad, mapname, seed, ticks = job
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from doom_env import make_env
    mod = load_agent(repo, weights, temp)
    env = make_env(wad, mapname, episode_ticks=ticks)
    agent = mod.DoomAgent(env.observation_space, env.action_space, "cpu")
    agent.rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    agent.reset()
    s0 = env.privileged_state()
    M = dict(exit=0, death=0, kills=0, steps=0, fist_switch_with_ammo=0, fist_steps_with_ammo=0, shots_no_enemy=0,
             shots=0, stall_steps=0, revisits=0, cells=1, dmg=0.0, dmg_while_strafing=0.0, medkit_seen_lowhp=0,
             medkit_taken_lowhp=0, healed=0.0, weapon_switch_presses=0, jitter=0)
    trail = [(s0["x"], s0["y"])]
    entries, last_cell = {(int(s0["x"] // 128), int(s0["y"] // 128)): 1}, None
    prev = s0
    prev_slot = int(obs["vitals"][2])
    prev_turn = 0
    seek_window = 0  # steps left in which a seen medkit can still be taken
    while True:
        a = agent.act(obs)
        v = obs["vitals"]
        ent = obs["visible_entities"]
        mask = ent["mask"] > 0
        enemy_vis = bool(np.any(mask & np.isin(ent["entity_type"], ENEMY)))
        hd = ent["features"][mask & (ent["entity_type"] == HEALTH), 0] * 1024
        if a[6]:
            M["shots"] += 1
            if not enemy_vis:
                M["shots_no_enemy"] += 1
        if a[8]:
            M["weapon_switch_presses"] += 1
        if v[0] < 60 and hd.size and hd.min() < 400 and seek_window == 0:
            M["medkit_seen_lowhp"] += 1
            seek_window = 40
        turn = int(a[4]) - int(a[5])
        if turn and prev_turn and turn != prev_turn and not enemy_vis:
            M["jitter"] += 1
        if turn:
            prev_turn = turn
        obs, _, te, tu, _ = env.step(a)
        s = env.privileged_state()
        M["steps"] += 1
        slot = int(obs["vitals"][2])
        gun_ammo = obs["vitals"][4] > 0 or obs["vitals"][5] > 0
        if slot == 1 and prev_slot > 1 and gun_ammo:
            M["fist_switch_with_ammo"] += 1
        if slot == 1 and gun_ammo:
            M["fist_steps_with_ammo"] += 1
        prev_slot = slot
        dd = max(0.0, s["damage_taken"] - prev["damage_taken"])
        M["dmg"] += dd
        if a[2] != a[3]:
            M["dmg_while_strafing"] += dd
        heal = max(0.0, s["health"] - prev["health"]) if not s["dead"] else 0
        M["healed"] += heal
        if seek_window:
            if heal > 0 and prev["health"] < 60:
                M["medkit_taken_lowhp"] += 1
                seek_window = 0
            else:
                seek_window -= 1
        c = (int(s["x"] // 128), int(s["y"] // 128))
        if c != last_cell:
            last_cell = c
            n = entries[c] = entries.get(c, 0) + 1
            if n == 1:
                M["cells"] += 1
            if n > 2:
                M["revisits"] += 1
        trail.append((s["x"], s["y"]))
        if len(trail) > 16:
            trail.pop(0)
            xs, ys = zip(*trail)
            if max(xs) - min(xs) < 48 and max(ys) - min(ys) < 48 and not enemy_vis:
                M["stall_steps"] += 1
        prev = s
        if te or tu:
            break
    M["exit"] = int(s["exit_reached"])
    M["death"] = int(s["dead"])
    M["kills"] = s["kills"]
    M["final_hp"] = s["health"]
    M["map"] = f"{Path(wad).stem}:{mapname}"
    M["seed"] = seed
    env.close()
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--weights", default=str(Path(__file__).resolve().parents[1] / "submission" / "weights" / "policy.pt"))
    ap.add_argument("--temp", type=float, default=None, help="override submission/agent.py TEMPERATURE")
    ap.add_argument("--compare", nargs="+", help="print saved results side by side and exit")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--maps", default="MAP01,MAP02,MAP03,MAP04,MAP05,MAP06,MAP07")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--ticks", type=int, default=2100)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    keys = ["exit", "death", "kills", "cells", "revisits", "stall_steps", "jitter", "shots_no_enemy", "shots",
            "fist_switch_with_ammo", "fist_steps_with_ammo", "dmg", "dmg_while_strafing", "medkit_seen_lowhp", "medkit_taken_lowhp", "steps"]
    outdir = Path(a.repo) / "training" / "runs"
    if a.compare:
        R = {t: json.loads((outdir / f"eval_{t}.json").read_text()) for t in a.compare}
        print(f"{'metric':22s}" + "".join(f"{t:>14s}" for t in a.compare))
        for k in keys:
            print(f"{k:22s}" + "".join(f"{np.mean([r[k] for r in R[t]]):14.2f}" for t in a.compare))
        return
    import vizdoom
    iw = str(Path(vizdoom.__file__).parent / "freedoom2.wad")
    jobs = [(a.repo, a.weights, a.temp, iw, m, s, a.ticks) for m in a.maps.split(",") for s in range(a.seeds)]
    t = time.time()
    with mp.get_context("spawn").Pool(a.procs) as p:
        res = p.map(run, jobs)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"eval_{a.tag}.json"
    out.write_text(json.dumps(res, indent=1))
    tot = {k: float(np.mean([r[k] for r in res])) for k in keys}
    print(f"[{a.tag}] {len(res)} episodes in {time.time() - t:.0f}s  " + " ".join(f"{k}={v:.2f}" for k, v in tot.items()))


if __name__ == "__main__":
    main()
