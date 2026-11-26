"""Recurrent PPO for doom_env, written from scratch (no SB3).

    python training/train.py --workers 10 --run main            # train (GPU learner if available)
    python training/train.py --run main --resume                # continue
    python training/train.py --run main --export                # latest checkpoint -> submission/weights/policy.pt

Design: one subprocess per CPU core runs doom_env + FeatureBuilder + reward shaping; the main
process batches observations, samples actions on the GPU, and runs PPO with truncated BPTT over
32-step chunks of the GRU.
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "training"))

from model.actions import mask_distant_punches, movement_logits, sample_exclusive  # noqa: E402
from model.features import FeatureBuilder  # noqa: E402
from model.policy import DoomPolicy  # noqa: E402

KEYS_FLOAT = ("ray_dist", "ent_feat", "ent_mask", "extra", "map")
KEYS_INT = ("ray_type", "ent_type")
KEYS_U8 = ("rgb",)
ALL_KEYS = KEYS_FLOAT + KEYS_INT + KEYS_U8


# ============================================================================ worker
def worker(idx: int, conn, seed: int, episodes_per_map: int, use_rgb: bool,
           curiosity: float, progression_only: bool, practice_maps: Path | None,
           weapon_discipline: bool, loaded_switch_cost: float, arena_maps: Path | None):
    os.environ["OMP_NUM_THREADS"] = "1"
    import warnings

    warnings.filterwarnings("ignore")
    from doom_env import make_env

    import maps
    from rewards import Shaper

    rng = np.random.default_rng(seed)
    pool = maps.build_pool("progression" if progression_only else "train", practice_maps, arena_maps)
    map_index = idx
    env, fb, shaper = None, FeatureBuilder(), Shaper(weapon_discipline, loaded_switch_cost)
    env_steps, eps_left, cur = 0, 0, None
    bad: set = set()

    def new_episode():
        nonlocal env, eps_left, cur, map_index
        while True:
            if env is None or eps_left <= 0:
                if env is not None:
                    env.close()
                    env = None
                available = [entry for entry in pool if entry[:2] not in bad]
                if not available:
                    raise RuntimeError("No usable training maps remain")
                if progression_only:
                    cand = available[map_index % len(available)]
                    map_index += 1
                    ticks = 2100
                else:
                    cand = maps.sample(available, env_steps, rng)
                    ticks = int(rng.choice([2100, 2800, 3500, 4200]))
                try:
                    env = make_env(cand[0], cand[1], episode_ticks=ticks, render_buffers=use_rgb)
                except Exception as e:  # unloadable map: skip it forever
                    print(f"[worker {idx}] skip {cand[1]} in {Path(cand[0]).name}: {e}", flush=True)
                    bad.add((cand[0], cand[1]))
                    env = None
                    continue
                cur, eps_left = cand, episodes_per_map
            eps_left -= 1
            try:
                obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
            except Exception as e:
                print(f"[worker {idx}] reset failed on {cur[1]}: {e}", flush=True)
                bad.add((cur[0], cur[1]))
                env.close()
                env = None
                continue
            fb.reset()
            shaper.reset(env, obs)
            return obs

    obs = new_episode()
    feats = fb.step(obs)
    conn.send((feats, 0.0, True, None))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        action, env_steps = msg
        fb.note_action(action)
        obs, _, term, trunc, _ = env.step(action)
        r = shaper.step(env, action, obs, term or trunc)
        info = None
        if term or trunc:
            info = dict(shaper.stats, map=f"{Path(cur[0]).stem}:{cur[1]}")
            obs = new_episode()
            feats = fb.step(obs)
            conn.send((feats, r, True, info))
        else:
            feats = fb.step(obs)
            # Intrinsic curiosity: count-based novelty on the agent's OWN memory map (odometry
            # cells, no ground truth). New cells pay most; revisits decay as 1/sqrt(visits).
            bonus = curiosity / math.sqrt(fb.visits_here())
            shaper.stats["intrinsic"] = shaper.stats.get("intrinsic", 0.0) + bonus
            conn.send((feats, r + bonus, False, info))
    if env is not None:
        env.close()


# ============================================================================ helpers
class RunningStd:
    def __init__(self):
        self.n, self.mean, self.m2 = 1e-4, 0.0, 1.0

    def update(self, x):
        x = np.asarray(x, np.float64).ravel()
        b_mean, b_var, b_n = x.mean(), x.var(), x.size
        d = b_mean - self.mean
        tot = self.n + b_n
        self.mean += d * b_n / tot
        self.m2 = (self.m2 * self.n + b_var * b_n + d * d * self.n * b_n / tot) / tot
        self.n = tot

    @property
    def std(self):
        return float(np.sqrt(self.m2) + 1e-6)


def stack(feat_list):
    return {k: np.stack([f[k] for f in feat_list]) for k in ALL_KEYS}


def to_torch(batch: dict, device):
    out = {}
    for k, v in batch.items():
        t = torch.from_numpy(v)
        out[k] = t.to(device, non_blocking=True)
    return out


def bernoulli_logp_ent(logits, actions):
    logp = -torch.nn.functional.binary_cross_entropy_with_logits(logits, actions, reduction="none").sum(-1)
    p = torch.sigmoid(logits)
    ent = torch.nn.functional.binary_cross_entropy_with_logits(logits, p, reduction="none").sum(-1)
    return logp, ent


def exclusive_logp_ent(logits, actions):
    movement = torch.distributions.Categorical(logits=movement_logits(logits), validate_args=False)
    choices = actions[..., :6:2].long() + 2 * actions[..., 1:6:2].long()
    logp, entropy = bernoulli_logp_ent(logits[..., 6:], actions[..., 6:])
    return logp + movement.log_prob(choices).sum(-1), entropy + movement.entropy().sum(-1)


def combat_parameters(model):
    """Update attack and next-weapon outputs while keeping the navigation policy fixed."""
    model.requires_grad_(False)
    model.v.requires_grad_(True)
    output = model.pi[-1]
    output.requires_grad_(True)
    keep = torch.zeros_like(output.bias)
    keep[[6, 8]] = 1
    output.bias.register_hook(lambda gradient: gradient * keep)
    output.weight.register_hook(lambda gradient: gradient * keep[:, None])
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--steps", type=int, default=128, help="rollout length per worker")
    ap.add_argument("--chunk", type=int, default=32, help="BPTT length")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.15)
    ap.add_argument("--ent", type=float, default=0.03)
    ap.add_argument("--ent-final", type=float, default=0.01)
    ap.add_argument("--total", type=float, default=60e6, help="env steps for the schedules")
    ap.add_argument("--episodes-per-map", type=int, default=3)
    ap.add_argument("--progression-only", action="store_true", help="cycle through door and keycard practice maps")
    ap.add_argument("--practice-maps", type=Path, help="add generated WADs to the progression curriculum")
    ap.add_argument("--arena-maps", type=Path, help="add generated combat arenas to the map pool")
    ap.add_argument("--curiosity", type=float, default=0.03, help="intrinsic novelty bonus scale (0 = off)")
    ap.add_argument("--combat-head-only", action=argparse.BooleanOptionalAction, default=None,
                    help="fine-tune attack and weapon outputs plus the critic")
    ap.add_argument("--loaded-switch-cost", type=float, default=0.05, help="training penalty for cycling a loaded weapon under fire")
    ap.add_argument("--weapon-discipline", action="store_true", help="train with melee feasibility and combat feedback")
    ap.add_argument("--exclusive-movement", action="store_true", help="learn one choice per opposing movement pair")
    ap.add_argument("--no-rgb", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--export-path", type=Path, default=ROOT / "submission" / "weights" / "policy.pt")
    ap.add_argument("--max-updates", type=int, default=10**9)
    ap.add_argument("--save-every", type=int, default=20)
    args = ap.parse_args()

    if not math.isfinite(args.loaded_switch_cost) or args.loaded_switch_cost < 0:
        ap.error("--loaded-switch-cost must be finite and nonnegative")
    if args.combat_head_only and not args.resume:
        ap.error("--combat-head-only needs a trained checkpoint via --resume")
    if args.practice_maps is not None and not any(args.practice_maps.glob("*.wad")):
        ap.error("--practice-maps must contain at least one WAD")
    if args.arena_maps is not None and not any(args.arena_maps.glob("*.wad")):
        ap.error("--arena-maps must contain at least one WAD")
    if args.progression_only and args.arena_maps is not None:
        ap.error("--arena-maps requires the mixed curriculum")
    if args.episodes_per_map < 1:
        ap.error("--episodes-per-map must be positive")
    if args.workers < 1 or args.chunk < 1 or args.steps < 1 or args.steps % args.chunk:
        ap.error("positive workers and steps divisible by a positive chunk are required")
    if not 1 <= args.minibatches <= args.workers * args.steps // args.chunk:
        ap.error("each minibatch must contain at least one recurrent sequence")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_dir = ROOT / "training" / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_rgb = not args.no_rgb
    exclusive_movement = args.exclusive_movement
    weapon_discipline = args.weapon_discipline
    latest = run_dir / "latest.pt"
    ck = None
    if args.resume or args.export:
        if not latest.is_file():
            ap.error(f"missing trained checkpoint: {latest}")
        ck = torch.load(latest, map_location=device, weights_only=False)
        use_rgb = ck["config"]["use_rgb"]
        exclusive_movement = exclusive_movement or ck["config"].get("exclusive_movement", False)
        weapon_discipline = weapon_discipline or ck["config"].get("weapon_discipline", False)
    prior_combat_only = bool(ck and ck["config"].get("combat_head_only", False))
    combat_only = prior_combat_only if args.combat_head_only is None else args.combat_head_only
    model = DoomPolicy(use_rgb=use_rgb, hidden=ck["config"].get("hidden", 512) if ck else 512).to(device)
    config = {"hidden": model.hidden, "use_rgb": use_rgb, "exclusive_movement": exclusive_movement,
              "weapon_discipline": weapon_discipline, "combat_head_only": combat_only}
    action_stats = exclusive_logp_ent if exclusive_movement else bernoulli_logp_ent
    parameters = combat_parameters(model) if combat_only else model.parameters()
    opt = torch.optim.Adam(parameters, lr=args.lr, eps=1e-5)
    ret_std = RunningStd()
    update, env_steps = 0, 0

    if ck is not None:
        model.load_state_dict(ck["model"])
        # Optimizer moments from full-policy training could update frozen action rows.
        if not args.export and combat_only == prior_combat_only:
            opt.load_state_dict(ck["opt"])
        update, env_steps = ck["update"], ck["env_steps"]
        ret_std.n, ret_std.mean, ret_std.m2 = ck["ret_std"]
        print(f"resumed update {update}, {env_steps / 1e6:.1f}M steps")
    if args.export:
        out = args.export_path
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {k: v.half().cpu() for k, v in model.state_dict().items()},
                    "config": config,
                    "meta": {"update": update, "env_steps": env_steps}}, out)
        print(f"exported -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return

    ctx = mp.get_context("spawn")
    conns, procs = [], []
    seed0 = args.seed + env_steps
    for i in range(args.workers):
        a, b = ctx.Pipe()
        p = ctx.Process(
            target=worker,
            args=(i, b, seed0 + 7919 * i, args.episodes_per_map, use_rgb,
                  args.curiosity, args.progression_only, args.practice_maps, weapon_discipline, args.loaded_switch_cost, args.arena_maps),
            daemon=True,
        )
        p.start()
        b.close()
        conns.append(a)
        procs.append(p)
    N, T = args.workers, args.steps
    first = [c.recv() for c in conns]
    cur = stack([f[0] for f in first])
    cur_first = np.ones(N, np.float32)
    h = model.initial_state(N, device)

    log_f = open(run_dir / "episodes.csv", "a", newline="")
    log = csv.writer(log_f)
    if log_f.tell() == 0:
        log.writerow(["env_steps", "map", "ret", "exit", "death", "kills", "keys", "doors", "locked", "cells", "steps", "dmg"])
    recent = deque(maxlen=200)
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(run_dir / "tb"))
    except Exception:
        tb = None

    t_start = time.time()
    disc_ret = np.zeros(N, np.float64)
    while update < args.max_updates:
        t0 = time.time()
        buf = {k: np.zeros((T, N) + cur[k].shape[1:], cur[k].dtype) for k in ALL_KEYS}
        acts = np.zeros((T, N, 9), np.float32)
        logps = np.zeros((T, N), np.float32)
        vals = np.zeros((T + 1, N), np.float32)
        rews = np.zeros((T, N), np.float32)
        firsts = np.zeros((T + 1, N), np.float32)
        h_chunks = []
        model.eval()
        for t in range(T):
            if t % args.chunk == 0:
                h_chunks.append(h.clone())
            for k in ALL_KEYS:
                buf[k][t] = cur[k]
            firsts[t] = cur_first
            with torch.no_grad():
                x = {k: v.unsqueeze(0) for k, v in to_torch(cur, device).items()}
                d = torch.from_numpy(cur_first).to(device).unsqueeze(0)
                logits, v, h = model(x, h, d)
                if weapon_discipline:
                    logits = mask_distant_punches(logits, x)
                logits = logits[0]
                a = sample_exclusive(logits) if exclusive_movement else torch.bernoulli(logits.sigmoid())
                lp, _ = action_stats(logits, a)
            a_np = a.cpu().numpy().astype(np.int8)
            acts[t] = a_np
            logps[t] = lp.cpu().numpy()
            vals[t] = v[0].cpu().numpy()
            for i, c in enumerate(conns):
                c.send((a_np[i], env_steps))
            res = [c.recv() for c in conns]
            cur = stack([r[0] for r in res])
            rews[t] = [r[1] for r in res]
            cur_first = np.array([r[2] for r in res], np.float32)
            for r in res:
                if r[3] is not None:
                    info = r[3]
                    recent.append(info)
                    log.writerow([env_steps, info["map"], round(info["ret"], 2), info["exit"], info["death"], info["kills"],
                                  info["keys"], info["doors"], info["locked"], info["cells"], info["steps"], round(info["dmg"])])
            env_steps += N
        with torch.no_grad():
            x = {k: v.unsqueeze(0) for k, v in to_torch(cur, device).items()}
            d = torch.from_numpy(cur_first).to(device).unsqueeze(0)
            _, v, _ = model(x, h, d)
            vals[T] = v[0].cpu().numpy()
        firsts[T] = cur_first
        t_roll = time.time() - t0

        # ---- returns / advantages (value is in normalised units)
        for t in range(T):  # scale rewards by the spread of discounted returns (standard PPO practice)
            disc_ret = disc_ret * args.gamma * (1.0 - firsts[t]) + rews[t]
            ret_std.update(disc_ret)
        r_n = np.clip(rews / ret_std.std, -10, 10)
        adv = np.zeros((T, N), np.float32)
        last = 0.0
        for t in reversed(range(T)):
            nonterm = 1.0 - firsts[t + 1]
            delta = r_n[t] + args.gamma * vals[t + 1] * nonterm - vals[t]
            last = delta + args.gamma * args.lam * nonterm * last
            adv[t] = last
        rets = adv + vals[:T]

        # ---- PPO update over chunks of length L
        frac = min(1.0, env_steps / args.total)
        ent_coef = args.ent + (args.ent_final - args.ent) * frac
        for g in opt.param_groups:
            g["lr"] = args.lr * max(0.1, 1.0 - frac)
        model.train()
        L = args.chunk
        n_chunks = T // L
        tb_ = {k: torch.from_numpy(v).to(device) for k, v in buf.items()}
        A = torch.from_numpy(acts).to(device)
        LP = torch.from_numpy(logps).to(device)
        ADV = torch.from_numpy(adv).to(device)
        RET = torch.from_numpy(rets).to(device)
        VOLD = torch.from_numpy(vals[:T]).to(device)
        FIRST = torch.from_numpy(firsts[:T]).to(device)
        H0 = torch.cat(h_chunks, 1)  # (1, n_chunks*N, H), chunk-major
        seq_ids = np.arange(n_chunks * N)
        stats = []
        for _ in range(args.epochs):
            np.random.shuffle(seq_ids)
            for mb in np.array_split(seq_ids, args.minibatches):
                ci, ei = mb // N, mb % N  # chunk index, env index
                tt = (ci[None, :] * L + np.arange(L)[:, None])  # (L, M)
                ee = np.broadcast_to(ei[None, :], tt.shape)
                tt_t, ee_t = torch.from_numpy(tt).to(device), torch.from_numpy(np.ascontiguousarray(ee)).to(device)
                x = {k: v[tt_t, ee_t] for k, v in tb_.items()}
                h0 = H0[:, torch.from_numpy(mb).to(device)]
                logits, v, _ = model(x, h0, FIRST[tt_t, ee_t])
                if weapon_discipline:
                    logits = mask_distant_punches(logits, x)
                lp, ent = action_stats(logits, A[tt_t, ee_t])
                ratio = torch.exp(lp - LP[tt_t, ee_t])
                adv_b = ADV[tt_t, ee_t]
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)
                pg = -torch.min(ratio * adv_b, ratio.clamp(1 - args.clip, 1 + args.clip) * adv_b).mean()
                v_old = VOLD[tt_t, ee_t]
                v_clip = v_old + (v - v_old).clamp(-0.2, 0.2)
                ret_b = RET[tt_t, ee_t]
                vf = 0.5 * torch.max((v - ret_b) ** 2, (v_clip - ret_b) ** 2).mean()
                loss = pg + 0.5 * vf - ent_coef * ent.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                stats.append((pg.item(), vf.item(), ent.mean().item(), kl))
        update += 1
        t_upd = time.time() - t0 - t_roll

        s = np.mean(stats, 0)
        sps = N * T / (time.time() - t0)
        if recent:
            m = {k: np.mean([e.get(k, 0.0) for e in recent]) for k in ("ret", "exit", "death", "kills", "keys", "doors", "cells", "steps", "intrinsic")}
        else:
            m = {}
        print(f"upd {update} steps {env_steps / 1e6:.2f}M sps {sps:.0f} (roll {t_roll:.1f}s upd {t_upd:.1f}s) "
              f"pg {s[0]:.3f} vf {s[1]:.3f} ent {s[2]:.2f} kl {s[3]:.4f} | "
              + " ".join(f"{k} {v:.2f}" for k, v in m.items()), flush=True)
        if tb is not None:
            for k, v in m.items():
                tb.add_scalar(f"ep/{k}", v, env_steps)
            tb.add_scalar("train/sps", sps, env_steps)
            tb.add_scalar("train/entropy", s[2], env_steps)
            tb.add_scalar("train/kl", s[3], env_steps)
        if update % args.save_every == 0:
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "update": update, "env_steps": env_steps,
                  "ret_std": (ret_std.n, ret_std.mean, ret_std.m2), "config": config}
            torch.save(ck, run_dir / "latest.tmp")
            os.replace(run_dir / "latest.tmp", latest)
            if update % (args.save_every * 5) == 0:
                torch.save({"model": {k: v.half().cpu() for k, v in model.state_dict().items()},
                            "config": ck["config"], "meta": {"update": update, "env_steps": env_steps}},
                           run_dir / f"policy_{env_steps // 1_000_000:04d}M.pt")
            log_f.flush()

    for c in conns:
        c.send(None)
    print(f"done in {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
