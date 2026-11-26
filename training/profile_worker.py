"""Time one training worker's step, part by part, on this machine (about 30 s).

    uv run python training/profile_worker.py

Prints per-map: engine step, reward shaping, feature building, message encoding, and the cost of loading a
map (make_env + reset). Compare with the 'timing per step' line of train.py to see where rollout time goes.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import vizdoom

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "submission")]
from doom_env import make_env  # noqa: E402
from model.features import FeatureBuilder  # noqa: E402
from rewards import Shaper  # noqa: E402

import maps  # noqa: E402

iw = str(Path(vizdoom.__file__).parent / "freedoom2.wad")
pool = maps.build_pool("train")
by_group = {}
for w, m, g in pool:
    by_group.setdefault(g, []).append((w, m))
picks = [(iw, "MAP01"), (iw, "MAP07")] + [v[0] for g, v in by_group.items() if g not in ("fd", "fd_early")]
for wad, m in picks[:6]:
    t = time.perf_counter()
    env = make_env(wad, m, episode_ticks=2100, render_buffers=True)
    obs, _ = env.reset(seed=0)
    t_make = time.perf_counter() - t
    fb, sh = FeatureBuilder(), Shaper()
    t = time.perf_counter()
    sh.reset(env)
    t_sh = time.perf_counter() - t
    rng = np.random.default_rng(0)
    T = dict(engine=0.0, rewards=0.0, features=0.0, message=0.0)
    n = 0
    while n < 200:
        a = rng.integers(0, 2, 9).astype(np.int8)
        a[0] = 1
        t = time.perf_counter(); obs, _, te, tu, _ = env.step(a); T["engine"] += time.perf_counter() - t
        t = time.perf_counter(); sh.step(env, a, obs, te or tu); T["rewards"] += time.perf_counter() - t
        t = time.perf_counter(); f = fb.step(obs); fb.note_action(a); T["features"] += time.perf_counter() - t
        t = time.perf_counter(); pickle.dumps(f); T["message"] += time.perf_counter() - t
        n += 1
        if te or tu:
            obs, _ = env.reset(); fb.reset(); sh.reset(env)
    env.close()
    per = {k: 1000 * v / n for k, v in T.items()}
    print(f"{Path(wad).stem + ':' + m:28s} load map {1000 * t_make:6.0f} ms + reward setup {1000 * t_sh:6.0f} ms | per step: "
          + " ".join(f"{k} {v:5.1f} ms" for k, v in per.items()) + f" | total {sum(per.values()):5.1f} ms", flush=True)
