# Training

Everything the judges need to reproduce the agent. Model code lives in `submission/model/` so that
training and the evaluated `submission/agent.py` share exactly the same preprocessing and network.

## Pieces

| File | What it does |
| --- | --- |
| `submission/model/features.py` | Odometry (heading from our own turn buttons using Doom's slow/fast turn rule; position from player-frame velocity; static-scan check for wall pushing) and the ego-centric memory map (free, wall, door, locked door, item, key, enemy, visited), cropped 32x32 ahead-up for the policy |
| `submission/model/policy.py` | Recurrent actor-critic: circular ray conv, masked entity self-attention, map CNN, RGB+depth CNN, GRU 512, 9 Bernoulli buttons + value |
| `training/train.py` | Our own recurrent PPO: one subprocess per core, GPU learner, truncated BPTT over 32-step chunks, GAE, return normalisation, entropy and LR schedules |
| `training/rewards.py` | Shaped reward from `privileged_state()` (training only): exit, keys, doors, locked doors, new areas, kills, damage, death |
| `training/maps.py` | Map pool (Freedoom 2 MAP01-32, bundled ViZDoom scenarios, repo test WADs, anything in `training/extra_maps/`) and the curriculum weights |
| `training/bench.py` | Held-out benchmark (6 Freedoom maps + `defend_the_line`) through the real `DoomAgent` |

## Run it (friend's laptop: RTX 4060, Windows or Linux)

```bash
uv sync                                   # vizdoom, gymnasium, the doom_env wrapper
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA torch for the RTX 4060
uv pip install tensorboard
uv run python -c "import torch; print(torch.cuda.is_available())"         # must print True
uv run python training/train.py --workers 10 --run main            # ~1 worker per physical core minus 2
uv run tensorboard --logdir training/runs                          # watch ep/exit, ep/cells, ep/ret
```

- Stop any time with Ctrl+C; continue with `--resume`. `latest.pt` is saved every 20 updates and an
  fp16 snapshot `policy_XXXXM.pt` every 100 updates.
- Check a snapshot: `uv run python training/bench.py --weights training/runs/main/policy_0020M.pt --procs 6`.
- Ship one: copy it to `submission/weights/policy.pt` (or `train.py --run main --export` for the latest), then
  `python validate.py --docker`, then `git add submission/weights/policy.pt && git commit && git push` (Git LFS).
- Want more maps? Put any Doom 2 format WAD into `training/extra_maps/` (fewer than 2,048 sectors per map).
- Throughput is CPU-bound (big Freedoom maps run 100-500 steps/s per core). If the reward curve is flat
  after 1-2 M steps, run with `DOOM_GROUPS="small=1,fd_early=1"` first, then resume without it.
