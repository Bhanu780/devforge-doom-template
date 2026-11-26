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
| `train.py --curiosity` | Intrinsic curiosity: 0.03 / sqrt(visits) for the agent's current cell in its *own* memory map (odometry, no ground truth). Default on |
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

## CPU progression baseline

Activate the project environment and install CPU PyTorch. The following Linux commands
run a short, seeded experiment with a single engine worker. The progression group
samples the public two-room map and the public red-key / locked-door fixture.
These are training layouts. Benchmark on the separate validation pool.

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ALSOFT_DRIVERS=null SDL_VIDEODRIVER=dummy
DOOM_GROUPS=progression=1 python training/train.py --run door_baseline --workers 1 --no-rgb --steps 128 --chunk 16 --epochs 2 --seed 42 --max-updates 32 --save-every 8
python training/train.py --run door_baseline --export
python training/bench.py --seeds 2 --procs 1 --ticks 2100
python validate.py
```

Compare exits, keys, unlocks, deaths and coverage on identical maps and seeds.
The printed proxy score uses team-selected weights. To continue training on the
mixed curriculum, repeat the training command with `--resume`, remove the
`DOOM_GROUPS` override and increase the absolute update limit. Export restores
the checkpoint's RGB setting. Archive the current result before exporting a new
candidate to the same submission path.

Design references: [DeepDoom's door and exit scenarios](https://github.com/Atlas-Soft/DeepDoom#scenarios)
for the progression curriculum and [CleanRL's recurrent PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py)
for comparison with the team's recurrent sequence handling.

For guaranteed practice on both progression maps, use `--progression-only
--episodes-per-map 1`. Workers start on staggered maps and cycle to the next map
after each episode. Export candidates to the run directory so the submission
checkpoint stays available for comparison.

```bash
python training/train.py --run balanced_progression --progression-only --episodes-per-map 1 --workers 2 --no-rgb --steps 128 --chunk 16 --epochs 2 --max-updates 48 --save-every 12
python training/train.py --run balanced_progression --export --export-path training/runs/balanced_progression/candidate.pt
python training/bench.py --weights training/runs/balanced_progression/candidate.pt --split progression --seeds 4 --procs 2 --ticks 2100
python training/bench.py --weights training/runs/balanced_progression/candidate.pt --seeds 2 --procs 2 --ticks 2100
```

The progression benchmark diagnoses key pickups, unlocks, and exits on the two
practice maps. Use the default validation split to evaluate generalization.

## Play or watch locally

From the repository root, activate the project environment and launch the bundled
two-room map on a desktop with a display.

```bash
source .venv/bin/activate
python training/play.py
python training/play.py --bot
python training/play.py --bot --weights training/runs/balanced_progression/candidate.pt
```

Use W/S to move forward/back, A/D to strafe, left/right arrows to turn, Space to
shoot, E to use doors and switches, Q to cycle weapons, and Esc to quit.
Restart the command for another episode. To try a different layout, supply
`--wad path/to/map.wad --map MAP01`.
