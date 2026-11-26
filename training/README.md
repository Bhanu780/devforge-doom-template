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

Press R to restart the current episode. The result screen distinguishes a completed
level, a death, and a time limit.

Generate and play the side-room keycard layout, then benchmark a checkpoint on it.

```bash
python training/keycard_map.py training/runs/keycard_holdout.wad
python training/play.py --wad training/runs/keycard_holdout.wad
python training/bench.py --wad training/runs/keycard_holdout.wad --seeds 4 --procs 2
```

This layout is outside the training map pool. Keep it out of `training/extra_maps/`
when measuring generalization.

## Automatic side-room training

The imitation trainer generates twelve side-room layouts with four starts each.
Six layouts have a bend between the junction and the key. A training-only
controller collects demonstrations using the engine state. The
neural policy fits the demonstrated buttons from the same observations it gets
at evaluation. Omit `--init` to train from fresh weights. For PPO fine-tuning,
pass only the generated practice directory.
Keep the reserved side-room WAD outside that directory.

```bash
python training/imitate_keycard.py --init training/runs/balanced_progression/candidate.pt
python training/train.py --run keycard_imitation --export --export-path training/runs/keycard_imitation/candidate.pt
python training/bench.py --weights training/runs/keycard_imitation/candidate.pt --wad training/runs/keycard_holdout.wad --seeds 4 --procs 2 --ticks 2100
python training/train.py --run keycard_imitation --resume --practice-maps training/runs/keycard_imitation/maps --workers 2 --episodes-per-map 1 --max-updates 24 --save-every 12
```

To retain combat behavior while fitting the keycard routes, collect additional
action-probability targets from an earlier policy on the bundled training combat
scenarios. The targets are interleaved with the navigation demonstrations.

```bash
python training/imitate_keycard.py --run keycard_bends --epochs 10 \
  --init training/runs/keycard_ppo/candidate.pt \
  --combat-reference training/runs/balanced_progression/candidate.pt
python training/train.py --run keycard_bends --export --export-path training/runs/keycard_bends/candidate.pt
python training/bench.py --weights training/runs/keycard_bends/candidate.pt --seeds 2 --procs 2 --ticks 2100
```

Use the same checkpoint, seed list, and tick cap when comparing navigation maps
and combat scenarios. The held-out suite measures transfer between layouts.

## Harder generated levels

`training/hard_levels.py` generates random corridor levels in three tiers: decoy dead-ends with a
hidden red key (1); plus a key room behind a plain door, longer bent branches and guards (2); a
blue-then-red two-key chain with ambushes in dead ends (3). The demonstration route is found by a
training-only BFS over (tile, keys held). `--check` plays every level with the demonstration
controller to prove it can be completed.

```bash
python training/hard_levels.py training/runs/hard_maps --count 36 --check
python training/imitate_keycard.py --run hard --hard-levels 36 --init training/runs/keycard_bends/candidate.pt
python training/train.py --run hard --resume --practice-maps training/runs/hard_maps --progression-only --episodes-per-map 1
```

## Weapon handling

`features.action_mask()` masks two buttons from observation fields (allowed by docs/RULES.md §1),
inside the network so training and evaluation match: SELECT_NEXT_WEAPON while holding a loaded gun
with no enemy within 96 units (one press from the pistol wraps to the fists), and ATTACK while
holding fists with no enemy within reach. The shaper adds `fists_vs_far` (-0.02 per step holding
fists while bullets or shells remain and the nearest visible enemy is out of reach) so the policy
learns to switch back. Episode logs report it as `fists`.
