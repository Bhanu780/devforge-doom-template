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
  step-numbered FP16 snapshot `policy_000000000.pt` at each save interval.
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

## Train recovery from missed turns

During extra practice rollouts, `--recovery-rate` lets the initial neural policy
choose a share of the actions. The trainer records the corrective demonstration
action for the actual observation and retains completed rollouts. This gives the
policy practice recovering from its own mistakes. The controller runs only on
the generated practice maps.

```bash
python training/imitate_keycard.py --run keycard_recovery --recovery-rate 0.25 --epochs 6 \
  --init training/runs/keycard_bends/candidate.pt \
  --combat-reference training/runs/balanced_progression/candidate.pt
python training/train.py --run keycard_recovery --export --export-path training/runs/keycard_recovery/candidate.pt
python training/keycard_map.py training/runs/keycard_holdout.wad
python training/keycard_map.py training/runs/keycard_bent_holdout.wad --layout bent
python training/bench.py --weights training/runs/keycard_recovery/candidate.pt \
  --wad training/runs/keycard_holdout.wad --wad training/runs/keycard_bent_holdout.wad \
  --seeds 4 --procs 2 --ticks 2100
```

Use `--first-seed 4` to run the next four seeds on both layouts.

## Train combat and keycard progression together

Generate practice layouts with a hitscan enemy before the key, a projectile
enemy after the door, or both. Fine-tune a copy of the recovery checkpoint so
the original stays available for comparison. Replace the run name when
starting a new experiment.

```bash
python training/keycard_curriculum.py training/runs/keycard_combat_maps
mkdir -p training/runs/keycard_combat_ppo
cp training/runs/keycard_recovery/latest.pt training/runs/keycard_combat_ppo/latest.pt
DOOM_GROUPS=progression=0.65,small=0.20,fd_early=0.15 python training/train.py \
  --run keycard_combat_ppo --resume --practice-maps training/runs/keycard_combat_maps \
  --workers 2 --episodes-per-map 1 --steps 128 --chunk 16 --epochs 2 \
  --lr 0.0001 --ent 0.01 --ent-final 0.01 --curiosity 0.015 --max-updates 32 --save-every 8
python training/train.py --run keycard_combat_ppo --export --export-path training/runs/keycard_combat_ppo/candidate.pt
```

Compare the candidate against the recovery checkpoint on the same validation
maps, seeds, and tick cap. MAP04 stays in the validation pool.

Add `--exclusive-movement` on a separate comparison run to let the model choose
neither or one button in each opposing pair. PPO and the exported agent share
that action distribution. Checkpoints saved before this option retain their
original button sampling.

## Weapon discipline experiment

Add `--weapon-discipline` when fine-tuning from a copied recovery checkpoint.
The policy uses an observation-based mask for fists when every visible target
is out of melee range. Switching remains a learned action. During training,
hits earn a small damage reward and cycling away from a loaded firearm in view
of an enemy costs a small penalty. The hit counter is used only in training.
The exported config keeps the mask consistent with PPO's action distribution.

### Design references

- [DeepDoom scenarios](https://github.com/Atlas-Soft/DeepDoom#scenarios) inform the door and exit curriculum.
- [CleanRL recurrent PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py) is a reference for recurrent state reset and sequence batching.
- [ViZDoom scenarios](https://github.com/Farama-Foundation/ViZDoom/blob/main/docs/environments/default.md) provide distinct combat and exploration practice tasks.
- [StarNet demos](https://github.com/Trinkle23897/VizDoom2018-Track1) illustrate the combat-to-progression sequences we measure in the mixed maps.
- [Curiosity-driven exploration](https://github.com/pathak22/noreward-rl) informs the novelty objective. This implementation uses counts from its own memory map.

## Inspect and tune combat independently

Run the held-out benchmark with `--combat-stats` to record decisions with the
fist selected, out-of-range punches, weapon-switch presses, switches from a
loaded weapon while an enemy is visible, and hits. The benchmark reads engine
counters after the policy acts.

To isolate combat learning, copy a completed checkpoint to a new run and add
`--combat-head-only --weapon-discipline` when resuming the trainer.
This freezes the observation encoders, recurrent memory, and the actor's
navigation and interaction output rows. PPO updates the attack and
weapon-selection rows and the value head. A fresh optimizer is created on the
first combat-only resume. Use `--no-combat-head-only` to resume full-policy
training with a fresh optimizer. Use `--loaded-switch-cost` to compare different
training penalties for cycling away from a loaded weapon under threat.

## Vary combat layouts and starting conditions

Generate combat practice separately from the arena benchmark. Practice covers
open rooms, pillars, chokepoints, and alternate routes, with four spawn
configurations per layout. The validation layouts use different cover and
enemy placements.

```bash
python training/combat_maps.py training/runs/arenas_train
python training/combat_maps.py training/runs/arenas_val --split val
```

Pass only `training/runs/arenas_train` to `train.py --arena-maps`. Combine it
with the existing keycard practice maps and bundled Freedoom training levels:

```bash
DOOM_GROUPS=arena=0.35,progression=0.35,small=0.10,fd_early=0.10,fd=0.10 \
  python training/train.py --run arenas_combat_only --resume --weapon-discipline \
  --combat-head-only --arena-maps training/runs/arenas_train \
  --practice-maps training/runs/keycard_combat_maps \
  --workers 2 --episodes-per-map 1 --steps 128 --chunk 16 --epochs 2 --lr 0.0001 \
  --loaded-switch-cost 0.08 --ent 0.01 --ent-final 0.01 --curiosity 0.015 --max-updates 48 --save-every 8
```

Seed each comparison run with a separate copy of the same starting checkpoint.
Use `--no-combat-head-only` for the full-policy comparison. Evaluate candidates
on the same reserved arenas, keycard layouts, and Freedoom maps.

## Teach exit navigation from more starting points

The exit lessons use cleared practice arenas so the model can learn movement
after an encounter as a separate skill. Each floor plan has several start
positions and headings. Only the demonstration collector follows those routes.
The neural policy is evaluated through the same observation-only entry point.

Enable RGB and depth in a separate warm-start experiment. The visual input
starts with zero contribution to the existing policy, then learns during
imitation. Rehearsing earlier combat decisions keeps combat examples in the
training batches.

```bash
python training/imitate_keycard.py --run arena_exit_rgb \
  --init training/runs/arenas_combat_only/candidate.pt \
  --combat-reference training/runs/arenas_full_policy/candidate.pt \
  --enable-rgb --arena-exits --epochs 5
python training/train.py --run arena_exit_rgb --export --export-path training/runs/arena_exit_rgb/candidate.pt
```

Compare the finished checkpoint on the reserved arena floor plans, keycard maps,
and broader Freedoom suite before continuing PPO or promoting it.

## Sweep the public maps and diagnose gate stalls

Use the public split to run all bundled Freedoom maps, ViZDoom scenarios, the
official two-room map, and the repository's test fixtures. It reports attempts
with errors separately and prints a row as each episode finishes. Use this as a
diagnostic across the public material. The `val` split and reserved WADs remain
separate from checkpoint training.

```bash
python training/bench.py --weights training/runs/arenas_combat_only/candidate.pt \
  --split public --seeds 1 --ticks 4200 --procs 2 --combat-stats --progress-stats
```

`near_gate` counts decisions with a door ray close and ahead. `gate_use` counts
fresh Use presses during those decisions. `stale_tail` counts decisions since
the last newly visited cell. These are local diagnostics, separate from the
official scoring criteria.

Train gate and exit interaction on authored rooms with doors approached from
several sides and varied initial headings.

```bash
python training/imitate_keycard.py --run gate_lessons --gate-lessons --arena-exits \
  --init training/runs/arenas_combat_only/candidate.pt \
  --combat-reference training/runs/arenas_full_policy/candidate.pt --epochs 5
python training/train.py --run gate_lessons --export --export-path training/runs/gate_lessons/candidate.pt
```

## Watch long runs in TensorBoard

Install TensorBoard in the project environment and point it at the training run.
The trainer writes the scalar files automatically.

```bash
source .venv/bin/activate
uv pip install tensorboard
tensorboard --logdir training/runs/long_gate_ppo --host 127.0.0.1 --port 6006
```

Open http://localhost:6006/ and select Scalars. `ep/exit` is the rolling training
exit rate. `ep/death`, `ep/doors`, and `ep/cells` track the corresponding
training outcomes. Compare saved checkpoints on a separate benchmark before
choosing submission weights.

For a longer PPO run, `--navigation-anchor 0.25` adds a soft penalty if movement
and Use probabilities drift from the starting checkpoint. It leaves the attack
and weapon-switch distributions outside that penalty. The trainer stores the
reference in the run directory so it can be restored on resume. Pass the same
anchor strength when resuming. Every save interval produces an immutable
step-numbered export for evaluation, and the final update is saved even when
it falls between intervals.
## Harder generated levels

`training/hard_levels.py` generates random corridor levels in three tiers: decoy dead-ends with a
hidden red key (1); plus a key room behind a plain door, longer bent branches and guards (2); a
blue-then-red two-key chain with ambushes in dead ends (3). The demonstration route is found by a
training-only BFS over (tile, keys held). `--check` plays every level with the demonstration
controller to prove it can be completed.

They are part of the default curriculum: `train.py` and `bench.py` write 60 training levels
(seed 0, group `hard`, 15-17% of episodes) to `training/runs/hard_maps/` and 9 held-out levels
(seed 1000) to `training/runs/hard_maps_val/` on first use. `--progression-only` cycles through
them too, the default `val` benchmark includes the held-out ones, and `bench.py --split hard`
runs only those.

```bash
python training/train.py --run main --resume                       # existing runs pick them up
python training/train.py --run main --export --export-path training/runs/main/candidate.pt
python training/bench.py --weights training/runs/main/candidate.pt --split hard --seeds 2
python training/hard_levels.py training/runs/hard_maps --count 36 --check
python training/imitate_keycard.py --run hard --hard-levels 36 --init training/runs/keycard_bends/candidate.pt
python training/train.py --run hard --resume --practice-maps training/runs/hard_maps --progression-only --episodes-per-map 1
```

## Weapon handling

`features.action_mask()` masks two buttons from observation fields (allowed by docs/RULES.md §1),
inside the network so training and evaluation match: SELECT_NEXT_WEAPON while holding a gun that
still has ammo (one press from the pistol wraps to the fists; Doom switches by itself when a gun runs
dry or a better one is picked up), and ATTACK while holding fists with no enemy within reach. The
shaper adds `fists_with_ammo` (-0.03 per step holding fists while bullets or shells remain) so the
policy learns to switch back. Episode logs report it as `fists`.

## Survival and momentum shaping (training/rewards.py)

| Term | Value | Fixes |
| --- | --- | --- |
| `revisit` | -0.03 x min(entries - 2, 4) on the 3rd+ entry into a 128-unit area | backtracking through cleared rooms |
| `cell` | +0.15 per new 128-unit area (was 0.1) | commits to unexplored ground |
| `bump` | -0.01 per step holding FORWARD while moving < 6 units, no enemy in view | stair lips, angled-wall scraping |
| `jitter` | -0.01 when the turn direction flips, no enemy in view | left/right dithering against walls |
| `stall` | -0.02 per step after < 48 units of movement in 16 steps, no enemy in view | standing still |
| cap | stall + bump + jitter + revisit <= 4.0 per episode | keeps dying (-8) worse than being stuck |
| `damage` | -0.03/HP x (1 + 2 x missing-health fraction) | face-tanking at low HP |
| `death` | -8 (was -5) | early deaths |
| `health_gain` | +0.04/HP healed x (1 + 2 x missing fraction before) | walking past medkits (medkits never moved `items`) |
| `seek_health` | +0.002 per unit closer to the nearest visible health pickup, below 60 HP | detours for medkits |
| `armor_gain` | +0.01 per armour point | |
| `under_fire_strafe` | +0.004 per step strafing within 12 steps of being hit | dodging instead of trading head-on |

Episode logs add `revisits`, `bumps`, `healed` and `stall`.

## Longer mixed training

Use different seeds and map families during training. The long local run mixes
Freedoom maps, generated hard keycard levels, combat arenas, and original gate
practice. The held-out hard levels are excluded from the training pool.

Action rewards use the frame before the button press when judging target
visibility, ammo waste, and whether a gate is close enough to use. During
weapon-discipline training, a confirmed hit receives small credit. Opening a
gate remains a progression event, and waiting while it opens avoids the stall
penalty. The exit bonus remains the largest single reward.

The checkpoint filenames contain the cumulative number of game decisions. Keep
the starting weights and benchmark immutable snapshots on the same maps and
seeds. The long run's local resume script records its mixture and exact
training budget in the run directory.

For a deadline-sized comparison, resume from a copied checkpoint with
`--epochs 1 --minibatches 2`, set a short absolute `--max-updates`, and save
at several intervals. One pass per rollout reduces learner time. Compare the
candidate against its starting weights before using it. No-RGB rollouts omit
the unused image tensors, and the recurrent model processes uninterrupted
sequences in batches while resetting memory at episode boundaries.

## Wall recovery and combat precision

`--collision-aware` uses the velocity corrected by the static raycast scan for the
policy's stuck feature. During training, blocked strafing and reversing receive
the same bounded penalty as blocked forward motion. Shots use the pre-action
center ray and measured damage for credit. Aimed shots at barrels are allowed
in the reward, since barrels can damage nearby enemies. The inference action
selection still comes from the same recurrent policy.

Start a run from a compatible exported policy with `--init-weights path/to/policy.pt`.
This starts a new optimizer and step counter. Use `--resume` to continue that run
with its optimizer, feature configuration, and step counter.

```bash
python training/hard_levels.py training/runs/recovery_practice --count 48 --seed 23061
DOOM_GROUPS=fd_early=0.40,fd=0.10,arena=0.15,small=0.10,progression=0.20,hard=0.05 \
  python training/train.py --run recovery_mixed --init-weights path/to/policy.pt \
  --collision-aware --weapon-discipline --seed 913 --workers 2 --steps 128 \
  --epochs 1 --minibatches 2 --lr 0.0001 --navigation-anchor 0.07 \
  --practice-maps training/runs/recovery_practice --arena-maps training/runs/arenas_train \
  --episodes-per-map 1 --max-updates 128 --save-every 32
```

Compare MAP01 runs with `behavior_eval.py --maps MAP01 --seeds 2 --ticks 4200`.
MAP01 is a development map in the training pool. Check transfer with `bench.py --split hard`
and reserved Freedoom maps such as MAP04.
