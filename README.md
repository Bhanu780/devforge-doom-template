# MapMind: a DOOM agent with path-integrated spatial memory

**Yugaantar 2026 · DevForge · Track 1 "Can It Play DOOM?"**

The agent plays unseen DOOM levels from the official ego-centric observations only. Its distinctive
part is a **self-built memory map**: the wrapper gives no position or heading, so the agent
dead-reckons its own pose (Doom's exact turn rule plus player-frame velocity) and draws every step's
64 raycasts and visible entities into a top-down grid. Walls, doors, locked doors, keys, items,
enemies and visited cells stay remembered for the whole map, which is what keycard backtracking needs.
Every action comes from a trained recurrent network; the map only encodes observations.

| Part | Where | What it does |
| --- | --- | --- |
| Odometry + memory map | `submission/model/features.py` | Heading from our own turn buttons (1.76 deg/tic for 5 tics, then 3.52), position from velocity, static-scan check when pushing into walls; 8-channel 192x192 grid, 32x32 ahead-up crop for the policy |
| Policy | `submission/model/policy.py` | Circular 1D conv over rays, masked self-attention over 0-32 entities, map CNN, RGB+depth CNN (exit switches), GRU 512, 9 Bernoulli buttons + value. 3.4 M params, ~2-3 ms/step on 1 CPU thread |
| Agent | `submission/agent.py` | Loads weights once, clears map + GRU on `reset()`, samples at temperature 0.7, never crashes |
| Training | `training/train.py` | Our own recurrent PPO (multiprocess envs, GPU learner, truncated BPTT, GAE, return normalisation) |
| Rewards + curiosity | `training/rewards.py`, `train.py` | Scorer-aligned shaping (exit, keys, doors, locked doors, new areas, kills, damage, death) + intrinsic count-based novelty on the agent's own map |
| Maps + curriculum | `training/maps.py` | Freedoom 2 MAP01-32, ViZDoom scenarios, test WADs; small maps first, full levels later; 7 maps held out |
| Benchmark | `training/bench.py` | Runs the real `DoomAgent` on held-out maps and reports exits, keys, doors, coverage, kills |

Measured: heading error 0 deg while alive; position drift 9-63 units after 300 random steps;
validator step latency p50 1.6 ms (Windows) / 3.2 ms (Linux) against a 40 ms budget.

**Train:** see [training/README.md](training/README.md). **Brief:** [brief/Project_Brief.pdf](brief/Project_Brief.pdf).

---

## Template information (from the official starter)

# devforge-doom-template

Starter template for **Yugaantar 2026 · DevForge — Track 1: "Can It Play DOOM?"**

> For decades, *"Can it run DOOM?"* was the ultimate hardware test. Now: **can your neural network
> actually play DOOM?** Train an agent that plays unseen levels **zero-shot**: survive ambushes,
> navigate multi-room layouts, find keycards, unlock doors and reach the exit.

No dataset is provided. Your agent learns by interacting with the engine.

## Quickstart

1. Click **"Use this template"** on GitHub to create your team's (public) repository.
2. Set up the environment (Python 3.13, [uv](https://docs.astral.sh/uv/)):

   ```bash
   uv sync                 # installs vizdoom, gymnasium and the official wrapper (env/) in editable mode
   uv run python validate.py
   ```

   Or use the official image, which is exactly what the scorer runs:

   ```bash
   docker build -t devforge-doom .            # or: docker pull <official image tag, announced>
   python validate.py --docker --image devforge-doom   # installs requirements.txt, then runs sandboxed like the scorer
   ```

3. Replace the random baseline in `submission/agent.py` with your trained agent.
4. Put weights in `submission/weights/` (Git LFS) and training code in `training/`.
5. Run `python validate.py --docker` and submit your repo URL.

**Platforms:** Linux (x86-64/ARM), macOS on Apple Silicon (macOS 15+) and Windows (x86-64) all
work with `uv sync` and `validate.py`. Intel Macs have no `vizdoom` 1.3.3 wheel: use Docker or a
Linux machine. Evaluation runs on Linux, so `validate.py --docker` is the closest check.

## Repository layout

```
env/                  official ViZDoom Gymnasium wrapper (doom_env): spaces, sensors, protocol, limits
maps/                 one small test map (NOT an evaluation map); bring your own for training
submission/agent.py   your DoomAgent: the evaluation entry point
submission/weights/   model weights (Git LFS)
training/             your training pipeline
validate.py           run before submitting
tests/                tests for the wrapper and validator (uv run pytest)
docs/                 contract, rules, submission guide, evaluation details
```

## Using the environment

```python
from doom_env import make_env

env = make_env("maps/two_rooms.wad")     # test map; use your own maps for training
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Observations are a dict of `rgb`, `depth`, `vitals`, `rays` and `visible_entities`, all
ego-centric and partially observable. See [docs/CONTRACT.md](docs/CONTRACT.md).
The wrapper's `reward` is a basic default for convenience. Shape your own reward for training.

**Compute tip:** training only on raw pixels is expensive. The `rays`, `visible_entities` and
`vitals` signals run thousands of steps per second on a CPU. Hybrid pixel + spatial models are welcome.

## Hardware at evaluation

One NVIDIA **RTX 3090 (24 GB)** shared between evaluation workers. Each agent gets **2 GB VRAM,
8 GB RAM and 1 CPU thread**, so keep models compact and inference fast
(see [hardware](docs/EVALUATION.md#hardware) and [timeouts](docs/EVALUATION.md#timeouts)).
`validate.py` runs your agent under the same CPU/RAM limits and reports GPU memory.

## Documentation

| Doc | What's inside |
| --- | --- |
| [docs/CONTRACT.md](docs/CONTRACT.md) | Observation/action spaces and the `DoomAgent` interface |
| [docs/RULES.md](docs/RULES.md) | Ground rules: what's allowed, what gets you disqualified |
| [docs/SUBMISSION.md](docs/SUBMISSION.md) | Submission checklist, Git LFS, 1 GB limit, `requirements.txt` |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Map tiers, scoring categories, timeouts, replays |

## Rules in one minute

- Actions must come from **trained neural parameters**: no rule trees, no `A*`/Dijkstra, no `atan2` aimbots,
  no unmodified off-the-shelf SB3.
- Change anything you want **during training**. Evaluation uses the **vanilla engine**, the
  official wrapper and **unseen maps**.
- Create the repo during the hackathon. Libraries are fine; the core architecture and training must be yours.
