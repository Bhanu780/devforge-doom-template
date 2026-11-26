"""Train a neural policy from automatic demonstrations on team-authored maps.

The demonstration controller runs only during data collection. The policy learns
from the official observations and previous actions through the same features as
DoomAgent.
"""

import argparse
from contextlib import closing
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from doom_env import make_env
from keycard_curriculum import PRACTICE_LAYOUTS, build_practice_map
from train import DoomPolicy, FeatureBuilder, ROOT, RunningStd, stack, to_torch


def collect_episode(path, route, seed, use_rgb):
    with closing(make_env(path, episode_ticks=2100, render_buffers=use_rgb)) as env:
        observation, _ = env.reset(seed=seed)
        features = FeatureBuilder()
        features.reset()
        inputs, actions = [], []
        waypoint = 0
        last_use = -12
        while waypoint < len(route):
            state = env.privileged_state()
            (target_x, target_y), purpose = route[waypoint]
            dx, dy = target_x - state["x"], target_y - state["y"]
            distance = math.hypot(dx, dy)
            if purpose == "walk" and distance < 24:
                waypoint += 1
                continue
            if purpose == "door" and state["sector_ceiling"][state["door_lock"] > 0].max() >= 90:
                waypoint += 1
                continue

            heading = math.degrees(math.atan2(dy, dx))
            error = (heading - state["angle"] + 180) % 360 - 180
            action = np.zeros(9, dtype=np.int8)
            if abs(error) > 9:
                action[4 if error > 0 else 5] = 1
            elif purpose != "walk" and distance < 52:
                if len(actions) - last_use >= 12:
                    action[7] = 1
                    last_use = len(actions)
            else:
                action[0] = 1

            inputs.append(features.step(observation))
            actions.append(action)
            features.note_action(action)
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                if env.privileged_state()["exit_reached"]:
                    return stack(inputs), np.asarray(actions, dtype=np.float32)
                return None
    return None


def fit(model, episodes, rng, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for epoch in range(epochs):
        losses = []
        for index in rng.permutation(len(episodes)):
            observations, actions = episodes[index]
            hidden = model.initial_state(1, device)
            for start in range(0, len(actions), 24):
                batch = slice(start, start + 24)
                inputs = to_torch({k: v[batch, None] for k, v in observations.items()}, device)
                target = torch.from_numpy(actions[batch]).to(device)
                logits, _, hidden = model(inputs, hidden)
                hidden = hidden.detach()
                # Use is rare. Give successful door and exit presses extra weight.
                weight = torch.ones_like(target)
                weight[:, 7] += 4 * target[:, 7]
                loss = F.binary_cross_entropy_with_logits(logits[:, 0], target, weight=weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                losses.append(loss.item())
        print(f"imitation epoch {epoch + 1}/{epochs} loss {np.mean(losses):.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="keycard_imitation")
    parser.add_argument("--init", type=Path, help="optional initial policy checkpoint")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.init, map_location="cpu", weights_only=True) if args.init else None
    config = checkpoint["config"] if checkpoint else {"hidden": 512, "use_rgb": False}
    model = DoomPolicy(**config).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    run = ROOT / "training" / "runs" / args.run
    episodes = []
    for layout in range(len(PRACTICE_LAYOUTS)):
        successes = 0
        for variation in range(4):
            path, route = build_practice_map(run / "maps", layout, variation)
            episode = collect_episode(path, route, args.seed + variation, model.use_rgb)
            if episode is not None:
                episodes.append(episode)
                successes += 1
        print(f"layout {layout}: {successes}/4 completed demonstrations", flush=True)
    if not episodes:
        raise RuntimeError("No completed demonstrations; check the practice maps and waypoint route")
    print(f"training on {len(episodes)} episodes, {sum(len(actions) for _, actions in episodes)} decisions", flush=True)
    fit(model, episodes, rng, args.epochs, device)

    # Reuse the PPO trainer's resume format with a fresh optimizer and reward scale.
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4, eps=1e-5)
    reward_scale = RunningStd()
    torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(), "update": 0, "env_steps": 0,
                "ret_std": (reward_scale.n, reward_scale.mean, reward_scale.m2), "config": config},
               run / "latest.pt")


if __name__ == "__main__":
    main()
