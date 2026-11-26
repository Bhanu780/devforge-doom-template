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
from maps import build_pool
from train import DoomPolicy, FeatureBuilder, ROOT, RunningStd, stack, to_torch
from model.actions import mask_distant_punches, movement_logits, sample_exclusive, sample_exclusive_numpy


@torch.inference_mode()
def collect_episode(path, route, seed, use_rgb, *, student=None, rng=None, recovery_rate=0.0,
                    exclusive_movement=False, weapon_discipline=False):
    with closing(make_env(path, episode_ticks=2100, render_buffers=use_rgb)) as env:
        observation, _ = env.reset(seed=seed)
        features = FeatureBuilder()
        features.reset()
        inputs, actions = [], []
        waypoint = 0
        last_use = -12
        if student is not None:
            device = next(student.parameters()).device
            hidden = student.initial_state(1, device)
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
            else:
                action[0] = 1

            frame = features.step(observation)
            inputs.append(frame)
            actions.append(action)
            executed = action
            if student is not None:
                batch = to_torch({key: value[None, None] for key, value in frame.items()}, device)
                logits, _, hidden = student(batch, hidden)
                if weapon_discipline:
                    logits = mask_distant_punches(logits, batch)
                if rng.random() < recovery_rate:
                    decision = logits[0, 0].cpu()
                    if exclusive_movement:
                        executed = sample_exclusive_numpy(decision, rng)
                    else:
                        executed = (rng.random(9) < decision.sigmoid().numpy()).astype(np.int8)
            if executed[7]:
                last_use = len(actions) - 1
            features.note_action(executed)
            observation, _, terminated, truncated, _ = env.step(executed)
            if terminated or truncated:
                if env.privileged_state()["exit_reached"]:
                    return stack(inputs), np.asarray(actions, dtype=np.float32)
                return None
    return None


@torch.inference_mode()
def collect_combat(reference, seed, device, exclusive_movement=False, weapon_discipline=False):
    """Keep examples of the previous policy's combat actions on training maps."""
    scenarios = {"basic", "defend_the_center", "deadly_corridor", "rocket_basic"}
    episodes = []
    for wad, map_name, _ in build_pool("train"):
        if Path(wad).stem not in scenarios:
            continue
        with closing(make_env(wad, map_name, episode_ticks=768, render_buffers=reference.use_rgb)) as env:
            for offset in range(2):
                observation, _ = env.reset(seed=seed + offset)
                features = FeatureBuilder()
                features.reset()
                hidden = reference.initial_state(1, device)
                inputs, actions = [], []
                while True:
                    frame = features.step(observation)
                    batch = to_torch({key: value[None, None] for key, value in frame.items()}, device)
                    logits, _, hidden = reference(batch, hidden)
                    if weapon_discipline:
                        logits = mask_distant_punches(logits, batch)
                    decision = logits[0, 0]
                    probabilities = decision.sigmoid()
                    if exclusive_movement:
                        action = sample_exclusive(decision)
                        probabilities[:6] = movement_logits(decision).softmax(dim=-1)[:, 1:].reshape(6)
                    else:
                        action = torch.bernoulli(probabilities)
                    action = action.cpu().numpy().astype(np.int8)
                    inputs.append(frame)
                    actions.append(probabilities.cpu().numpy())
                    features.note_action(action)
                    observation, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
                episodes.append((stack(inputs), np.asarray(actions, dtype=np.float32)))
    return episodes


def fit(model, episodes, rng, epochs, device, exclusive_movement=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    batch_size, chunk_size = 6, 32
    for epoch in range(epochs):
        losses = []
        order = rng.permutation(len(episodes))
        for group_start in range(0, len(order), batch_size):
            group = [episodes[index] for index in order[group_start:group_start + batch_size]]
            hidden = model.initial_state(len(group), device)
            for start in range(0, max(len(actions) for _, actions in group), chunk_size):
                lengths = [max(0, min(chunk_size, len(actions) - start)) for _, actions in group]
                size = max(lengths)
                mask = torch.arange(size, device=device)[:, None] < torch.tensor(lengths, device=device)
                observations = group[0][0]
                batch = {name: np.zeros((size, len(group), *values.shape[1:]), dtype=values.dtype)
                         for name, values in observations.items()}
                targets = np.zeros((size, len(group), 9), dtype=np.float32)
                for index, ((frames, actions), length) in enumerate(zip(group, lengths)):
                    for name in batch:
                        batch[name][:length, index] = frames[name][start:start + length]
                    targets[:length, index] = actions[start:start + length]
                target = torch.from_numpy(targets).to(device)
                logits, _, hidden = model(to_torch(batch, device), hidden)
                hidden = hidden.detach()
                # Use is rare. Give the demonstrated door and exit presses extra weight.
                weight = torch.ones_like(target)
                weight[:, :, 7] += 4 * (target[:, :, 7] == 1)
                observed_logits, labels, sample_weight = logits[mask], target[mask], weight[mask]
                if exclusive_movement:
                    pairs = labels[:, :6].reshape(-1, 3, 2)
                    neither = (1 - pairs.sum(dim=-1, keepdim=True)).clamp_min(0)
                    move_targets = torch.cat((neither, pairs), dim=-1)
                    move_loss = -(move_targets * movement_logits(observed_logits).log_softmax(dim=-1)).sum(dim=-1)
                    other_loss = F.binary_cross_entropy_with_logits(
                        observed_logits[:, 6:], labels[:, 6:], weight=sample_weight[:, 6:], reduction="none"
                    )
                    loss = (move_loss.sum(dim=-1) + other_loss.sum(dim=-1)).mean() / 6
                else:
                    loss = F.binary_cross_entropy_with_logits(observed_logits, labels, weight=sample_weight)
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
    parser.add_argument("--combat-reference", type=Path, help="retain examples from a prior combat policy")
    parser.add_argument("--recovery-rate", type=float, default=0.0,
                        help="probability of letting the initial policy act in a recovery rollout")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if not 0 <= args.recovery_rate <= 1:
        parser.error("--recovery-rate must be between 0 and 1")
    if args.recovery_rate and args.init is None:
        parser.error("--recovery-rate requires --init")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.init, map_location="cpu", weights_only=True) if args.init else None
    config = checkpoint["config"] if checkpoint else {"hidden": 512, "use_rgb": False}
    model = DoomPolicy(hidden=config["hidden"], use_rgb=config["use_rgb"]).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    run = ROOT / "training" / "runs" / args.run
    episodes = []
    recovered = []
    model.eval()
    for layout in range(len(PRACTICE_LAYOUTS)):
        successes = 0
        for variation in range(4):
            path, route = build_practice_map(run / "maps", layout, variation)
            episode = collect_episode(path, route, args.seed + variation, model.use_rgb)
            if episode is not None:
                episodes.append(episode)
                successes += 1
            if args.recovery_rate and variation % 2 == layout % 2:
                recovery = collect_episode(path, route, args.seed + variation, model.use_rgb,
                                           student=model, rng=rng, recovery_rate=args.recovery_rate,
                                           exclusive_movement=config.get("exclusive_movement", False),
                                           weapon_discipline=config.get("weapon_discipline", False))
                if recovery is not None:
                    recovered.append(recovery)
        print(f"layout {layout}: {successes}/4 completed demonstrations", flush=True)
    if not episodes:
        raise RuntimeError("No completed demonstrations; check the practice maps and waypoint route")
    if args.recovery_rate:
        print(f"adding {len(recovered)}/{2 * len(PRACTICE_LAYOUTS)} completed recovery episodes", flush=True)
        episodes.extend(recovered)
    if args.combat_reference:
        prior = torch.load(args.combat_reference, map_location="cpu", weights_only=True)
        if prior["config"]["use_rgb"] != model.use_rgb:
            parser.error("--combat-reference must use the same RGB setting as the initial policy")
        reference = DoomPolicy(hidden=prior["config"]["hidden"], use_rgb=prior["config"]["use_rgb"]).to(device)
        reference.load_state_dict(prior["model"])
        combat = collect_combat(reference.eval(), args.seed, device,
                                exclusive_movement=prior["config"].get("exclusive_movement", False),
                                weapon_discipline=prior["config"].get("weapon_discipline", False))
        print(f"adding {len(combat)} combat episodes from the reference policy", flush=True)
        episodes.extend(combat)
    print(f"training on {len(episodes)} episodes, {sum(len(actions) for _, actions in episodes)} decisions", flush=True)
    fit(model.train(), episodes, rng, args.epochs, device, config.get("exclusive_movement", False))

    # Reuse the PPO trainer's resume format with a fresh optimizer and reward scale.
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4, eps=1e-5)
    reward_scale = RunningStd()
    torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(), "update": 0, "env_steps": 0,
                "ret_std": (reward_scale.n, reward_scale.mean, reward_scale.m2), "config": config},
               run / "latest.pt")


if __name__ == "__main__":
    main()
