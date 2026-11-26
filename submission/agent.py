"""Evaluation entry point (docs/CONTRACT.md §6).

Every action comes from the trained network. The FeatureBuilder only encodes observations
(odometry + memory map); it never chooses actions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS = SUBMISSION_DIR / "weights" / "policy.pt"
sys.path.insert(0, str(SUBMISSION_DIR))

from model.features import FeatureBuilder  # noqa: E402
from model.policy import DoomPolicy  # noqa: E402

TEMPERATURE = 0.7  # < 1 = less random than training, > 0 so fixed seeds can't trap us in a loop


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        torch.set_num_threads(1)
        self.action_space = action_space
        self.device = "cpu"  # a ~3 M parameter net at batch 1 is faster on CPU than via CUDA launches
        self.fb = FeatureBuilder()
        self.rng = np.random.default_rng(0)
        if not WEIGHTS.is_file():
            raise FileNotFoundError(f"Missing trained policy: {WEIGHTS}")
        ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        self.model = DoomPolicy(hidden=cfg.get("hidden", 512), use_rgb=cfg.get("use_rgb", True))
        self.model.load_state_dict({k: v.float() for k, v in ckpt["model"].items()})
        self.model.eval()
        self.temperature = TEMPERATURE
        self.reset()

    def reset(self) -> None:
        self.fb.reset()
        self.h = self.model.initial_state(1)
        self.last = np.zeros(9, np.int8)

    @torch.no_grad()
    def act(self, observation: dict) -> np.ndarray:
        feats = self.fb.step(observation)
        x = {k: torch.from_numpy(np.asarray(v)).unsqueeze(0).unsqueeze(0) for k, v in feats.items()}
        logits, _, self.h = self.model(x, self.h)
        p = torch.sigmoid(logits[0, 0] / self.temperature).numpy()
        action = (self.rng.random(9) < p).astype(np.int8)
        self.fb.note_action(action)
        return action
