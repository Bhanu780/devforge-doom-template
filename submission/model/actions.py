"""Distributions for Doom's three mutually exclusive movement button pairs."""

import numpy as np
import torch
from torch.distributions import Categorical


def movement_logits(logits: torch.Tensor) -> torch.Tensor:
    """Give each pair a categorical logit for neither, first, and second."""
    pairs = logits[..., :6].reshape(*logits.shape[:-1], 3, 2)
    return torch.cat((torch.zeros_like(pairs[..., :1]), pairs), dim=-1)


def sample_exclusive(logits: torch.Tensor) -> torch.Tensor:
    moves = Categorical(logits=movement_logits(logits), validate_args=False).sample()
    buttons = torch.stack((moves == 1, moves == 2), dim=-1).flatten(-2).to(logits.dtype)
    return torch.cat((buttons, torch.bernoulli(logits[..., 6:].sigmoid())), dim=-1)


def sample_exclusive_numpy(logits: torch.Tensor, rng: np.random.Generator) -> np.ndarray:
    probabilities = movement_logits(logits).softmax(dim=-1).numpy()
    choices = (rng.random(3)[:, None] >= np.cumsum(probabilities[:, :2], axis=-1)).sum(axis=-1)
    buttons = np.stack((choices == 1, choices == 2), axis=-1).reshape(6)
    other = rng.random(3) < logits[6:].sigmoid().numpy()
    return np.concatenate((buttons, other)).astype(np.int8)


def mask_distant_punches(logits: torch.Tensor, features: dict) -> torch.Tensor:
    """Mask fist attacks when visible breakable targets are all beyond melee reach."""
    entity_types = features["ent_type"]
    targets = (features["ent_mask"] > 0) & (((entity_types >= 1) & (entity_types <= 3)) | (entity_types == 10))
    # Distance is normalized by 1024. Leave room for target radius and four ticks of movement.
    nearby = (targets & (features["ent_feat"][..., 0] <= 128 / 1024)).any(dim=-1)
    fist = (features["extra"][..., 2] * 7).round() == 1
    block = fist & targets.any(dim=-1) & ~nearby
    attack = logits[..., 6].masked_fill(block, -100)
    return torch.cat((logits[..., :6], attack.unsqueeze(-1), logits[..., 7:]), dim=-1)
