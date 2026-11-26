"""Recurrent actor-critic for doom_env. ~3 M parameters, < 5 ms per step on one CPU thread.

Inputs come from features.FeatureBuilder. Each modality has its own encoder:
  rays     -> circular 1D conv over the 64 ego-centric rays (hit-type embedding + distance)
  entities -> per-entity MLP + masked self-attention + pooling (handles 0..32 entities)
  map      -> small CNN over the 32x32 ahead-up crop of the memory map
  rgb      -> small CNN over 60x80 RGB+depth (exit switches/signs are only visible here)
  extra    -> vitals, previous action and own counters
The fused vector drives a GRU; heads output 9 independent Bernoulli logits and a value.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import CROP, EXTRA_DIM, MAP_CHANNELS, NUM_RAYS

NUM_BUTTONS = 9
NUM_HIT_TYPES = 6
NUM_ENT_TYPES = 12


class RayEncoder(nn.Module):
    def __init__(self, out=256):
        super().__init__()
        self.emb = nn.Embedding(NUM_HIT_TYPES, 8)
        self.c1 = nn.Conv1d(10, 32, 5)
        self.c2 = nn.Conv1d(32, 64, 5, stride=2)
        self.c3 = nn.Conv1d(64, 64, 3, stride=2)
        self.fc = nn.Linear(64 * 16, out)

    @staticmethod
    def _circ(x, k):  # circular padding keeps "behind" next to "left/right"
        p = k // 2
        return torch.cat([x[..., -p:], x, x[..., :p]], dim=-1)

    def forward(self, dist, htype):
        d = dist.unsqueeze(1)
        x = torch.cat([self.emb(htype).transpose(1, 2), d, (d < 0.05).float()], 1)  # (B,10,64)
        x = F.relu(self.c1(self._circ(x, 5)))
        x = F.relu(self.c2(self._circ(x, 5)))[..., :32]
        x = F.relu(self.c3(self._circ(x, 3)))[..., :16]
        return F.relu(self.fc(x.flatten(1)))


class EntityEncoder(nn.Module):
    def __init__(self, d=64, out=128):
        super().__init__()
        self.emb = nn.Embedding(NUM_ENT_TYPES, 16)
        self.inp = nn.Sequential(nn.Linear(5 + 16, d), nn.ReLU(), nn.Linear(d, d))
        self.null = nn.Parameter(torch.zeros(1, 1, d))  # always-present token: no all-masked rows
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.out = nn.Linear(2 * d, out)

    def forward(self, feat, etype, mask):
        b = feat.shape[0]
        x = self.inp(torch.cat([feat, self.emb(etype)], -1))  # (B,32,d)
        x = torch.cat([self.null.expand(b, -1, -1), x], 1)
        m = torch.cat([torch.ones(b, 1, device=mask.device), mask], 1) > 0.5
        h = self.n1(x + self.attn(x, x, x, key_padding_mask=~m, need_weights=False)[0])
        h = self.n2(h + self.ff(h))
        mf = m.unsqueeze(-1).float()
        mean = (h * mf).sum(1) / mf.sum(1)
        mx = h.masked_fill(~m.unsqueeze(-1), -1e4).max(1).values
        return F.relu(self.out(torch.cat([mean, mx], -1)))


def conv_stack(cin, chans, out_hw, out):
    layers, c = [], cin
    for k in chans:
        layers += [nn.Conv2d(c, k, 3, stride=2, padding=1), nn.ReLU()]
        c = k
    return nn.Sequential(*layers, nn.Flatten(), nn.Linear(c * out_hw, out), nn.ReLU())


class DoomPolicy(nn.Module):
    def __init__(self, hidden=512, use_rgb=True):
        super().__init__()
        self.use_rgb = use_rgb
        self.rays = RayEncoder(256)
        self.ents = EntityEncoder(64, 128)
        self.map = conv_stack(len(MAP_CHANNELS), (32, 64, 64), (CROP // 8) ** 2, 256)
        self.rgb = conv_stack(4, (16, 32, 64, 64), 4 * 5, 256)  # 60x80 -> 30x40 -> 15x20 -> 8x10 -> 4x5
        self.extra = nn.Sequential(nn.Linear(EXTRA_DIM, 128), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(256 + 128 + 256 + 256 + 128, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden)
        self.pi = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, NUM_BUTTONS))
        self.v = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, 1))
        self.hidden = hidden
        with torch.no_grad():  # start with few buttons pressed, forward slightly favoured
            self.pi[-1].weight.mul_(0.01)
            self.pi[-1].bias.copy_(torch.tensor([0.5, -2, -1.5, -1.5, -1, -1, -1, -2, -4.0]))

    def encode(self, x: dict) -> torch.Tensor:
        """x: dict of (B, ...) tensors from FeatureBuilder outputs."""
        parts = [
            self.rays(x["ray_dist"], x["ray_type"]),
            self.ents(x["ent_feat"], x["ent_type"], x["ent_mask"]),
            self.map(x["map"]),
            self.rgb(x["rgb"].float() / 255.0) if self.use_rgb else torch.zeros(x["extra"].shape[0], 256, device=x["extra"].device),
            self.extra(x["extra"]),
        ]
        return self.fuse(torch.cat(parts, -1))

    def forward(self, x: dict, h: torch.Tensor, done: torch.Tensor | None = None):
        """Sequence forward. x tensors are (T, B, ...); h is (1, B, H); done (T, B) resets h
        *before* step t. Returns logits (T,B,9), value (T,B), h."""
        T, B = x["extra"].shape[:2]
        flat = {k: v.reshape(T * B, *v.shape[2:]) for k, v in x.items()}
        z = self.encode(flat).reshape(T, B, -1)
        if done is None:
            o, h = self.gru(z, h)
        elif T == 1:
            o, h = self.gru(z, h * (1.0 - done[0]).view(1, B, 1))
        else:
            # Batch uninterrupted stretches in one GRU call, preserving episode resets.
            boundaries = torch.nonzero(done.any(dim=1), as_tuple=True)[0].tolist()
            outs = []
            start = 0
            for boundary in boundaries + [T]:
                if boundary > start:
                    output, h = self.gru(z[start:boundary], h)
                    outs.append(output)
                if boundary < T:
                    h = h * (1.0 - done[boundary]).view(1, B, 1)
                start = boundary
            o = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
        logits = self.pi(o)
        if "act_mask" in x:  # masked buttons: probability ~3e-7, identical in training and evaluation
            logits = logits.masked_fill(x["act_mask"] < 0.5, -15.0)
        return logits, self.v(o).squeeze(-1), h

    def initial_state(self, batch: int, device="cpu") -> torch.Tensor:
        return torch.zeros(1, batch, self.hidden, device=device)
