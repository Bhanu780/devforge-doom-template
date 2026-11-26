"""Reward shaping from env.privileged_state() (training only; never seen by the agent).

Mirrors the scorer's categories: exit (largest), progression (keys, doors), exploration
(unique areas), combat (kills, damage, death).
"""

from __future__ import annotations

import numpy as np

R = dict(
    exit=20.0,
    exit_health=0.05,  # per remaining HP at exit
    key=5.0,
    door=1.0,
    locked_door=4.0,
    cell=0.1,  # new 128x128-unit area visited
    kill=1.0,
    damage=-0.02,  # per HP lost
    death=-5.0,
    item=0.1,
    step=-0.002,
    use_press=-0.02,  # discourages USE spam (it bounces doors)
    attack_no_target=-0.003,
)
CELL = 128.0
PLAYER_HEIGHT = 56.0


class Shaper:
    def reset(self, env) -> None:
        s = env.privileged_state()
        self.prev = s
        self.cells = {self._cell(s)}
        lock = s["door_lock"]
        self.door_ids = np.flatnonzero(lock >= 0) if lock is not None else np.zeros(0, int)
        self.door_open = self._open(s)  # doors already open at spawn don't count
        self.door_ever = self.door_open.copy()
        self.prev_use = False
        self.stats = dict(ret=0.0, exit=0, death=0, kills=0, keys=0, doors=0, locked=0, cells=1, steps=0, dmg=0.0)

    @staticmethod
    def _cell(s):
        return (int(s["x"] // CELL), int(s["y"] // CELL))

    def _open(self, s):
        if s["sector_floor"] is None or len(self.door_ids) == 0:
            return np.zeros(len(self.door_ids), bool)
        return (s["sector_ceiling"][self.door_ids] - s["sector_floor"][self.door_ids]) >= PLAYER_HEIGHT

    def step(self, env, action, obs, terminated: bool) -> float:
        s = env.privileged_state()
        p = self.prev
        r = R["step"]
        st = self.stats
        if s["exit_reached"]:
            r += R["exit"] + R["exit_health"] * max(s["health"], 0)
            st["exit"] = 1
        if s["dead"]:
            r += R["death"]
            st["death"] = 1
        dk = s["kills"] - p["kills"]
        r += R["kill"] * dk
        st["kills"] += dk
        dd = max(0.0, s["damage_taken"] - p["damage_taken"])
        r += R["damage"] * dd
        st["dmg"] += dd
        nk = float(np.sum(s["keys"]) - np.sum(p["keys"]))
        r += R["key"] * nk
        st["keys"] += int(nk)
        r += R["item"] * max(0, s["items"] - p["items"])
        c = self._cell(s)
        if c not in self.cells and not terminated:
            self.cells.add(c)
            r += R["cell"]
            st["cells"] += 1
        if len(self.door_ids) and not terminated:
            now = self._open(s)
            new = now & ~self.door_ever
            if new.any():
                locked = s["door_lock"][self.door_ids[new]] > 0
                r += R["door"] * int((~locked).sum()) + R["locked_door"] * int(locked.sum())
                st["doors"] += int((~locked).sum())
                st["locked"] += int(locked.sum())
                self.door_ever |= now
        use = bool(action[7])
        if use and not self.prev_use:
            r += R["use_press"]
        self.prev_use = use
        if action[6] and not obs["visible_entities"]["mask"].any():
            r += R["attack_no_target"]
        self.prev = s
        st["ret"] += r
        st["steps"] += 1
        return float(r)
