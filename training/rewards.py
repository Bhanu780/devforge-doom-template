"""Reward shaping from env.privileged_state() (training only; never seen by the agent).

Mirrors the scorer's categories: exit (largest), progression (keys, doors), exploration
(unique areas), combat (kills, damage, death).
"""

from __future__ import annotations

import math

import numpy as np
from vizdoom import GameVariable

import navfield

R = dict(
    exit=20.0,
    exit_health=0.05,  # per remaining HP at exit
    key=5.0,
    door=1.0,
    locked_door=4.0,
    cell=0.1,  # new 128x128-unit area visited
    kill=0.3,  # was 1.0: 20-40 kills per map out-paid the exit, so the bot farmed fights
    damage=-0.02,  # per HP lost
    death=-5.0,
    item=0.1,
    step=-0.002,
    use_press=-0.02,  # discourages USE spam (it bounces doors)
    attack_no_target=-0.02,  # ATTACK with no enemy near the crosshair (was -0.003: the bot emptied its pistol into walls)
    ammo_wasted=-0.05,  # per round of ammo spent while no enemy is visible at all
    stall=-0.02,  # per step while the player has moved < STALL_DIST units over the last STALL_STEPS steps
    fists_vs_far=-0.02,
    hit_damage=0.008,  # credit landed damage while keeping exits more valuable
    # Potential-based shaping on walking distance to the exit (through missing keys), see navfield.py:
    # +k per map unit closer, -k per unit farther, 0 standing still; telescopes, so loops can't farm it.
    exit_dist=0.002,
    exit_dist_clip=0.5,  # per step; teleports and unmodelled lifts make the distance jump
)
ENEMY_TYPES = (1, 2, 3)  # doom_env.spaces.ENTITY_TYPES: enemy_melee, enemy_hitscan, enemy_projectile
SLOT_MELEE = 1  # vitals[2]: 1 fist/chainsaw, 2 pistol, 3 shotgun, ...
MELEE_RANGE = 96.0  # same as submission/model/features.py
AIM_SIN = 0.2  # |sin(rel_angle)| below this counts as "near the crosshair" (~11.5 deg)
CELL = 128.0
PLAYER_HEIGHT = 56.0
STALL_STEPS = 16  # ~1.8 s of game time
STALL_DIST = 48.0  # map units; walking speed covers this in under 2 steps


class Shaper:
    def __init__(self, weapon_discipline: bool = False, loaded_switch_cost: float = 0.05):
        self.weapon_discipline = weapon_discipline
        self.loaded_switch_cost = loaded_switch_cost

    def reset(self, env, observation=None) -> None:
        s = env.privileged_state()
        self.prev = s
        self.cells = {self._cell(s)}
        lock = s["door_lock"]
        self.door_ids = np.flatnonzero(lock >= 0) if lock is not None else np.zeros(0, int)
        self.door_open = self._open(s)  # doors already open at spawn don't count
        self.door_ever = self.door_open.copy()
        self.prev_use = False
        self.prev_switch = False
        self.prev_observation = observation
        if self.weapon_discipline:
            self.damage_dealt = env._game.get_game_variable(GameVariable.DAMAGECOUNT)
        self.stats = dict(ret=0.0, exit=0, death=0, kills=0, keys=0, doors=0, locked=0, cells=1, steps=0, dmg=0.0, prog=0.0)
        self.nav = navfield.for_env(env) if R["exit_dist"] else None
        self.d = self.d0 = self.dmin = self._exit_distance(s)
        self.trail = [(s["x"], s["y"])]  # recent positions, for the stall penalty

    def _exit_distance(self, s) -> float:
        if self.nav is None:
            return math.inf
        return self.nav.distance(s["x"], s["y"], navfield.colors_held(s["keys"]))

    @staticmethod
    def _cell(s):
        return (int(s["x"] // CELL), int(s["y"] // CELL))

    def _open(self, s):
        if s["sector_floor"] is None or len(self.door_ids) == 0:
            return np.zeros(len(self.door_ids), bool)
        return (s["sector_ceiling"][self.door_ids] - s["sector_floor"][self.door_ids]) >= PLAYER_HEIGHT

    @staticmethod
    def _enemy_in_sights(obs) -> bool:
        ent = obs["visible_entities"]
        enemy = (np.asarray(ent["mask"]) > 0) & np.isin(np.asarray(ent["entity_type"]), ENEMY_TYPES)
        return bool(np.any(enemy & (np.abs(np.asarray(ent["features"])[:, 1]) < AIM_SIN)))

    @staticmethod
    def _enemy_visible(obs) -> bool:
        ent = obs["visible_entities"]
        return bool(np.any((np.asarray(ent["mask"]) > 0) & np.isin(np.asarray(ent["entity_type"]), ENEMY_TYPES)))

    @staticmethod
    def _fists_vs_far(obs) -> bool:
        vit = np.asarray(obs["vitals"])
        if int(vit[2]) != SLOT_MELEE or (vit[4] <= 0 and vit[5] <= 0):  # no bullets or shells: fists are all we have
            return False
        ent = obs["visible_entities"]
        enemy = (np.asarray(ent["mask"]) > 0) & np.isin(np.asarray(ent["entity_type"]), ENEMY_TYPES)
        d = np.asarray(ent["features"])[enemy, 0] * 1024.0
        return bool(d.size and d.min() >= MELEE_RANGE)

    @staticmethod
    def _gate_ahead(obs) -> bool:
        rays = obs["rays"]
        forward = [63, 0, 1]
        distance = np.asarray(rays["distance"])[forward]
        hit_type = np.asarray(rays["hit_type"])[forward]
        return bool(np.any((distance <= 64 / 1024) & np.isin(hit_type, (2, 3))))

    def step(self, env, action, obs, terminated: bool) -> float:
        s = env.privileged_state()
        p = self.prev
        r = R["step"]
        st = self.stats
        before = self.prev_observation if self.prev_observation is not None else obs
        landed = 0.0
        if self.weapon_discipline:
            dealt = env._game.get_game_variable(GameVariable.DAMAGECOUNT)
            landed = max(0.0, dealt - self.damage_dealt)
            self.damage_dealt = dealt
            r += R["hit_damage"] * landed
        if s["exit_reached"]:
            r += R["exit"] + R["exit_health"] * max(s["health"], 0)
            st["exit"] = 1
            st["prog"] = 1.0
        elif not terminated and self.nav is not None:
            d = self._exit_distance(s)
            if math.isfinite(d):
                if math.isfinite(self.d):  # skip steps where the distance was undefined; keeps telescoping
                    r += float(np.clip(R["exit_dist"] * (self.d - d), -R["exit_dist_clip"], R["exit_dist_clip"]))
                self.d = d
                self.dmin = min(self.dmin, d)
                if math.isfinite(self.d0) and self.d0 > 0:
                    st["prog"] = max(0.0, 1.0 - self.dmin / self.d0)
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
        if use and not self.prev_use and not s["exit_reached"] and not self._gate_ahead(before):
            r += R["use_press"]
        self.prev_use = use
        if action[6] and not landed and not self._enemy_in_sights(before):
            r += R["attack_no_target"]
        if self.weapon_discipline:
            switch = bool(action[8])
            if switch and not self.prev_switch:
                r -= 0.01
                if before["vitals"][3] > 0 and self._enemy_visible(before):
                    r -= self.loaded_switch_cost
            self.prev_switch = switch
        if self._fists_vs_far(obs):
            r += R["fists_vs_far"]
            st["fists"] = st.get("fists", 0) + 1
        # Stall: pinned against a wall or jittering in place for ~2 s (the bot spent the last 40 s of a map
        # turning left/right at one sign). Potential shaping and curiosity pay 0 there, so it needs a cost.
        opening = bool(self.door_ids.size and not terminated and
                       np.any(s["sector_ceiling"][self.door_ids] > p["sector_ceiling"][self.door_ids]))
        self.trail.append((s["x"], s["y"]))
        if len(self.trail) > STALL_STEPS:
            self.trail.pop(0)
            xs, ys = zip(*self.trail)
            if max(xs) - min(xs) < STALL_DIST and max(ys) - min(ys) < STALL_DIST and not terminated \
                    and not opening and not self._enemy_visible(obs):  # holding position to fight is fine
                r += R["stall"]
                st["stall"] = st.get("stall", 0) + 1
        # Ammo spent with no enemy in view at all (rounds of the selected weapon, from the observation).
        vit = np.asarray(obs["vitals"])
        prior_vitals = before["vitals"]
        ammo_spent = float(prior_vitals[3] - vit[3])
        if vit[2] == prior_vitals[2] and ammo_spent > 0 and not landed and not self._enemy_visible(before):
            r += R["ammo_wasted"] * ammo_spent
            st["wasted"] = st.get("wasted", 0) + int(ammo_spent)
        self.prev_observation = obs
        self.prev = s
        st["ret"] += r
        st["steps"] += 1
        return float(r)
