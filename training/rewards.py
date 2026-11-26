"""Reward shaping from env.privileged_state() (training only; never seen by the agent).

Mirrors the scorer's categories: exit (largest), progression (keys, doors), exploration
(unique areas), combat (kills, damage, death).
"""

from __future__ import annotations

import math

import numpy as np

import navfield

R = dict(
    exit=20.0,
    exit_health=0.05,  # per remaining HP at exit
    key=5.0,
    door=1.0,
    locked_door=4.0,
    cell=0.15,  # new 128x128-unit area visited (was 0.1)
    revisit=-0.03,  # per entry into a 128-unit area already entered twice, x min(entries - 2, 4): stops backtracking loops
    kill=0.3,  # was 1.0: 20-40 kills per map out-paid the exit, so the bot farmed fights
    damage=-0.03,  # per HP lost, x (1 + 2 * missing-health fraction): 0.03/HP at full health, ~0.07/HP below 40 HP
    death=-8.0,  # was -5: dying early also costs the scorer's survival points
    health_gain=0.04,  # per HP healed, x (1 + 2 * missing-health fraction before the pickup); medkits never moved items
    armor_gain=0.01,  # per armour point picked up
    seek_health=0.002,  # below SEEK_HP: per map unit closer to the nearest visible health pickup (clipped per step)
    under_fire_strafe=0.004,  # per step strafing while hit in the last UNDER_FIRE steps (dodging instead of face-tanking)
    item=0.1,
    step=-0.002,
    use_press=-0.02,  # discourages USE spam (it bounces doors)
    attack_no_target=-0.02,  # ATTACK with no enemy near the crosshair (was -0.003: the bot emptied its pistol into walls)
    ammo_wasted=-0.05,  # per round of ammo spent while no enemy is visible at all
    stall=-0.02,  # per step while the player has moved < STALL_DIST units over the last STALL_STEPS steps
    fists_with_ammo=-0.03,  # per step on fists while bullets or shells remain (was fists_vs_far, only for distant enemies)
    bump=-0.01,  # per step holding FORWARD but moving < BUMP_DIST (stairs, angled walls), no enemy in view
    jitter=-0.01,  # turn direction reversed from the previous step with no enemy in view (wall-scrape dithering)
    # Potential-based shaping on walking distance to the exit (through missing keys), see navfield.py:
    # +k per map unit closer, -k per unit farther, 0 standing still; telescopes, so loops can't farm it.
    exit_dist=0.002,
    exit_dist_clip=0.5,  # per step; teleports and unmodelled lifts make the distance jump  # per step holding fists while a gun has ammo and an enemy is out of reach
)
ENEMY_TYPES = (1, 2, 3)  # doom_env.spaces.ENTITY_TYPES: enemy_melee, enemy_hitscan, enemy_projectile
SLOT_MELEE = 1  # vitals[2]: 1 fist/chainsaw, 2 pistol, 3 shotgun, ...
MELEE_RANGE = 96.0  # same as submission/model/features.py
AIM_SIN = 0.2  # |sin(rel_angle)| below this counts as "near the crosshair" (~11.5 deg)
CELL = 128.0
PLAYER_HEIGHT = 56.0
STALL_STEPS = 16  # ~1.8 s of game time
STALL_DIST = 48.0  # map units; walking speed covers this in under 2 steps
BUMP_DIST = 6.0  # map units per step (4 tics); full walking speed is ~30
SEEK_HP = 60.0
UNDER_FIRE = 12  # steps (~1.4 s) after taking damage
ENT_HEALTH = 5
# stall + bump + jitter + revisit together never cost more than this per episode. Uncapped, ~200 stuck steps
# would cost more than dying (-8), and the cheapest way out of a dead end would be suicide.
MOVE_PENALTY_CAP = 4.0


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
        self.stats = dict(ret=0.0, exit=0, death=0, kills=0, keys=0, doors=0, locked=0, cells=1, steps=0, dmg=0.0, prog=0.0)
        self.nav = navfield.for_env(env) if R["exit_dist"] else None
        self.d = self.d0 = self.dmin = self._exit_distance(s)
        self.trail = [(s["x"], s["y"])]  # recent positions, for the stall penalty
        self.prev_ammo = None
        self.entries = {self._cell(s): 1}  # 128-unit area -> times entered this episode
        self.last_cell = self._cell(s)
        self.prev_turn = 0
        self.since_hit = 10**6
        self.prev_armor = None
        self.health_d = math.inf
        self.move_pen = 0.0

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
    def _fists_with_ammo(obs) -> bool:
        vit = np.asarray(obs["vitals"])
        return int(vit[2]) == SLOT_MELEE and (vit[4] > 0 or vit[5] > 0)  # a gun could be firing instead

    @staticmethod
    def _nearest_health(obs) -> float:
        ent = obs["visible_entities"]
        sel = (np.asarray(ent["mask"]) > 0) & (np.asarray(ent["entity_type"]) == ENT_HEALTH)
        d = np.asarray(ent["features"])[sel, 0] * 1024.0
        return float(d.min()) if d.size else math.inf

    def _move_penalty(self, x: float) -> float:
        """Apply a (negative) movement penalty within the per-episode cap."""
        x = max(x, -(MOVE_PENALTY_CAP - self.move_pen))
        self.move_pen -= x
        return x

    def step(self, env, action, obs, terminated: bool) -> float:
        s = env.privileged_state()
        p = self.prev
        r = R["step"]
        st = self.stats
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
        missing = min(max(1.0 - max(s["health"], 0) / 100.0, 0.0), 1.0)
        r += R["damage"] * dd * (1.0 + 2.0 * missing)  # each HP matters more when little is left
        st["dmg"] += dd
        self.since_hit = 0 if dd > 0 else self.since_hit + 1
        healed = max(0.0, s["health"] - p["health"]) if not s["dead"] else 0.0
        if healed > 0:
            missing_before = min(max(1.0 - p["health"] / 100.0, 0.0), 1.0)
            r += R["health_gain"] * healed * (1.0 + 2.0 * missing_before)
            st["healed"] = st.get("healed", 0) + int(healed)
        nk = float(np.sum(s["keys"]) - np.sum(p["keys"]))
        r += R["key"] * nk
        st["keys"] += int(nk)
        r += R["item"] * max(0, s["items"] - p["items"])
        c = self._cell(s)
        if c not in self.cells and not terminated:
            self.cells.add(c)
            r += R["cell"]
            st["cells"] += 1
        if c != self.last_cell and not terminated:  # entered a different 128-unit area
            self.last_cell = c
            n = self.entries[c] = self.entries.get(c, 0) + 1
            if n > 2:  # third and later entries: walking the same cleared ground again
                r += self._move_penalty(R["revisit"] * min(n - 2, 4))
                st["revisits"] = st.get("revisits", 0) + 1
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
        if action[6] and not self._enemy_in_sights(obs):
            r += R["attack_no_target"]
        if self._fists_with_ammo(obs):
            r += R["fists_with_ammo"]
            st["fists"] = st.get("fists", 0) + 1
        enemy = self._enemy_visible(obs)
        moved = math.hypot(s["x"] - p["x"], s["y"] - p["y"])
        if action[0] and not action[1] and moved < BUMP_DIST and not enemy and not terminated:
            r += self._move_penalty(R["bump"])  # pressing on into a stair lip or wall
            st["bumps"] = st.get("bumps", 0) + 1
        turn = int(action[4]) - int(action[5])
        if turn and self.prev_turn and turn != self.prev_turn and not enemy:
            r += self._move_penalty(R["jitter"])
        if turn:
            self.prev_turn = turn
        elif not action[0]:
            self.prev_turn = 0
        if self.since_hit < UNDER_FIRE and (action[2] != action[3]) and not terminated:
            r += R["under_fire_strafe"]  # sidestep while being shot at
        vit_now = np.asarray(obs["vitals"])
        armor = float(vit_now[1])
        if self.prev_armor is not None and armor > self.prev_armor:
            r += R["armor_gain"] * (armor - self.prev_armor)
        self.prev_armor = armor
        hd = self._nearest_health(obs) if s["health"] < SEEK_HP else math.inf
        if math.isfinite(hd) and math.isfinite(self.health_d):
            r += float(np.clip(R["seek_health"] * (self.health_d - hd), -0.2, 0.2))
        self.health_d = hd
        # Stall: pinned against a wall or jittering in place for ~2 s (the bot spent the last 40 s of a map
        # turning left/right at one sign). Potential shaping and curiosity pay 0 there, so it needs a cost.
        self.trail.append((s["x"], s["y"]))
        if len(self.trail) > STALL_STEPS:
            self.trail.pop(0)
            xs, ys = zip(*self.trail)
            if max(xs) - min(xs) < STALL_DIST and max(ys) - min(ys) < STALL_DIST and not terminated \
                    and not self._enemy_visible(obs):  # holding position to fight is fine
                r += self._move_penalty(R["stall"])
                st["stall"] = st.get("stall", 0) + 1
        # Ammo spent with no enemy in view at all (rounds of the selected weapon, from the observation).
        vit = np.asarray(obs["vitals"])
        ammo = float(vit[3])
        if self.prev_ammo is not None and int(vit[2]) == int(self.prev_slot) and ammo < self.prev_ammo \
                and not self._enemy_visible(obs):
            r += R["ammo_wasted"] * (self.prev_ammo - ammo)
            st["wasted"] = st.get("wasted", 0) + int(self.prev_ammo - ammo)
        self.prev_ammo, self.prev_slot = ammo, vit[2]
        self.prev = s
        st["ret"] += r
        st["steps"] += 1
        return float(r)
