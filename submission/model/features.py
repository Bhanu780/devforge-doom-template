"""Observation preprocessing shared by training and the evaluated agent.

Turns the official observation dict into fixed-size arrays for the policy, and maintains the
agent's own spatial memory:

* Odometry: the wrapper gives no position or heading. Heading is integrated from our own TURN
  buttons using vanilla Doom's turn rule (1.76 deg/tic while a turn button has been held for fewer
  than 6 tics, then 3.52 deg/tic). Position is integrated from the player-frame velocity in
  `vitals[14:16]`, rotated by that heading. Both are pure dead reckoning: no map coordinates.
* Memory map: an allocentric grid (agent starts in the centre) where each step writes what the 64
  rays and the visible entities hit, plus where the agent has been. The policy sees a 32x32 crop
  rotated so that "ahead" is always up.

Nothing here chooses actions; it only encodes observations (allowed by docs/RULES.md §1).
"""

from __future__ import annotations

import math

import numpy as np

NUM_RAYS = 64
RAY_RANGE = 1024.0
MAX_ENT = 32
FRAME_SKIP = 4

# --- odometry constants (vanilla Doom, measured against the engine) ---------------------------
SLOW_TURN_DEG = 360.0 * 320 / 65536  # 1.7578 deg per tic for the first 5 tics of a turn
FAST_TURN_DEG = 360.0 * 640 / 65536  # 3.5156 deg per tic afterwards
SLOW_TURN_TICS = 6
START_DEAD_TICS = 7  # measured: no rotation during the first 7 tics of an episode
# Step displacement ~= W_CUR * v_now + W_PREV * v_prev (least squares fit on Freedoom maps, rmse ~2.5 units/step)
W_CUR, W_PREV = 2.6, 1.7

# --- memory map ------------------------------------------------------------------------------
CELL = 64.0  # map units per cell
MAP_SIZE = 192  # cells per side (12,288 units); recentred when the agent nears an edge
CROP = 32  # policy view: CROP x CROP cells around the agent, rotated ahead-up
MAP_CHANNELS = ("free", "wall", "door", "locked", "item", "key", "enemy", "visited")
C_FREE, C_WALL, C_DOOR, C_LOCKED, C_ITEM, C_KEY, C_ENEMY, C_VISITED = range(len(MAP_CHANNELS))
ENEMY_DECAY = 0.97  # per step (enemy marks fade over ~10 s)

# entity type ids (doom_env.spaces.ENTITY_TYPES)
ENT_ENEMY = (1, 2, 3)
ENT_KEY = 9
ENT_ITEMS = (5, 6, 7, 8, 11)

# --- weapons (vitals[2] = selected slot: 1 fist/chainsaw, 2 pistol, 3 shotgun, ...) ----------
SLOT_MELEE = 1
MELEE_RANGE = 96.0  # map units: fist reach (64) plus monster radius
BUTTON_ATTACK, BUTTON_NEXT_WEAPON = 6, 8


def enemy_distances(obs: dict) -> np.ndarray:
    """Map-unit distances of the visible enemies (empty if none)."""
    ent = obs["visible_entities"]
    sel = (np.asarray(ent["mask"]) > 0) & np.isin(np.asarray(ent["entity_type"]), ENT_ENEMY)
    return np.asarray(ent["features"], np.float32)[sel, 0] * RAY_RANGE


def action_mask(obs: dict) -> np.ndarray:
    """(9,) 1 = button allowed. Action masking from observation fields (docs/RULES.md section 1):
    * holding a loaded gun with no enemy in melee range: SELECT_NEXT_WEAPON is blocked. In Doom one
      press from the pistol wraps straight to the fists, so random presses left the bot punching at
      enemies across the room.
    * holding fists/chainsaw with no enemy in melee range: ATTACK is blocked (it only swings at air).
    The network still decides everything else, including switching to fists when an enemy is close
    and switching back to a gun when holding fists."""
    mask = np.ones(9, np.float32)
    vit = np.asarray(obs["vitals"], np.float32)
    slot, ammo = int(vit[2]), float(vit[3])
    d = enemy_distances(obs)
    near = bool(d.size and d.min() < MELEE_RANGE)
    if slot > SLOT_MELEE and ammo > 0 and not near:
        mask[BUTTON_NEXT_WEAPON] = 0.0
    if slot == SLOT_MELEE and not near:
        mask[BUTTON_ATTACK] = 0.0
    return mask


RGB_H, RGB_W = 60, 80
EXTRA_DIM = 17 + 9 + 6  # vitals + previous action + own features


def rgb_small(obs: dict) -> np.ndarray:
    """(4, 60, 80) uint8: RGB + depth at half resolution."""
    rgb = obs["rgb"][::2, ::2]
    depth = obs["depth"][::2, ::2, None]
    return np.ascontiguousarray(np.concatenate([rgb, depth], axis=2).transpose(2, 0, 1))


class FeatureBuilder:
    """Stateful encoder. Call reset() at the start of each map and step(obs) once per decision,
    then note_action(action) with the action actually sent."""

    def __init__(self, include_rgb: bool = True):
        self.include_rgb = include_rgb
        # Crop sampling offsets in the agent frame (x forward, y left), one per crop cell.
        # Row 0 = far ahead, column 0 = far left, so the crop shows ahead-up.
        idx = np.arange(CROP) - CROP / 2 + 0.5
        fwd = -idx[:, None] * np.ones(CROP)[None, :]  # rows: ahead at the top
        left = -idx[None, :] * np.ones(CROP)[:, None]  # cols: left on the left
        self._crop_fwd = fwd.ravel() * CELL
        self._crop_left = left.ravel() * CELL
        ang = np.arange(NUM_RAYS) * (2 * math.pi / NUM_RAYS)
        self._ray_cos, self._ray_sin = np.cos(ang), np.sin(ang)
        self._samples = np.linspace(0.0, 1.0, 24)[None, :]  # points along each ray for "free"
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.map = np.zeros((len(MAP_CHANNELS), MAP_SIZE, MAP_SIZE), np.float32)
        self.x = self.y = 0.0  # map units, relative to the start
        self.heading = 0.0  # radians, relative to the start
        self.turn_held = 0
        self.tic = 0
        self.prev_vel = (0.0, 0.0)
        self._prev_dist = self._prev_hit = None
        self._pending_turn = 0.0
        self.prev_action = np.zeros(9, np.float32)
        self.steps = 0
        self.steps_since_use = 100
        self.stuck = 0
        self.steps_since_damage = 100
        self.last_health = None
        self.origin = np.array([MAP_SIZE / 2, MAP_SIZE / 2])  # cell of world (0, 0)

    def note_action(self, action) -> None:
        """Record the action sent this step (drives heading odometry and the action input)."""
        a = np.asarray(action, np.float32).reshape(9)
        # Heading: the engine applies the turn buttons during the 4 tics of this step.
        left, right = a[4] > 0.5, a[5] > 0.5
        deg = 0.0
        if left or right:
            for _ in range(FRAME_SKIP):
                self.turn_held += 1
                self.tic += 1
                if self.tic <= START_DEAD_TICS:  # the engine ignores turning right after spawn
                    continue
                step = SLOW_TURN_DEG if self.turn_held < SLOW_TURN_TICS else FAST_TURN_DEG
                deg += step * (int(left) - int(right))
        else:
            self.turn_held = 0
            self.tic += FRAME_SKIP
        self._pending_turn = math.radians(deg)
        self.steps_since_use = 0 if a[7] > 0.5 else self.steps_since_use + 1
        self.prev_action = a

    # ------------------------------------------------------------------ main
    def step(self, obs: dict) -> dict:
        """Update odometry and memory from a fresh observation; return policy inputs."""
        vit = np.asarray(obs["vitals"], np.float32)
        self._pending_turn_applied = getattr(self, "_pending_turn", 0.0)
        if self.steps > 0:
            self.heading += self._pending_turn_applied
            # Velocity is reported in the frame of the *new* heading after the step.
            fwd, right = float(vit[14]), float(vit[15])
            fwd, right = self._unblock(obs, fwd, right)
            c, s = math.cos(self.heading), math.sin(self.heading)
            vx, vy = fwd * c + right * s, fwd * s - right * c  # world-frame velocity now
            self.x += W_CUR * vx + W_PREV * self.prev_vel[0]
            self.y += W_CUR * vy + W_PREV * self.prev_vel[1]
            self.prev_vel = (vx, vy)
        self._pending_turn = 0.0
        self.steps += 1

        speed = math.hypot(float(vit[14]), float(vit[15]))
        self.stuck = self.stuck + 1 if (speed < 0.5 and self.prev_action[:4].any()) else 0
        hp = float(vit[0])
        if self.last_health is not None and hp < self.last_health:
            self.steps_since_damage = 0
        else:
            self.steps_since_damage += 1
        self.last_health = hp

        self._recentre_if_needed()
        self._write_map(obs)
        cx, cy = self._cell(self.x, self.y)
        here_visits = float(self.map[C_VISITED, cy, cx])

        extra = np.zeros(EXTRA_DIM, np.float32)
        v = extra[:17]
        v[:] = vit
        v[0] /= 100.0
        v[1] /= 100.0
        v[2] /= 7.0
        v[3:8] = np.log1p(np.maximum(vit[3:8], 0)) / 5.0
        v[14:17] = vit[14:17] / 16.0
        extra[17:26] = self.prev_action
        extra[26] = min(self.steps_since_use, 20) / 20.0
        extra[27] = min(self.stuck, 20) / 20.0
        extra[28] = min(self.steps_since_damage, 40) / 40.0
        extra[29] = math.log1p(here_visits) / 4.0
        extra[30] = min(self.turn_held, 12) / 12.0
        extra[31] = 1.0  # bias

        ent = obs["visible_entities"]
        features = {
            "ray_dist": np.asarray(obs["rays"]["distance"], np.float32),
            "ray_type": np.asarray(obs["rays"]["hit_type"], np.int64),
            "ent_feat": np.clip(np.asarray(ent["features"], np.float32), -8, 8),
            "ent_type": np.asarray(ent["entity_type"], np.int64),
            "ent_mask": np.asarray(ent["mask"], np.float32),
            "extra": extra,
            "map": self._crop(),
            "act_mask": action_mask(obs),
        }
        if self.include_rgb:
            features["rgb"] = rgb_small(obs)
        return features

    def _unblock(self, obs: dict, fwd: float, right: float) -> tuple[float, float]:
        """Doom keeps reporting momentum while the player pushes into a wall. Remove it:
        A static scan (no turn, wall rays unchanged) means we did not move."""
        dist = np.asarray(obs["rays"]["distance"], np.float64) * RAY_RANGE
        hit = np.asarray(obs["rays"]["hit_type"])
        static = False
        if self._prev_dist is not None:
            turn = getattr(self, "_pending_turn_applied", 0.0)
            if turn == 0.0:
                walls = (hit >= 1) & (hit <= 3) & (self._prev_hit >= 1) & (self._prev_hit <= 3)
                static = walls.sum() >= 16 and np.max(np.abs(dist[walls] - self._prev_dist[walls])) < 0.75
            else:
                # previous scan seen from the new heading: new ray i looks where old ray i + turn did
                pos = (np.arange(NUM_RAYS) + turn / (2 * math.pi / NUM_RAYS)) % NUM_RAYS
                lo = np.floor(pos).astype(int)
                fr = pos - lo
                hi = (lo + 1) % NUM_RAYS
                prev = self._prev_dist[lo] * (1 - fr) + self._prev_dist[hi] * fr
                pw = (self._prev_hit[lo] >= 1) & (self._prev_hit[lo] <= 3) & (self._prev_hit[hi] >= 1) & (self._prev_hit[hi] <= 3)
                walls = (hit >= 1) & (hit <= 3) & pw & (dist < 512)
                static = walls.sum() >= 16 and np.percentile(np.abs(dist[walls] - prev[walls]), 80) < 1.5
        self._prev_dist, self._prev_hit = dist, hit
        if static:
            self.prev_vel = (0.0, 0.0)
            return 0.0, 0.0
        return fwd, right

    # ------------------------------------------------------------------ map helpers
    def _cell(self, x, y):
        cx = np.floor(self.origin[0] + np.asarray(x) / CELL).astype(np.int64)
        cy = np.floor(self.origin[1] - np.asarray(y) / CELL).astype(np.int64)  # +y = up = row--
        return cx, cy

    def _recentre_if_needed(self) -> None:
        cx, cy = self._cell(self.x, self.y)
        margin = CROP
        if margin <= cx < MAP_SIZE - margin and margin <= cy < MAP_SIZE - margin:
            return
        dx = int(MAP_SIZE / 2 - cx)
        dy = int(MAP_SIZE / 2 - cy)
        shifted = np.zeros_like(self.map)
        sx0, sx1 = max(0, -dx), min(MAP_SIZE, MAP_SIZE - dx)
        sy0, sy1 = max(0, -dy), min(MAP_SIZE, MAP_SIZE - dy)
        shifted[:, sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = self.map[:, sy0:sy1, sx0:sx1]
        self.map = shifted
        self.origin = self.origin + np.array([dx, dy])

    def _write_map(self, obs: dict) -> None:
        m = self.map
        m[C_ENEMY] *= ENEMY_DECAY
        h = self.heading
        c, s = math.cos(h), math.sin(h)
        # ray directions in the map frame (ray 0 = ahead, counter-clockwise)
        dx = c * self._ray_cos - s * self._ray_sin
        dy = s * self._ray_cos + c * self._ray_sin
        dist = np.asarray(obs["rays"]["distance"], np.float64) * RAY_RANGE
        hit = np.asarray(obs["rays"]["hit_type"])

        # free space along each ray (stop just short of the hit)
        t = self._samples * np.maximum(dist[:, None] - CELL * 0.5, 0.0)
        fx, fy = self._cell(self.x + dx[:, None] * t, self.y + dy[:, None] * t)
        ok = (fx >= 0) & (fx < MAP_SIZE) & (fy >= 0) & (fy < MAP_SIZE)
        m[C_FREE, fy[ok], fx[ok]] = 1.0

        # hit points
        has = hit > 0
        if has.any():
            hx, hy = self._cell(self.x + dx[has] * dist[has], self.y + dy[has] * dist[has])
            ht = hit[has]
            ok = (hx >= 0) & (hx < MAP_SIZE) & (hy >= 0) & (hy < MAP_SIZE)
            hx, hy, ht = hx[ok], hy[ok], ht[ok]
            for code, ch in ((1, C_WALL), (2, C_DOOR), (3, C_LOCKED), (5, C_ITEM)):
                sel = ht == code
                if sel.any():
                    m[ch, hy[sel], hx[sel]] = 1.0
            # an open door no longer blocks rays: clear stale door marks along free space
            m[C_DOOR] *= 1.0 - m[C_FREE] * (m[C_WALL] == 0)

        # visible entities
        ent = obs["visible_entities"]
        mask = np.asarray(ent["mask"]) > 0
        if mask.any():
            f = np.asarray(ent["features"])[mask]
            et = np.asarray(ent["entity_type"])[mask]
            d = f[:, 0] * RAY_RANGE
            sa, ca = f[:, 1], f[:, 2]
            wx = self.x + d * (ca * c - sa * s)
            wy = self.y + d * (ca * s + sa * c)
            ex, ey = self._cell(wx, wy)
            ok = (ex >= 0) & (ex < MAP_SIZE) & (ey >= 0) & (ey < MAP_SIZE)
            for sel, ch in ((np.isin(et, ENT_ENEMY), C_ENEMY), (et == ENT_KEY, C_KEY), (np.isin(et, ENT_ITEMS), C_ITEM)):
                sel = sel & ok
                if sel.any():
                    m[ch, ey[sel], ex[sel]] = 1.0

        # where we are; a key we just picked up disappears, so clear key marks under us
        cx, cy = self._cell(self.x, self.y)
        if 0 <= cx < MAP_SIZE and 0 <= cy < MAP_SIZE:
            m[C_VISITED, cy, cx] += 1.0
            m[C_KEY, max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = 0.0
            m[C_ITEM, max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = 0.0
            m[C_FREE, cy, cx] = 1.0

    def visits_here(self) -> float:
        """How many steps the agent has spent in its current memory-map cell (>= 1 after step()).
        Used for the intrinsic curiosity reward during training."""
        cx, cy = self._cell(self.x, self.y)
        if 0 <= cx < MAP_SIZE and 0 <= cy < MAP_SIZE:
            return float(self.map[C_VISITED, cy, cx])
        return 1.0

    def cell_key(self) -> tuple[int, int]:
        """The agent's current memory-map cell in start-relative coordinates (unaffected by recentring)."""
        return int(math.floor(self.x / CELL)), int(math.floor(self.y / CELL))

    def _crop(self) -> np.ndarray:
        c, s = math.cos(self.heading), math.sin(self.heading)
        wx = self.x + self._crop_fwd * c - self._crop_left * s
        wy = self.y + self._crop_fwd * s + self._crop_left * c
        cx, cy = self._cell(wx, wy)
        ok = (cx >= 0) & (cx < MAP_SIZE) & (cy >= 0) & (cy < MAP_SIZE)
        out = np.zeros((len(MAP_CHANNELS), CROP * CROP), np.float32)
        out[:, ok] = self.map[:, cy[ok], cx[ok]]
        out[C_VISITED] = np.log1p(out[C_VISITED]) / 3.0
        return out.reshape(len(MAP_CHANNELS), CROP, CROP)
