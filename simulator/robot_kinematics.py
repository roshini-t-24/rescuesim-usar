"""simulator/robot_kinematics.py

Two heterogeneous agents that cooperate inside a DisasterWorld.

QuadrupedRover (ground, ANYmal/Spot class)
    state  [x, y, z, theta, v, omega, battery_pct]
    - hard slope limit ||grad(elevation)|| <= 0.55 (about 28 deg)
    - energy  dE = k1*|v|*dt + k2*max(0, dz)  (+ idle power)
    - global planner: Risk-Aware A*   C = d + alpha*slope + beta*instability (+ gamma*smoke)
    - local planner: Dynamic Window Approach over (v, omega) samples

MicroDrone (aerial scout, sub-250 g indoor UAV)
    state  [x, y, z, vx, vy, vz, battery_pct]
    - flies z in [1.5, 12] m, pushed around by Ornstein-Uhlenbeck wind
    - 3D Artificial Potential Field: attracted to unexplored frontiers and thermal
      candidates, repelled by solid voxels (pillars, rubble, ceiling slab)

SharedMap is the knowledge both robots share. The rover plans optimistically through
unknown cells (small penalty), and local sensing reveals true terrain as it moves, so an
aftershock that invalidates part of the map really does force replanning.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label

from config import settings as cfg
from simulator.disaster_world import DisasterWorld

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
COLLAPSE_LOSS_RATE_PER_S = float(__import__("os").environ.get("RESCUESIM_COLLAPSE_RATE", "0.002"))
DRONE_CEILING_HAZARD_FACTOR: float = 0.35    # small flier is less exposed than a heavy rover
UNKNOWN_PRIOR_INSTABILITY: float = 0.35      # planner's guess for unmapped cells
UNKNOWN_CELL_PENALTY: float = 0.2
HIGH_RISK_INSTABILITY: float = 0.7
ROVER_SENSE_RADIUS_M: float = 6.0
SQRT2: float = math.sqrt(2.0)
_NEIGHBORS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------
# Hazard exposure (shared by both robots)
# --------------------------------------------------------------------------
def structural_loss_probability(instability: float, dt: float, factor: float = 1.0) -> float:
    """P(robot lost to a secondary collapse during dt) with hazard rate = k * instability^3."""
    rate = COLLAPSE_LOSS_RATE_PER_S * factor * max(instability, 0.0) ** 3
    return 1.0 - math.exp(-rate * dt)


def aftershock_crush_probability(event: Dict[str, Any], x: float, y: float,
                                 instability: float, factor: float = 1.0) -> float:
    ex, ey = event["epicenter_xy"]
    radius = float(event["radius_m"])
    d = math.hypot(x - ex, y - ey)
    if d > radius:
        return 0.0
    w = math.exp(-(d ** 2) / (2.0 * (radius / 2.0) ** 2))
    scale = float(np.clip((float(event["magnitude"]) - 3.0) / 3.5, 0.1, 1.0))
    return float(np.clip(factor * scale * w * (0.25 + 0.75 * instability), 0.0, 0.95))


# --------------------------------------------------------------------------
# Shared knowledge map
# --------------------------------------------------------------------------
class SharedMap:
    """Boolean 'cell has been observed' grid shared by drone and rover."""

    def __init__(self, nx: int, ny: int, cell: float = 1.0) -> None:
        self.known = np.zeros((nx, ny), dtype=bool)
        self.cell = cell

    def _disc(self, x: float, y: float, r: float) -> Tuple[slice, slice, np.ndarray]:
        nx, ny = self.known.shape
        i0, i1 = max(0, int((x - r) / self.cell)), min(nx, int((x + r) / self.cell) + 2)
        j0, j1 = max(0, int((y - r) / self.cell)), min(ny, int((y + r) / self.cell) + 2)
        ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
        mask = np.hypot((ii + 0.5) * self.cell - x, (jj + 0.5) * self.cell - y) <= r
        return slice(i0, i1), slice(j0, j1), mask

    def mark_disc(self, x: float, y: float, r: float) -> None:
        si, sj, m = self._disc(x, y, r)
        self.known[si, sj] |= m

    def invalidate(self, x: float, y: float, r: float) -> None:
        """Forget a region (e.g. after an aftershock changed the terrain there)."""
        si, sj, m = self._disc(x, y, r)
        self.known[si, sj] &= ~m

    def mark_cells(self, ii: np.ndarray, jj: np.ndarray) -> None:
        nx, ny = self.known.shape
        ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        self.known[ii[ok], jj[ok]] = True

    def fraction(self) -> float:
        return float(self.known.mean())


# --------------------------------------------------------------------------
# Risk-aware A*
# --------------------------------------------------------------------------
@dataclass
class PathResult:
    cells: List[Tuple[int, int]]
    cost: float
    length_m: float
    expansions: int


def build_cost_field(world: DisasterWorld, weights: cfg.PolicyWeights,
                     known: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cell risk cost (added to step distance) and the blocked mask for the planner.

    Edge cost into a cell:  C = d + alpha*slope + beta*instability + gamma*smoke
    (+ a soft FEMA stand-off penalty near high-risk cells). Unknown cells use optimistic priors.
    """
    cell = world.cell
    slope, inst = world.slope_map, world.structural_instability
    k = np.clip(((world.elevation_map + 0.5) / cell).astype(int), 0, world.nz - 1)
    smoke_g = np.take_along_axis(world.smoke_density, k[:, :, None], axis=2)[:, :, 0]

    high = inst > HIGH_RISK_INSTABILITY
    if high.any():
        dist_m = distance_transform_edt(~high) * cell
    else:
        dist_m = np.full(inst.shape, 1e3)
    margin = np.clip(1.0 - dist_m / max(world.fema_margin_m, 0.5), 0.0, 1.0)

    trav, pillar = world.traversable_mask(), world.pillar_mask
    if known is not None:
        slope = np.where(known, slope, 0.0)
        inst = np.where(known, inst, UNKNOWN_PRIOR_INSTABILITY)
        smoke_g = np.where(known, smoke_g, 0.0)
        margin = np.where(known, margin, 0.0)
        trav = np.where(known, trav, True)
        pillar = np.where(known, pillar, False)
        unknown_pen = np.where(known, 0.0, UNKNOWN_CELL_PENALTY)
    else:
        unknown_pen = 0.0

    cost = (weights.alpha_slope * slope + weights.beta_collapse * (inst + 0.5 * margin)
            + weights.gamma_smoke * smoke_g + unknown_pen)
    blocked = (~trav) | binary_dilation(pillar, iterations=1)
    return cost, blocked


def risk_aware_astar(cost: np.ndarray, blocked: np.ndarray, start: Tuple[int, int],
                     goal: Tuple[int, int], cell: float = 1.0,
                     goal_radius_m: float = 1.5) -> Optional[PathResult]:
    """8-connected A* with octile heuristic. The start cell may be blocked; goals are reached
    when any free cell within goal_radius_m of the goal is expanded."""
    nx, ny = cost.shape
    flat_cost = cost.ravel().tolist()
    flat_block = blocked.ravel().tolist()
    sx, sy = start
    gx, gy = goal
    r_cells = goal_radius_m / cell
    inf = float("inf")
    g = [inf] * (nx * ny)
    parent = [-1] * (nx * ny)
    closed = [False] * (nx * ny)

    def h(x: int, y: int) -> float:
        dx, dy = abs(x - gx), abs(y - gy)
        return max(0.0, (max(dx, dy) + (SQRT2 - 1.0) * min(dx, dy)) * cell - goal_radius_m)

    s = sx * ny + sy
    g[s] = 0.0
    heap: List[Tuple[float, float, int]] = [(h(sx, sy), 0.0, s)]
    expansions = 0
    while heap:
        _, gc, idx = heapq.heappop(heap)
        if closed[idx]:
            continue
        closed[idx] = True
        expansions += 1
        x, y = divmod(idx, ny)
        if math.hypot(x - gx, y - gy) <= r_cells and not flat_block[idx]:
            cells = []
            cur = idx
            while cur != -1:
                cells.append(divmod(cur, ny))
                cur = parent[cur]
            cells.reverse()
            length = sum(math.hypot(cells[i + 1][0] - cells[i][0], cells[i + 1][1] - cells[i][1])
                         for i in range(len(cells) - 1)) * cell
            return PathResult(cells, gc, length, expansions)
        for dx, dy, dm in _NEIGHBORS:
            ax, ay = x + dx, y + dy
            if ax < 0 or ax >= nx or ay < 0 or ay >= ny:
                continue
            nidx = ax * ny + ay
            if closed[nidx] or flat_block[nidx]:
                continue
            if dx and dy and (flat_block[(x + dx) * ny + y] or flat_block[x * ny + y + dy]):
                continue     # no corner cutting
            ng = gc + dm * cell + flat_cost[nidx]
            if ng < g[nidx]:
                g[nidx] = ng
                parent[nidx] = idx
                heapq.heappush(heap, (ng + h(ax, ay), ng, nidx))
    return None


# --------------------------------------------------------------------------
# Quadruped rover
# --------------------------------------------------------------------------
class QuadrupedRover:
    def __init__(self, world: DisasterWorld, shared_map: SharedMap,
                 weights: Optional[cfg.PolicyWeights] = None,
                 start_xy: Optional[Tuple[float, float]] = None,
                 rng: Optional[np.random.Generator] = None,
                 auto_replan: bool = True) -> None:
        self.world, self.map = world, shared_map
        self.weights = weights or cfg.PolicyWeights()
        self.rng = rng if rng is not None else np.random.default_rng(world.seed + 202)
        self.auto_replan = auto_replan
        sx, sy = start_xy or world.entry_xy
        self.x, self.y = float(sx), float(sy)
        self.z = world.elevation_at(self.x, self.y)
        self.theta, self.v, self.omega = 0.0, 0.0, 0.0
        self.energy_used_wh = 0.0
        self.distance_m = 0.0
        self.pods = cfg.ROVER.payload_pods
        self.alive = True
        self.loss_reason: Optional[str] = None
        self.status = "idle"
        self.goal: Optional[Tuple[float, float]] = None
        self.goal_reached = False
        self.goal_radius_m = 1.5
        self.plan: Optional[PathResult] = None
        self.path_xy = np.zeros((0, 2))
        self.path_idx = 0
        self._plan_version = -1
        self.stuck_steps = 0
        self._idle_steps = 0
        self._stuck_replans = 0
        self._stuck_anchor = (self.x, self.y)
        self.goal_unreachable = False
        self.replans = 0
        self.risk_exposure = 0.0          # integral of instability dt  (structural risk exposure)
        self.trail: List[Tuple[float, float, float]] = []
        self._maps_version = -1
        self._maps: Dict[str, Any] = {}
        self.map.mark_disc(self.x, self.y, ROVER_SENSE_RADIUS_M)

    # ---- state ----
    @property
    def battery_pct(self) -> float:
        return max(0.0, 100.0 * (1.0 - self.energy_used_wh / cfg.ROVER.battery_capacity_wh))

    @property
    def state_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.theta, self.v, self.omega, self.battery_pct])

    def set_weights(self, weights: cfg.PolicyWeights) -> None:
        self.weights = weights
        if self.goal is not None:
            self._plan()

    # ---- maps of the *true* local terrain (what the legs actually feel) ----
    def _local_maps(self) -> Dict[str, Any]:
        w = self.world
        if self._maps_version != w.terrain_version or not self._maps:
            blocked = (~w.traversable_mask()) | binary_dilation(w.pillar_mask, iterations=1)
            _, idx = distance_transform_edt(blocked, return_indices=True)
            self._maps = {
                "blocked": blocked,
                "dist_obs": distance_transform_edt(~blocked) * w.cell,
                "nearest_free": idx,
            }
            self._maps_version = w.terrain_version
        return self._maps

    # ---- global planning ----
    def plan_to(self, goal_xy: Tuple[float, float]) -> bool:
        self.goal = (float(goal_xy[0]), float(goal_xy[1]))
        self.goal_reached = False
        self.goal_unreachable = False
        self._idle_steps = self._stuck_replans = 0
        return self._plan()

    def _plan(self) -> bool:
        if self.goal is None:
            return False
        w = self.world
        cost, blocked = build_cost_field(w, self.weights, self.map.known)
        res = risk_aware_astar(cost, blocked, w._ij(self.x, self.y), w._ij(*self.goal),
                               cell=w.cell, goal_radius_m=self.goal_radius_m)
        self._plan_version = w.terrain_version
        if res is None:
            self.plan, self.path_xy, self.path_idx = None, np.zeros((0, 2)), 0
            self.status = "no_path"
            return False
        self.plan = res
        self.path_xy = np.array([((i + 0.5) * w.cell, (j + 0.5) * w.cell) for i, j in res.cells])
        self.path_idx = 0
        self.status = "navigating"
        return True

    def plan_summary(self) -> Dict[str, Any]:
        if self.plan is None:
            return {"ok": False}
        w = self.world
        inst = [float(w.structural_instability[i, j]) for i, j in self.plan.cells]
        return {"ok": True, "length_m": round(self.plan.length_m, 1), "cost": round(self.plan.cost, 1),
                "mean_instability": round(float(np.mean(inst)), 3),
                "max_instability": round(float(np.max(inst)), 3), "expansions": self.plan.expansions}

    # ---- local planning (DWA) ----
    def _los_clear(self, q: np.ndarray, blocked: np.ndarray) -> bool:
        w = self.world
        d = math.hypot(q[0] - self.x, q[1] - self.y)
        n = max(2, int(math.ceil(d / 0.05)) + 1)      # fine sampling: do not miss corner slivers
        xs = np.linspace(self.x, q[0], n)
        ys = np.linspace(self.y, q[1], n)
        ii = np.clip(np.floor(xs / w.cell).astype(int), 0, w.nx - 1)
        jj = np.clip(np.floor(ys / w.cell).astype(int), 0, w.ny - 1)
        return not bool(blocked[ii, jj].any())

    def _lookahead(self, dist_m: float = 2.5) -> np.ndarray:
        """Farthest path point (within dist_m) that is reachable along a clear straight line,
        so the local planner never aims across a blocked corner."""
        p = self.path_xy
        lo, hi = self.path_idx, min(len(p), self.path_idx + 15)
        d = np.hypot(p[lo:hi, 0] - self.x, p[lo:hi, 1] - self.y)
        self.path_idx = lo + int(d.argmin())
        blocked = self._local_maps()["blocked"]
        j = self.path_idx
        while j < len(p) - 1:
            nxt = p[j + 1]
            near_now = math.hypot(p[j][0] - self.x, p[j][1] - self.y) < 0.8
            if not (math.hypot(nxt[0] - self.x, nxt[1] - self.y) <= dist_m or near_now):
                break
            if not self._los_clear(nxt, blocked):
                break
            j += 1
        return p[j]

    def _dwa(self, target: np.ndarray, dt: float, maps: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        R, w = cfg.ROVER, self.world
        slope_here = w.slope_at(self.x, self.y)
        vmax = R.max_speed_mps * (1.0 - 0.5 * min(slope_here / R.max_slope, 1.0))
        vs = np.linspace(max(0.0, self.v - R.max_accel_mps2 * dt), min(vmax, self.v + R.max_accel_mps2 * dt),
                         R.dwa_v_samples)
        ws = np.linspace(max(-R.max_yaw_rate_rps, self.omega - R.max_yaw_accel_rps2 * dt),
                         min(R.max_yaw_rate_rps, self.omega + R.max_yaw_accel_rps2 * dt), R.dwa_w_samples)
        V, W = np.meshgrid(vs, ws, indexing="ij")
        V, W = V.ravel(), W.ravel()
        h = 0.25
        n = int(R.dwa_horizon_s / h)
        T = np.arange(1, n + 1) * h
        th = self.theta + W[:, None] * T[None, :]
        small = np.abs(W) < 1e-3
        sw = np.where(small, 1.0, W)
        xs = np.where(small[:, None], self.x + V[:, None] * math.cos(self.theta) * T[None, :],
                      self.x + (V / sw)[:, None] * (np.sin(th) - math.sin(self.theta)))
        ys = np.where(small[:, None], self.y + V[:, None] * math.sin(self.theta) * T[None, :],
                      self.y - (V / sw)[:, None] * (np.cos(th) - math.cos(self.theta)))
        ii = np.floor(xs / w.cell).astype(int)
        jj = np.floor(ys / w.cell).astype(int)
        inb = (ii >= 0) & (ii < w.nx) & (jj >= 0) & (jj < w.ny)
        ic, jc = np.clip(ii, 0, w.nx - 1), np.clip(jj, 0, w.ny - 1)
        coll = (~inb) | maps["blocked"][ic, jc]
        any_c = coll.any(axis=1)
        first = np.where(any_c, coll.argmax(axis=1), n)
        dist_free = first * np.maximum(V, 1e-6) * h
        admissible = (V < 1e-6) | (V <= np.sqrt(2.0 * R.max_accel_mps2 * dist_free) + 1e-9)
        k_chk = max(1, int(math.ceil(dt / h)))          # the whole control step must be collision-free
        valid = admissible & ~coll[:, :k_chk].any(axis=1)

        ang = np.arctan2(target[1] - ys[:, -1], target[0] - xs[:, -1])
        herr = np.abs((ang - th[:, -1] + math.pi) % (2.0 * math.pi) - math.pi)
        heading = 1.0 - herr / math.pi
        clearance = np.minimum(maps["dist_obs"][ic[:, -1], jc[:, -1]], 3.0) / 3.0
        risk = w.structural_instability[ic, jc].mean(axis=1)
        score = (2.0 * heading + 0.5 * clearance + 0.5 * V / R.max_speed_mps
                 - (0.05 + 0.1 * self.weights.beta_collapse) * risk - 0.15 * (V < 1e-6))
        score = np.where(valid, score, -np.inf)
        best = int(score.argmax())
        if not np.isfinite(score[best]):
            return None
        return float(V[best]), float(W[best])

    def _escape(self, maps: Dict[str, Any]) -> Tuple[float, float]:
        """Current cell became blocked (aftershock). Drive to the nearest free cell."""
        w = self.world
        ci, cj = w._ij(self.x, self.y)
        ti, tj = maps["nearest_free"][0][ci, cj], maps["nearest_free"][1][ci, cj]
        tx, ty = (ti + 0.5) * w.cell, (tj + 0.5) * w.cell
        herr = wrap_angle(math.atan2(ty - self.y, tx - self.x) - self.theta)
        return (0.3 if abs(herr) < 0.8 else 0.0), float(np.clip(2.0 * herr, -cfg.ROVER.max_yaw_rate_rps,
                                                                  cfg.ROVER.max_yaw_rate_rps))

    # ---- integration ----
    def step(self, dt: float = cfg.DT_S) -> List[Dict[str, Any]]:
        ev: List[Dict[str, Any]] = []
        if not self.alive:
            return ev
        w, R = self.world, cfg.ROVER
        if self.battery_pct <= 0.0:
            self.v = self.omega = 0.0
            self.status = "battery_depleted"
            return ev

        if self.goal is not None and self.auto_replan and w.terrain_version != self._plan_version:
            ok = self._plan()
            self.replans += 1
            ev.append({"type": "rover_replan", "reason": "terrain_change", "ok": ok,
                       "terrain_version": w.terrain_version})
        if w.step_count % 4 == 0:
            self.map.mark_disc(self.x, self.y, ROVER_SENSE_RADIUS_M)

        escaping = False
        navigating = False
        v_cmd, w_cmd = 0.0, 0.0
        if self.goal_unreachable:
            self.status = "blocked"
        elif self.goal is not None and not self.goal_reached and self.plan is not None:
            at_path_end = (self.path_idx >= len(self.path_xy) - 2 and
                           math.hypot(self.path_xy[-1][0] - self.x, self.path_xy[-1][1] - self.y) <= 0.6)
            if at_path_end or math.hypot(self.goal[0] - self.x, self.goal[1] - self.y) <= self.goal_radius_m:
                self.goal_reached = True
                self.status = "at_goal"
                ev.append({"type": "rover_goal_reached", "x": round(self.x, 1), "y": round(self.y, 1)})
            else:
                maps = self._local_maps()
                ci, cj = w._ij(self.x, self.y)
                if maps["blocked"][ci, cj]:
                    v_cmd, w_cmd = self._escape(maps)
                    escaping = True
                else:
                    cmd = self._dwa(self._lookahead(), dt, maps)
                    if cmd is None:
                        self._handle_stuck(ev)
                    else:
                        v_cmd, w_cmd = cmd
                        self.stuck_steps = 0
                        navigating = True
        elif self.goal is None or self.goal_reached:
            self.status = "idle" if self.goal is None else "at_goal"

        old_z = self.z
        th_mid = self.theta + 0.5 * w_cmd * dt
        nx_, ny_ = self.x + v_cmd * math.cos(th_mid) * dt, self.y + v_cmd * math.sin(th_mid) * dt
        if w.in_bounds(nx_, ny_):
            ci, cj = w._ij(nx_, ny_)
            if escaping or not self._local_maps()["blocked"][ci, cj]:
                self.distance_m += math.hypot(nx_ - self.x, ny_ - self.y)
                self.x, self.y = nx_, ny_
                self.v, self.omega = v_cmd, w_cmd
            else:
                self.v = self.omega = 0.0
        else:
            self.v = self.omega = 0.0
        self.theta = wrap_angle(self.theta + self.omega * dt)
        self.z = w.elevation_at(self.x, self.y)
        if navigating:                                   # actual (not commanded) progress
            self._idle_steps = self._idle_steps + 1 if abs(self.v) < 0.05 else 0
            if self._idle_steps >= 10:
                self._handle_stuck(ev, force=True)

        # energy: dE = k1*|v|*dt + k2*max(0, dz) + idle
        self.energy_used_wh += (R.k1_wh_per_m * abs(self.v) * dt + R.k2_wh_per_m_climb * max(0.0, self.z - old_z)
                                + R.idle_power_w * dt / 3600.0)

        inst = w.instability_at(self.x, self.y)
        self.risk_exposure += inst * dt
        if self.rng.random() < structural_loss_probability(inst, dt):
            self._lose("structural_collapse", ev)
        if w.step_count % 2 == 0 and len(self.trail) < 4000:
            self.trail.append((round(self.x, 2), round(self.y, 2), round(self.z, 2)))
        return ev

    def _handle_stuck(self, ev: List[Dict[str, Any]], force: bool = False) -> None:
        self.stuck_steps += 1
        self.v = self.omega = 0.0
        if force or self.stuck_steps >= 3:
            if math.hypot(self.x - self._stuck_anchor[0], self.y - self._stuck_anchor[1]) > 1.5:
                self._stuck_replans = 0              # we made real progress since the last stall
            self._stuck_anchor = (self.x, self.y)
            self.map.mark_disc(self.x, self.y, 5.0)   # local sensing reveals what is really there
            ok = self._plan()
            self.replans += 1
            self._stuck_replans += 1
            self._idle_steps = 0
            self.stuck_steps = 0
            ev.append({"type": "rover_stuck_replan", "ok": ok})
            if not ok or self._stuck_replans >= 4:
                self.goal_unreachable = True
                self.status = "blocked"
                ev.append({"type": "rover_blocked", "x": round(self.x, 1), "y": round(self.y, 1)})

    def _lose(self, reason: str, ev: List[Dict[str, Any]]) -> None:
        self.alive, self.loss_reason, self.status = False, reason, "lost"
        self.v = self.omega = 0.0
        ev.append({"type": "rover_lost", "reason": reason, "x": round(self.x, 1), "y": round(self.y, 1)})

    def apply_aftershock(self, event: Dict[str, Any]) -> bool:
        """Roll for being crushed by an aftershock. Returns True if the rover was lost."""
        if not self.alive:
            return False
        p = aftershock_crush_probability(event, self.x, self.y, self.world.instability_at(self.x, self.y))
        if self.rng.random() < p:
            self._lose("aftershock_crush", [])
            return True
        return False

    def deliver_pod(self) -> bool:
        if self.pods <= 0 or not self.alive:
            return False
        self.pods -= 1
        return True

    def telemetry(self) -> Dict[str, Any]:
        return {"x": round(self.x, 2), "y": round(self.y, 2), "z": round(self.z, 2),
                "theta": round(self.theta, 2), "v": round(self.v, 2), "omega": round(self.omega, 2),
                "battery_pct": round(self.battery_pct, 1), "status": self.status, "alive": self.alive,
                "loss_reason": self.loss_reason, "distance_m": round(self.distance_m, 1),
                "energy_wh": round(self.energy_used_wh, 2), "replans": self.replans,
                "pods": self.pods, "risk_exposure": round(self.risk_exposure, 2),
                "weights": self.weights.as_dict()}


# --------------------------------------------------------------------------
# Micro drone
# --------------------------------------------------------------------------
class MicroDrone:
    APF_PAD = 3

    def __init__(self, world: DisasterWorld, shared_map: SharedMap,
                 start_xy: Optional[Tuple[float, float]] = None, start_z: float = 3.0,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.world, self.map = world, shared_map
        self.rng = rng if rng is not None else np.random.default_rng(world.seed + 303)
        sx, sy = start_xy or world.entry_xy
        self.pos = np.array([sx, sy, start_z], dtype=float)
        self.vel = np.zeros(3)
        self.energy_used_wh = 0.0
        self.alive = True
        self.loss_reason: Optional[str] = None
        self.status = "exploring"
        self.attractors: List[Tuple[np.ndarray, float]] = []
        self.frontier_target: Optional[np.ndarray] = None
        self.trail: List[Tuple[float, float, float]] = []
        self.distance_m = 0.0
        self.risk_exposure = 0.0
        self.wind_drift_m = 0.0
        self._stuck = 0
        self._target_age = 0
        self._target_best = float("inf")
        self._no_prog = 0
        self._blacklist: List[Tuple[float, float, int]] = []
        self._escape_steps = 0
        self._escape_dir = np.zeros(3)
        self._pad_version = -1
        self._solid_pad: Optional[np.ndarray] = None
        self.map_points: List[Tuple[float, float, float]] = []

    @property
    def battery_pct(self) -> float:
        return max(0.0, 100.0 * (1.0 - self.energy_used_wh / cfg.DRONE.battery_capacity_wh))

    @property
    def state_vector(self) -> np.ndarray:
        return np.array([*self.pos, *self.vel, self.battery_pct])

    # ---- mapping ----
    def update_map(self, scan: Any) -> None:
        """Mark the 2.5D columns a LiDAR scan observed (up to the hit or the smoke limit),
        plus a downward-looking footprint under the drone."""
        w = self.world
        reach = np.where(scan.hit, np.nan_to_num(scan.ranges, nan=0.0), scan.clear_range)
        d = np.arange(0.5, cfg.SENSORS.lidar_range_m + 0.5, 1.0)
        mask = d[None, :] <= reach[:, None] + 0.5
        xs = self.pos[0] + np.cos(scan.angles)[:, None] * d[None, :]
        ys = self.pos[1] + np.sin(scan.angles)[:, None] * d[None, :]
        ii = np.floor(xs[mask] / w.cell).astype(int)
        jj = np.floor(ys[mask] / w.cell).astype(int)
        self.map.mark_cells(ii, jj)
        self.map.mark_disc(self.pos[0], self.pos[1], float(np.clip(0.6 * self.pos[2], 2.5, 6.0)))
        pts = scan.hit_points()
        if len(pts) and len(self.map_points) < 3000:
            self.map_points.extend((round(float(p[0]), 1), round(float(p[1]), 1), round(float(p[2]), 1)) for p in pts)

    def set_thermal_attractors(self, items: Sequence[Tuple[Sequence[float], float]]) -> None:
        self.attractors = [(np.asarray(p, dtype=float), float(wt)) for p, wt in items]

    def _update_frontier(self) -> None:
        """Pick the best frontier *cell* (unknown cell touching known space), weighting by the size
        of its frontier cluster (information gain) and distance. Cells, not centroids: the centroid
        of a curved frontier lies inside already-mapped space."""
        known = self.map.known
        frontier = (~known) & binary_dilation(known)
        lab, n = label(frontier, structure=np.ones((3, 3)))
        if n == 0:
            self.frontier_target = None
            return
        sizes = np.bincount(lab.ravel())[1:]
        cells = np.argwhere(frontier)
        csize = sizes[lab[frontier] - 1]
        cx = (cells[:, 0] + 0.5) * self.world.cell
        cy = (cells[:, 1] + 0.5) * self.world.cell
        d = np.hypot(cx - self.pos[0], cy - self.pos[1])
        ok = (csize >= 4) & (d >= 2.5)
        for bx, by, expire in self._blacklist:
            if expire > self.world.step_count:
                ok &= np.hypot(cx - bx, cy - by) > 4.0
        if not ok.any():
            ok = d >= 2.5
        if not ok.any():
            ok = np.ones_like(d, dtype=bool)
        utility = np.where(ok, np.sqrt(csize) / (d + 5.0), -np.inf)
        k = int(np.argmax(utility))
        tz = float(np.clip(self.world.elevation_at(cx[k], cy[k]) + 4.0, 3.0, cfg.DRONE.z_max_m - 1.0))
        self.frontier_target = np.array([cx[k], cy[k], tz])
        self._target_best, self._no_prog = float("inf"), 0

    # ---- potential field ----
    def _repulsion(self) -> np.ndarray:
        w, D, P = self.world, cfg.DRONE, self.APF_PAD
        if self._pad_version != w.terrain_version or self._solid_pad is None:
            self._solid_pad = np.pad(w.solid, P, mode="constant", constant_values=True)
            self._pad_version = w.terrain_version
        i = int(np.clip(math.floor(self.pos[0] / w.cell), 0, w.nx - 1))
        j = int(np.clip(math.floor(self.pos[1] / w.cell), 0, w.ny - 1))
        k = int(np.clip(math.floor(self.pos[2] / w.cell), 0, w.nz - 1))
        blk = self._solid_pad[i:i + 2 * P + 1, j:j + 2 * P + 1, k:k + 2 * P + 1]
        idx = np.argwhere(blk)
        if len(idx) == 0:
            return np.zeros(3)
        centers = (np.array([i, j, k]) + idx - P + 0.5) * w.cell
        diff = self.pos[None, :] - centers
        dist = np.maximum(np.linalg.norm(diff, axis=1), 1e-6)
        d_s = np.maximum(dist - 0.5 * w.cell - D.radius_m, 0.05)
        near = d_s < D.repel_influence_m
        if not near.any():
            return np.zeros(3)
        mag = D.k_repel_obstacle * (1.0 / d_s[near] - 1.0 / D.repel_influence_m) / d_s[near] ** 2
        f = (mag[:, None] * diff[near] / dist[near, None]).sum(axis=0)
        n = float(np.linalg.norm(f))
        return f * (8.0 / n) if n > 8.0 else f

    def _forces(self) -> np.ndarray:
        D, w = cfg.DRONE, self.world
        f = np.zeros(3)
        if self.frontier_target is not None:
            d = self.frontier_target - self.pos
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                f += D.k_attract_frontier * d / n * min(1.0, n / 3.0)
        for p, wt in self.attractors:
            d = p - self.pos
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                f += D.k_attract_thermal * wt * d / n * min(1.0, n / 3.0)
        rep = self._repulsion()
        f += rep + 0.6 * np.array([-rep[1], rep[0], 0.0])      # slide around obstacles (breaks symmetric traps)
        gap = w.ceiling_at(self.pos[0], self.pos[1]) - self.pos[2]
        if gap < 2.5:
            f[2] -= D.k_ceiling_repel * (1.0 / max(gap, 0.2) - 1.0 / 2.5)
        if self._escape_steps > 0:
            f += self._escape_dir
            self._escape_steps -= 1
        return f

    # ---- integration ----
    def step(self, dt: float = cfg.DT_S) -> List[Dict[str, Any]]:
        ev: List[Dict[str, Any]] = []
        if not self.alive:
            return ev
        w, D = self.world, cfg.DRONE
        if self.battery_pct <= 0.0:
            self.status = "battery_depleted"
            self.vel[:] = 0.0
            return ev
        tgt = self.frontier_target
        reached = tgt is not None and math.hypot(tgt[0] - self.pos[0], tgt[1] - self.pos[1]) < 2.5
        self._target_age += 1
        if tgt is None or reached or self._target_age > 60:
            self._update_frontier()
            self._target_age = 0

        if self.frontier_target is not None:                 # local-minimum guard
            dd = float(np.linalg.norm(self.frontier_target - self.pos))
            if dd < self._target_best - 0.5:
                self._target_best, self._no_prog = dd, 0
            else:
                self._no_prog += 1
            if self._no_prog > 40:
                self._blacklist.append((float(self.frontier_target[0]), float(self.frontier_target[1]),
                                        w.step_count + 400))
                self._update_frontier()
                self._target_age = 0
        f = self._forces()
        fn = float(np.linalg.norm(f))
        v_des = f / fn * D.max_speed_mps * min(1.0, fn / 1.5) if fn > 1e-6 else np.zeros(3)
        a_cmd = 2.5 * (v_des - self.vel)
        an = float(np.linalg.norm(a_cmd))
        if an > D.max_accel_mps2:
            a_cmd *= D.max_accel_mps2 / an
        wind = w.wind_at(*self.pos)
        a_wind = D.wind_gain * wind
        acc = a_cmd - D.drag_coeff * self.vel + a_wind
        self.vel = self.vel + acc * dt
        sp = float(np.linalg.norm(self.vel))
        if sp > 1.25 * D.max_speed_mps:
            self.vel *= 1.25 * D.max_speed_mps / sp
        new = self.pos + self.vel * dt
        self.wind_drift_m += float(np.linalg.norm(a_wind)) * dt * dt

        if w.is_solid(*new):
            self.alive, self.loss_reason, self.status = False, "collision", "lost"
            ev.append({"type": "drone_lost", "reason": "collision", "x": round(float(new[0]), 1),
                       "y": round(float(new[1]), 1), "z": round(float(new[2]), 1)})
            return ev
        z_lo = max(D.z_min_m, w.elevation_at(new[0], new[1]) + 0.7)
        z_hi = min(D.z_max_m, w.ceiling_at(new[0], new[1]) - 0.6)
        if z_hi < z_lo:
            z_hi = z_lo
        clamped = float(np.clip(new[2], z_lo, z_hi))
        if clamped != new[2]:
            self.vel[2] = 0.0
        new[2] = clamped
        self.distance_m += float(np.linalg.norm(new - self.pos))
        self.pos = new

        # stuck detection -> random escape push (local-minimum breaker)
        if float(np.linalg.norm(self.vel)) < 0.08:
            self._stuck += 1
            if self._stuck > 10 and self._escape_steps == 0:
                a = float(self.rng.uniform(0, 2 * math.pi))
                self._escape_dir = np.array([math.cos(a), math.sin(a), 0.2]) * 1.5
                self._escape_steps = 15
                self._stuck = 0
        else:
            self._stuck = 0

        power = D.hover_power_w + D.k_speed_w_per_mps * float(np.linalg.norm(self.vel))
        self.energy_used_wh += power * dt / 3600.0
        inst = w.instability_at(self.pos[0], self.pos[1])
        self.risk_exposure += inst * dt
        if self.rng.random() < structural_loss_probability(inst, dt, DRONE_CEILING_HAZARD_FACTOR):
            self.alive, self.loss_reason, self.status = False, "structural_collapse", "lost"
            ev.append({"type": "drone_lost", "reason": "structural_collapse"})
        if w.step_count % 2 == 0 and len(self.trail) < 4000:
            self.trail.append((round(float(self.pos[0]), 2), round(float(self.pos[1]), 2), round(float(self.pos[2]), 2)))
        return ev

    def apply_aftershock(self, event: Dict[str, Any]) -> bool:
        if not self.alive:
            return False
        p = aftershock_crush_probability(event, self.pos[0], self.pos[1],
                                         self.world.instability_at(self.pos[0], self.pos[1]), 0.5)
        if self.rng.random() < p:
            self.alive, self.loss_reason, self.status = False, "aftershock_debris", "lost"
            return True
        return False

    def telemetry(self) -> Dict[str, Any]:
        return {"x": round(float(self.pos[0]), 2), "y": round(float(self.pos[1]), 2),
                "z": round(float(self.pos[2]), 2), "vx": round(float(self.vel[0]), 2),
                "vy": round(float(self.vel[1]), 2), "vz": round(float(self.vel[2]), 2),
                "battery_pct": round(self.battery_pct, 1), "status": self.status, "alive": self.alive,
                "loss_reason": self.loss_reason, "distance_m": round(self.distance_m, 1),
                "energy_wh": round(self.energy_used_wh, 3), "mapped_fraction": round(self.map.fraction(), 3),
                "risk_exposure": round(self.risk_exposure, 2)}