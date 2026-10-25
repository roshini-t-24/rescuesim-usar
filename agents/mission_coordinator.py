"""Closed-loop multi-agent mission coordinator: Sense -> Plan -> Act.

Policies: greedy (no risk weights), static (fixed conservative A*), adaptive (Nemotron planner + reflex agent).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import settings as cfg
from simulator.disaster_world import DisasterWorld
from simulator.robot_kinematics import MicroDrone, QuadrupedRover, SharedMap
from simulator.sensors import SensorSuite, rover_self_noise_db, triangulate_bearings
from agents.nemotron_orchestrator import NemotronOrchestrator

POLICIES = ("greedy", "static", "adaptive")

ARRIVE_RADIUS_M = 2.0
RESCUE_RADIUS_M = 4.0
LOCATE_RADIUS_M = 4.5
WAYPOINT_RADIUS_M = 3.5
GATE_THERMAL_M = 3.5
GATE_ACOUSTIC_M = 6.0
CONFIRM_CONF = 0.60
VERIFY_CONF = 0.28
REFLEX_BAND = (0.25, 0.60)
REFLEX_COOLDOWN_STEPS = 60
MAX_REFLEX_CALLS = 10
HOT_DECOY_TEMP_C = 44.0
PERIODIC_REPLAN_STEPS = 120
STUCK_WINDOW_STEPS = 60
STUCK_MIN_MOVE_M = 0.6
SKIP_STEPS = 80
SHELTER_PATIENCE_STEPS = 240
SHELTER_ENABLED = __import__("os").environ.get("RESCUESIM_SHELTER", "0") == "1"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def cfg_entry_xy() -> Tuple[float, float]:
    from simulator.disaster_world import ENTRY_XY
    return float(ENTRY_XY[0]), float(ENTRY_XY[1])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    return str(o)


def _first(sources: Sequence[Any], names: Sequence[str], default: Any = None) -> Any:
    for src in sources:
        if src is None:
            continue
        for n in names:
            if isinstance(src, dict):
                if n in src and src[n] is not None:
                    return src[n]
            else:
                v = getattr(src, n, None)
                if v is not None and not callable(v):
                    return v
    return default


def _xyz(robot: Any, tele: Dict[str, Any]) -> Tuple[float, float, float]:
    for key in ("pos", "position", "xyz"):
        v = _first((tele, robot), (key,))
        if v is not None:
            try:
                a = np.asarray(v, dtype=float).ravel()
                if a.size >= 2:
                    return float(a[0]), float(a[1]), (float(a[2]) if a.size > 2 else 0.0)
            except (TypeError, ValueError):
                pass
    x = _first((tele, robot), ("x",))
    y = _first((tele, robot), ("y",))
    if x is not None and y is not None:
        return float(x), float(y), float(_first((tele, robot), ("z",), 0.0))
    st = _first((robot,), ("state",))
    if st is not None:
        a = np.asarray(st, dtype=float).ravel()
        return float(a[0]), float(a[1]), float(a[2])
    raise AttributeError("cannot read robot position from telemetry or attributes")


def _battery_pct(robot: Any, tele: Dict[str, Any]) -> float:
    return float(_first((tele, robot), ("battery_pct", "battery"), 100.0))


def _energy_used_wh(robot: Any, tele: Dict[str, Any], capacity_wh: float) -> float:
    used = _first((tele, robot), ("energy_used_wh", "energy_consumed_wh", "wh_used"))
    if used is not None:
        return float(used)
    return max(0.0, (100.0 - _battery_pct(robot, tele)) / 100.0 * capacity_wh)


def _ev_type(ev: Any) -> str:
    if not isinstance(ev, dict):
        return str(ev).lower()
    return str(ev.get("type") or ev.get("event") or ev.get("kind") or "").lower()


def _lost(robot: Any, tele: Dict[str, Any], events: Sequence[Any]) -> bool:
    flag = _first((tele, robot), ("lost", "is_lost", "destroyed", "crushed"))
    if isinstance(flag, (bool, np.bool_, int)) and bool(flag):
        return True
    alive = _first((tele, robot), ("alive",))
    if isinstance(alive, (bool, np.bool_)) and not bool(alive):
        return True
    status = str(_first((tele, robot), ("status",), "")).lower()
    if any(k in status for k in ("lost", "crush", "destroy")):
        return True
    for ev in events:
        t = _ev_type(ev)
        if any(k in t for k in ("lost", "crush", "destroy")):
            return True
    return False


def sample_hazards(seed: int) -> Dict[str, Any]:
    """Seed-deterministic hazard modifiers used for Monte Carlo scenarios."""
    rng = np.random.default_rng(int(seed) + 7919)
    mods = dict(cfg.OFFLINE_HAZARD_PROFILE)
    draws = {
        "seismic_aftershock_prob": rng.uniform(0.08, 0.45),
        "wind_turbulence_mps": rng.uniform(0.5, 8.0),
        "ambient_temp_c": rng.uniform(2.0, 34.0),
        "smoke_toxicity_index": rng.uniform(0.15, 0.9),
        "fema_safety_margin_m": rng.uniform(1.0, 4.0),
    }
    for key, val in draws.items():
        lo, hi = cfg.HAZARD_BOUNDS.get(key, (-math.inf, math.inf))
        mods[key] = float(min(max(val, lo), hi))
    if "source_urls" in mods:
        mods["source_urls"] = []
    return mods


# --------------------------------------------------------------------------- #
# Track fusion
# --------------------------------------------------------------------------- #
@dataclass
class Track:
    tid: int
    x: float
    y: float
    z: float = 0.0
    est_temp_c: float = 0.0
    snr_db: float = 0.0
    smoke_t: float = 1.0
    n_obs: int = 0
    acoustic_hits: int = 0
    confidence: float = 0.0
    label: str = "uncertain"
    rescued: bool = False
    weight: float = 0.0
    bearings: List[Tuple[float, float, float]] = field(default_factory=list)
    skip_until: int = 0
    reflex_step: int = -10**9
    reflex_label: str = ""
    reflex_hits: int = 0
    rejected: bool = False
    sid: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tid": self.tid,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "z": round(self.z, 2),
            "est_temp_c": round(self.est_temp_c, 2),
            "snr_db": round(self.snr_db, 2),
            "smoke_t": round(self.smoke_t, 3),
            "n_obs": self.n_obs,
            "acoustic_hits": self.acoustic_hits,
            "confidence": round(self.confidence, 3),
            "label": self.label,
            "rescued": self.rescued,
        }


# --------------------------------------------------------------------------- #
# Coordinator
# --------------------------------------------------------------------------- #
class MissionCoordinator:
    def __init__(
        self,
        policy: str = "adaptive",
        seed: int = 0,
        hazards: Optional[Dict[str, Any]] = None,
        num_survivors: int = cfg.DEFAULT_NUM_SURVIVORS,
        max_steps: Optional[int] = None,
        dt: float = cfg.DT_S,
        orchestrator: Optional[NemotronOrchestrator] = None,
        record: bool = False,
        frame_every: int = 4,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.policy = policy
        self.seed = int(seed)
        self.dt = float(dt)
        self.max_steps = int(max_steps or cfg.MAX_MISSION_STEPS)
        self.hazards: Dict[str, Any] = dict(hazards) if hazards is not None else sample_hazards(seed)
        self.record = record
        self.frame_every = max(1, frame_every)

        self.world = DisasterWorld(seed=self.seed, num_survivors=num_survivors, hazard_modifiers=self.hazards)
        nx = int(round(cfg.WORLD_X_M / cfg.CELL_SIZE_M))
        ny = int(round(cfg.WORLD_Y_M / cfg.CELL_SIZE_M))
        self.smap = SharedMap(nx, ny, cfg.CELL_SIZE_M)
        self.sensors = SensorSuite(self.world, rng=np.random.default_rng(self.seed + 11))

        self.weights = cfg.POLICY_BASELINE_GREEDY if policy == "greedy" else cfg.POLICY_STATIC_CONSERVATIVE
        self.rover = QuadrupedRover(self.world, self.smap, weights=self.weights, rng=np.random.default_rng(self.seed + 21))
        self.drone = MicroDrone(self.world, self.smap, rng=np.random.default_rng(self.seed + 31))
        self.orch = orchestrator if orchestrator is not None else (NemotronOrchestrator() if policy == "adaptive" else None)

        self.k = 0
        self.t = 0.0
        self.done = False
        self.end_reason = ""
        self.tracks: Dict[int, Track] = {}
        self._next_tid = 1
        self.target: Optional[Dict[str, Any]] = None
        self.order: List[int] = []
        self._retarget = False
        self._pending_trigger: Optional[str] = None
        self.last_plan_k = -10**9
        self.plans: List[Dict[str, Any]] = []
        self.plan_modes: Counter = Counter()
        self.reflex_calls = 0
        self.aftershocks = 0
        self.event_types: Counter = Counter()
        self.exposure_by_kind: Counter = Counter()
        self._last_kind = "none"
        self.loss_kind: Optional[str] = None
        self.log: List[Dict[str, Any]] = []
        self.frames: List[Dict[str, Any]] = []
        self.r_trail: Deque[Tuple[float, float, float]] = deque(maxlen=600)
        self.d_trail: Deque[Tuple[float, float, float]] = deque(maxlen=600)

        self.rover_lost = False
        self.drone_lost = False
        self.rover_lost_t: Optional[float] = None
        self.drone_lost_t: Optional[float] = None
        self.risk_exposure = 0.0
        self.first_located_s: Optional[float] = None
        self.first_rescued_s: Optional[float] = None

        self.d_yaw = 0.0
        self.r_sweep = 0.0
        self._ac_votes = [0, 0]
        self._ac_rel = False
        self._stuck_ref: Tuple[float, float, int] = (0.0, 0.0, 0)

        self.waypoints = [
            (float(x), float(y))
            for x in np.arange(6.0, cfg.WORLD_X_M - 3.0, 10.0)
            for y in np.arange(6.0, cfg.WORLD_Y_M - 3.0, 10.0)
        ]
        self.wp_visited: set = set()
        self._idle_since: Optional[int] = None

        self.r_tele: Dict[str, Any] = {}
        self.d_tele: Dict[str, Any] = {}
        self.r_xyz = (0.0, 0.0, 0.0)
        self.d_xyz = (0.0, 0.0, 0.0)
        self.r_yaw = 0.0
        self.r_speed = 0.0
        self._refresh()
        self._stuck_ref = (self.r_xyz[0], self.r_xyz[1], 0)
        self._say("coordinator", f"mission start policy={policy} seed={seed} survivors={len(self.world.survivors)}")

    # ------------------------------------------------------------------ utils
    def _say(self, src: str, msg: str) -> None:
        self.log.append({"t": round(self.t, 1), "src": src, "msg": msg})
        if len(self.log) > 400:
            del self.log[:100]

    def _refresh(self) -> None:
        self.r_tele = _clean(self.rover.telemetry())
        self.d_tele = _clean(self.drone.telemetry())
        self.r_xyz = _xyz(self.rover, self.r_tele)
        self.d_xyz = _xyz(self.drone, self.d_tele)
        self.r_yaw = float(_first((self.r_tele, self.rover), ("theta", "yaw", "heading"), 0.0))
        self.r_speed = abs(float(_first((self.r_tele, self.rover), ("v", "speed", "speed_mps"), 0.0)))

    def _mark_loss(self, which: str, events: Sequence[Any]) -> None:
        if which == "rover" and not self.rover_lost:
            if _lost(self.rover, self.r_tele, events):
                self.rover_lost, self.rover_lost_t = True, self.t
                self.loss_kind = self._last_kind
                self.target = None
                self._say("rover", "ROVER LOST (structural collapse / hazard)")
        if which == "drone" and not self.drone_lost:
            if _lost(self.drone, self.d_tele, events):
                self.drone_lost, self.drone_lost_t = True, self.t
                self._say("drone", "DRONE LOST")

    def _hazards_for_plan(self) -> Dict[str, Any]:
        hz = _clean(dict(self.hazards))
        hz["aftershocks_so_far"] = self.aftershocks
        return hz

    def _world_summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for fn in (self.world.summary, self.world.physics_snapshot):
            try:
                out.update(_clean(fn()))
            except Exception:  # noqa: BLE001
                continue
        out["map_explored_fraction"] = float(self.smap.fraction())
        return out

    # ------------------------------------------------------------------ world events
    def _on_world_event(self, ev: Any) -> None:
        etype = _ev_type(ev)
        self.event_types[f"world:{etype}"] += 1
        if "aftershock" in etype:
            self.aftershocks += 1
            mag = ev.get("magnitude") if isinstance(ev, dict) else None
            self._say("world", f"AFTERSHOCK magnitude={mag}")
            for robot in (self.rover, self.drone):
                try:
                    robot.apply_aftershock(ev)
                except Exception as exc:  # noqa: BLE001
                    self._say("coordinator", f"apply_aftershock failed: {type(exc).__name__}")
            self._pending_trigger = "aftershock"
            self._retarget = True

    # ------------------------------------------------------------------ SENSE
    def _new_track(self, x: float, y: float) -> Track:
        tr = Track(tid=self._next_tid, x=x, y=y, z=float(self.world.elevation_at(x, y)))
        self._next_tid += 1
        self.tracks[tr.tid] = tr
        return tr

    def _nearest_track(self, x: float, y: float, gate: float) -> Optional[Track]:
        best, best_d = None, gate
        for tr in self.tracks.values():
            d = math.hypot(tr.x - x, tr.y - y)
            if d <= best_d:
                best, best_d = tr, d
        return best

    def _fuse_thermal(self, det: Any, src: str) -> None:
        px, py, pz = det.est_position
        tr = self._nearest_track(px, py, GATE_THERMAL_M)
        w = max(0.2, float(det.snr_db) / 10.0)
        if tr is None:
            tr = self._new_track(px, py)
            tr.z = float(pz)
            tr.est_temp_c = float(det.est_temp_c)
            tr.weight = 0.0
            self._say(src, f"new thermal contact T{tr.tid} at ({px:.1f},{py:.1f}) snr={det.snr_db:.1f}dB est={det.est_temp_c:.1f}C")
        tot = tr.weight + w
        tr.x = (tr.x * tr.weight + px * w) / tot
        tr.y = (tr.y * tr.weight + py * w) / tot
        tr.z = (tr.z * tr.weight + float(pz) * w) / tot
        tr.est_temp_c = (tr.est_temp_c * tr.weight + float(det.est_temp_c) * w) / tot
        tr.smoke_t = (tr.smoke_t * tr.weight + float(det.smoke_transmittance) * w) / tot
        tr.snr_db = max(tr.snr_db, float(det.snr_db))
        tr.weight = tot
        tr.n_obs += 1

    def _acoustic_bearing(self, det: Any, heading: float, rx: float, ry: float) -> float:
        th = float(det.bearing_rad)
        tid = int(getattr(det, "truth_id", -1))
        if tid >= 0:
            s = self.world.survivor_by_id(tid)
            if s is not None:
                true_b = math.atan2(s.y - ry, s.x - rx)
                self._ac_votes[0 if abs(_wrap(th - true_b)) <= abs(_wrap(th + heading - true_b)) else 1] += 1
                self._ac_rel = self._ac_votes[1] > self._ac_votes[0]
        return _wrap(th + heading) if self._ac_rel else th

    def _fuse_acoustic(self, det: Any, heading: float) -> None:
        rx, ry, _ = self.r_xyz
        bearing = self._acoustic_bearing(det, heading, rx, ry)
        r = max(1.0, float(det.range_est_m))
        px, py = rx + r * math.cos(bearing), ry + r * math.sin(bearing)
        tr = self._nearest_track(px, py, GATE_ACOUSTIC_M)
        if tr is None:
            tr = self._new_track(px, py)
            self._say("rover", f"acoustic contact T{tr.tid} bearing={math.degrees(bearing):.0f}deg range~{r:.1f}m level={det.level_db:.1f}dB")
        tr.acoustic_hits += 1
        tr.bearings.append((rx, ry, bearing))
        if len(tr.bearings) >= 3:
            try:
                res = triangulate_bearings(tr.bearings[-10:])
            except Exception:  # noqa: BLE001
                res = None
            if res is not None:
                (tx, ty), _q = res
                if math.hypot(tx - tr.x, ty - tr.y) < 8.0:
                    tr.x, tr.y = 0.5 * tr.x + 0.5 * tx, 0.5 * tr.y + 0.5 * ty

    def _confidence(self, tr: Track) -> float:
        ambient = float(self.hazards.get("ambient_temp_c", cfg.SENSORS.ambient_temp_default_c))
        thermal = min(1.0, max(0.0, (tr.snr_db - 6.0) / 18.0)) * min(1.0, tr.n_obs / 3.0)
        if tr.est_temp_c > HOT_DECOY_TEMP_C and tr.acoustic_hits == 0:
            thermal *= 0.25
        elif tr.n_obs > 0 and tr.est_temp_c < ambient + 1.5 and tr.acoustic_hits == 0:
            thermal *= 0.7
        acoustic = 1.0 - 0.55 ** tr.acoustic_hits
        return float(1.0 - (1.0 - 0.8 * thermal) * (1.0 - acoustic))

    def _reflex(self, tr: Track) -> None:
        if self.orch is None or self.reflex_calls >= MAX_REFLEX_CALLS:
            return
        rx, ry, _ = self.r_xyz
        features = {
            "tid": tr.tid,
            "est_temp_c": tr.est_temp_c,
            "snr_db": tr.snr_db,
            "smoke_transmittance": tr.smoke_t,
            "acoustic_hits": tr.acoustic_hits,
            "n_obs": tr.n_obs,
            "confidence": tr.confidence,
            "range_m": math.hypot(tr.x - rx, tr.y - ry),
            "ambient_temp_c": float(self.hazards.get("ambient_temp_c", 18.0)),
            "smoke_toxicity_index": float(self.hazards.get("smoke_toxicity_index", 0.3)),
            "x": tr.x,
            "y": tr.y,
        }
        tr.reflex_step = self.k
        tr.reflex_hits = tr.acoustic_hits
        self.reflex_calls += 1
        try:
            verdict = self.orch.classify_anomaly(_clean(features))
        except Exception as exc:  # noqa: BLE001
            self._say("reflex", f"T{tr.tid} reflex failed: {type(exc).__name__}")
            return
        lab = str(verdict.label).lower()
        if float(verdict.confidence) >= 0.5:
            if any(w in lab for w in ("decoy", "fire", "false", "hotspot", "machine", "non", "equipment", "electrical")):
                tr.reflex_label = "decoy"
            elif any(w in lab for w in ("survivor", "human", "victim", "person")):
                tr.reflex_label = "survivor"
        self._say("reflex", f"T{tr.tid} -> {verdict.label} ({verdict.confidence:.2f}, {verdict.mode}/{verdict.model}) {verdict.rationale[:90]}")

    def _update_tracks(self) -> None:
        for tr in self.tracks.values():
            if tr.rescued:
                tr.label = "rescued"
                continue
            conf = self._confidence(tr)
            if tr.reflex_label == "decoy" and tr.acoustic_hits >= tr.reflex_hits + 2:
                tr.reflex_label = ""
            if (
                self.policy == "adaptive"
                and tr.label == "uncertain"
                and not tr.reflex_label
                and REFLEX_BAND[0] <= conf < REFLEX_BAND[1]
                and tr.est_temp_c >= float(self.hazards.get("ambient_temp_c", 18.0)) + 8.0
                and tr.n_obs >= 2
                and self.k - tr.reflex_step >= REFLEX_COOLDOWN_STEPS
            ):
                self._reflex(tr)
            if tr.reflex_label == "survivor":
                conf = max(conf, 0.65)
            tr.confidence = conf
            if tr.rejected or tr.reflex_label == "decoy":
                tr.label = "decoy"
            elif conf >= CONFIRM_CONF:
                tr.label = "survivor"
            elif tr.est_temp_c > HOT_DECOY_TEMP_C and tr.acoustic_hits == 0 and tr.n_obs >= 2:
                tr.label = "decoy"
            else:
                tr.label = "uncertain"
            if tr.label == "survivor" and tr.sid < 0:
                self._locate(tr)

    def _locate(self, tr: Track) -> None:
        best, best_d = None, LOCATE_RADIUS_M
        for s in self.world.survivors:
            if s.rescued or s.vitality <= 0:
                continue
            d = math.hypot(s.x - tr.x, s.y - tr.y)
            if d <= best_d:
                best, best_d = s, d
        if best is None:
            return
        tr.sid = best.sid
        if not best.located:
            self.world.mark_located(best.sid)
            if self.first_located_s is None:
                self.first_located_s = self.t
            self._say("coordinator", f"SURVIVOR CONFIRMED T{tr.tid} at ({tr.x:.1f},{tr.y:.1f}) conf={tr.confidence:.2f}")

    def _sense(self) -> None:
        if not self.drone_lost and self.k % 2 == 0:
            self.d_yaw = _wrap(self.d_yaw + 2.0)
            try:
                scan = self.sensors.lidar_scan(self.d_xyz, self.d_yaw)
                self.drone.update_map(scan)
            except Exception as exc:  # noqa: BLE001
                self._say("drone", f"lidar error {type(exc).__name__}")
            for det in self.sensors.thermal_scan(self.d_xyz, self.d_yaw, fov_deg=120.0, max_range_m=25.0):
                self._fuse_thermal(det, "drone")
        if not self.rover_lost and self.k % 3 == 0:
            rx, ry, rz = self.r_xyz
            for det in self.sensors.acoustic_scan(
                self.r_xyz, self.r_yaw, own_noise_db=rover_self_noise_db(self.r_speed)
            ):
                self._fuse_acoustic(det, self.r_yaw)
            self.r_sweep = _wrap(self.r_sweep + 2.0)
            for det in self.sensors.thermal_scan((rx, ry, rz + 0.5), self.r_sweep, fov_deg=120.0, max_range_m=12.0):
                self._fuse_thermal(det, "rover")
        self._update_tracks()

    # ------------------------------------------------------------------ PLAN
    def _viable(self) -> List[Track]:
        return [
            t for t in self.tracks.values()
            if not t.rescued and t.label != "decoy" and t.confidence >= VERIFY_CONF and self.k >= t.skip_until
        ]

    def _call_plan(self, trigger: str) -> None:
        if self.orch is None:
            return
        cands = [{**t.to_dict(), "id": t.tid} for t in self._viable()]
        rx, ry, _ = self.r_xyz
        robots = {
            "rover": {**self.r_tele, "lost": self.rover_lost},
            "drone": {**self.d_tele, "lost": self.drone_lost},
        }
        try:
            dec = self.orch.plan(
                trigger=trigger,
                hazards=self._hazards_for_plan(),
                world=self._world_summary(),
                candidates=cands,
                rover_xy=(rx, ry),
                current_weights=self.weights,
                robots=_clean(robots),
            )
        except Exception as exc:  # noqa: BLE001
            self._say("planner", f"plan failed ({exc!r}); keeping weights")
            self.last_plan_k = self.k
            return
        self.last_plan_k = self.k
        self.weights = dec.weights
        self.order = [int(i) for i in dec.rover_target_order]
        try:
            self.rover.set_weights(dec.weights)
        except Exception as exc:  # noqa: BLE001
            self._say("planner", f"set_weights failed ({type(exc).__name__})")
        self.plan_modes[dec.mode] += 1
        self.plans.append(_clean(dec.to_dict()))
        self._retarget = True
        self._say(
            "planner",
            f"[{trigger}] {dec.mode}/{dec.model} a={dec.weights.alpha_slope:.2f} b={dec.weights.beta_collapse:.2f} "
            f"g={dec.weights.gamma_smoke:.2f} order={self.order} {dec.rationale[:110]}",
        )

    def _maybe_plan(self) -> None:
        if self.policy != "adaptive" or self.rover_lost:
            return
        due = self.k - self.last_plan_k >= cfg.REPLAN_COOLDOWN_STEPS
        if self.k == 1:
            self._call_plan("mission_start")
        elif self._pending_trigger and due:
            trig, self._pending_trigger = self._pending_trigger, None
            self._call_plan(trig)
        elif due and self.k - self.last_plan_k >= PERIODIC_REPLAN_STEPS:
            self._call_plan("periodic")
        elif due and any(t.label == "survivor" and t.tid not in self.order for t in self._viable()):
            self._call_plan("new_candidate")

    def _choose_track(self, viable: List[Track]) -> Track:
        rx, ry, _ = self.r_xyz
        if self.policy == "adaptive" and self.order:
            by_id = {t.tid: t for t in viable}
            for tid in self.order:
                if tid in by_id:
                    return by_id[tid]
        if self.policy == "greedy":
            return min(viable, key=lambda t: math.hypot(t.x - rx, t.y - ry))
        return min(viable, key=lambda t: math.hypot(t.x - rx, t.y - ry) / (0.3 + t.confidence))

    def _next_waypoint(self) -> Optional[int]:
        rx, ry, _ = self.r_xyz
        best, best_d = None, 1e9
        for i, (x, y) in enumerate(self.waypoints):
            if i in self.wp_visited:
                continue
            d = math.hypot(x - rx, y - ry)
            if d < best_d:
                best, best_d = i, d
        return best

    def _go_shelter(self) -> bool:
        rx, ry, _ = self.r_xyz
        pts = [(p[0], p[1]) for p in list(self.r_trail)[::5]] + [cfg_entry_xy()]
        best = min(
            pts,
            key=lambda q: float(self.world.instability_at(q[0], q[1])) + 0.01 * math.hypot(q[0] - rx, q[1] - ry),
        )
        if math.hypot(best[0] - rx, best[1] - ry) <= 1.5:
            self.target = {"kind": "shelter", "xy": best}
            return True
        if self.rover.plan_to(best):
            self.target = {"kind": "shelter", "xy": best}
            self._say("coordinator", f"rover -> shelter ({best[0]:.1f},{best[1]:.1f}) while drone scouts")
            return True
        return False

    def _assign_target(self) -> None:
        if self.rover_lost:
            self.target = None
            return
        valid = self.target is not None
        viable_ids = {t.tid for t in self._viable()}
        if valid and self.target["kind"] == "track" and self.target["tid"] not in viable_ids:
            valid = False
        if valid and self.target["kind"] in ("wp", "shelter") and viable_ids:
            valid = False
        if valid and self._retarget and self.policy == "adaptive" and self.target["kind"] == "track":
            valid = False
        if valid:
            return
        for _ in range(6):
            viable = self._viable()
            if viable:
                self._idle_since = None
                tr = self._choose_track(viable)
                if self.rover.plan_to((tr.x, tr.y)):
                    self.target = {"kind": "track", "tid": tr.tid, "xy": (tr.x, tr.y)}
                    self._say("coordinator", f"rover -> T{tr.tid} ({tr.x:.1f},{tr.y:.1f}) conf={tr.confidence:.2f} [{tr.label}]")
                    break
                tr.skip_until = self.k + SKIP_STEPS
                self._say("rover", f"T{tr.tid} unreachable, deferring")
                continue
            if self._idle_since is None:
                self._idle_since = self.k
            if (
                self.policy == "adaptive"
                and SHELTER_ENABLED
                and not self.drone_lost
                and self.k - self._idle_since < SHELTER_PATIENCE_STEPS
                and self._go_shelter()
            ):
                break
            wi = self._next_waypoint()
            if wi is None:
                self.target = None
                break
            if self.rover.plan_to(self.waypoints[wi]):
                self.target = {"kind": "wp", "idx": wi, "xy": self.waypoints[wi]}
                break
            self.wp_visited.add(wi)
        else:
            self.target = None
        self._retarget = False
        self._stuck_ref = (self.r_xyz[0], self.r_xyz[1], self.k)

    # ------------------------------------------------------------------ ACT
    def _on_arrival(self, tr: Track) -> None:
        rx, ry, _ = self.r_xyz
        best, best_d = None, RESCUE_RADIUS_M
        for s in self.world.survivors:
            if s.rescued or s.vitality <= 0:
                continue
            d = math.hypot(s.x - rx, s.y - ry)
            if d <= best_d:
                best, best_d = s, d
        self.target = None
        if best is None:
            tr.rejected = True
            tr.label = "decoy"
            self._say("rover", f"T{tr.tid} on-site check: no casualty (false positive)")
            return
        if not best.located:
            self.world.mark_located(best.sid)
            if self.first_located_s is None:
                self.first_located_s = self.t
        pod = False
        try:
            pod = bool(self.rover.deliver_pod())
        except Exception:  # noqa: BLE001
            pod = False
        if self.world.rescue_survivor(best.sid):
            tr.rescued, tr.label, tr.sid = True, "rescued", best.sid
            if self.first_rescued_s is None:
                self.first_rescued_s = self.t
            self._say("rover", f"SURVIVOR S{best.sid} RESCUED ({'pod delivered' if pod else 'no pod left, marked for extraction'})")
        else:
            tr.skip_until = self.k + SKIP_STEPS

    def _act(self) -> None:
        if not self.rover_lost:
            self._last_kind = self.target["kind"] if self.target else "none"
            ev = self.rover.step(self.dt) or []
            for e in ev:
                self.event_types[f"rover:{_ev_type(e)}"] += 1
            self.r_tele = _clean(self.rover.telemetry())
            self.r_xyz = _xyz(self.rover, self.r_tele)
            self.r_yaw = float(_first((self.r_tele, self.rover), ("theta", "yaw", "heading"), 0.0))
            self.r_speed = abs(float(_first((self.r_tele, self.rover), ("v", "speed", "speed_mps"), 0.0)))
            self._mark_loss("rover", ev)
            if not self.rover_lost:
                rx, ry, _ = self.r_xyz
                self.risk_exposure += float(self.world.instability_at(rx, ry)) * self.dt
                self.exposure_by_kind[self._last_kind] += float(self.world.instability_at(rx, ry)) * self.dt
                tgt = self.target
                if tgt is not None:
                    d = math.hypot(tgt["xy"][0] - rx, tgt["xy"][1] - ry)
                    if tgt["kind"] == "track":
                        tr = self.tracks.get(tgt["tid"])
                        if tr is not None and d <= ARRIVE_RADIUS_M:
                            self._on_arrival(tr)
                    elif tgt["kind"] == "wp" and d <= WAYPOINT_RADIUS_M:
                        self.wp_visited.add(tgt["idx"])
                        self._idle_since = None
                        self.target = None
                    elif tgt["kind"] == "shelter" and d <= 1.5:
                        self.target = None
                if self.k - self._stuck_ref[2] >= STUCK_WINDOW_STEPS:
                    moved = math.hypot(rx - self._stuck_ref[0], ry - self._stuck_ref[1])
                    if self.target is not None and moved < STUCK_MIN_MOVE_M:
                        if self.target["kind"] == "track":
                            tr = self.tracks.get(self.target["tid"])
                            if tr is not None:
                                tr.skip_until = self.k + 2 * SKIP_STEPS
                        elif self.target["kind"] == "wp":
                            self.wp_visited.add(self.target["idx"])
                        self._say("rover", "stuck watchdog: abandoning target")
                        self.target = None
                    self._stuck_ref = (rx, ry, self.k)
        if not self.drone_lost:
            if self.k % 5 == 0:
                items = [
                    ((t.x, t.y, float(self.world.elevation_at(t.x, t.y)) + 1.5), 1.0 + 3.0 * t.confidence)
                    for t in self.tracks.values()
                    if not t.rescued and t.label != "decoy"
                ]
                try:
                    self.drone.set_thermal_attractors(items)
                except Exception:  # noqa: BLE001
                    pass
            ev = self.drone.step(self.dt) or []
            for e in ev:
                self.event_types[f"drone:{_ev_type(e)}"] += 1
            self.d_tele = _clean(self.drone.telemetry())
            self.d_xyz = _xyz(self.drone, self.d_tele)
            self._mark_loss("drone", ev)
        if self.k % 2 == 0:
            self.r_trail.append(self.r_xyz)
            self.d_trail.append(self.d_xyz)

    # ------------------------------------------------------------------ loop
    def _check_end(self) -> None:
        surv = self.world.survivors
        if all(s.rescued or s.vitality <= 0 for s in surv):
            self.done, self.end_reason = True, "all_survivors_resolved"
        elif self.rover_lost:
            self.done, self.end_reason = True, "rover_lost"
        elif _battery_pct(self.rover, self.r_tele) <= 0.0:
            self.done, self.end_reason = True, "rover_battery_depleted"
        elif self.k >= self.max_steps:
            self.done, self.end_reason = True, "time_limit"
        if self.done:
            self._say("coordinator", f"mission end: {self.end_reason}")

    def step(self) -> Dict[str, Any]:
        if self.done:
            return self.frame()
        self.k += 1
        self.t = self.k * self.dt
        for ev in self.world.step(self.dt) or []:
            self._on_world_event(ev)
        self._refresh()
        self._mark_loss("rover", [])
        self._mark_loss("drone", [])
        self._sense()
        self._maybe_plan()
        self._assign_target()
        self._act()
        self._check_end()
        fr = self.frame()
        if self.record and (self.k % self.frame_every == 0 or self.done):
            self.frames.append(fr)
        return fr

    def run(self) -> Dict[str, Any]:
        while not self.done:
            self.step()
        return self.metrics()

    # ------------------------------------------------------------------ outputs
    def frame(self) -> Dict[str, Any]:
        surv = self.world.survivors
        return {
            "step": self.k,
            "t": round(self.t, 1),
            "policy": self.policy,
            "done": self.done,
            "end_reason": self.end_reason,
            "rover": {
                "x": self.r_xyz[0], "y": self.r_xyz[1], "z": self.r_xyz[2], "yaw": self.r_yaw,
                "battery_pct": _battery_pct(self.rover, self.r_tele), "lost": self.rover_lost,
                "target": self.target["xy"] if self.target else None,
                "trail": [list(p) for p in list(self.r_trail)[-150:]],
            },
            "drone": {
                "x": self.d_xyz[0], "y": self.d_xyz[1], "z": self.d_xyz[2],
                "battery_pct": _battery_pct(self.drone, self.d_tele), "lost": self.drone_lost,
                "trail": [list(p) for p in list(self.d_trail)[-150:]],
            },
            "survivors": [_clean(s.to_dict()) for s in surv],
            "tracks": [t.to_dict() for t in self.tracks.values()],
            "weights": self.weights.as_dict(),
            "kpis": {
                "located": sum(1 for s in surv if s.located),
                "rescued": sum(1 for s in surv if s.rescued),
                "total": len(surv),
                "risk_exposure": round(self.risk_exposure, 2),
                "map_explored": round(float(self.smap.fraction()), 3),
                "aftershocks": self.aftershocks,
            },
            "log": self.log[-14:],
        }

    def metrics(self) -> Dict[str, Any]:
        surv = self.world.survivors
        total = len(surv)
        rescued = sum(1 for s in surv if s.rescued)
        located = sum(1 for s in surv if s.located or s.rescued)
        dead = sum(1 for s in surv if (not s.rescued) and s.vitality <= 0)
        energy = _energy_used_wh(self.rover, self.r_tele, cfg.ROVER.battery_capacity_wh) + _energy_used_wh(
            self.drone, self.d_tele, cfg.DRONE.battery_capacity_wh
        )
        return {
            "policy": self.policy,
            "seed": self.seed,
            "time_s": round(self.t, 2),
            "rescue_rate": (rescued / total) if total else 0.0,
            "energy_wh": round(energy, 3),
            "energy_wh_per_rescued": round(energy / rescued, 3) if rescued else None,
            "first_located_s": self.first_located_s,
            "first_rescued_s": self.first_rescued_s,
            "rover_risk_exposure": round(self.risk_exposure, 3),
            "survivors_total": total,
            "survivors_located": located,
            "survivors_rescued": rescued,
            "survivors_dead": dead,
            "rover_lost": self.rover_lost,
            "drone_lost": self.drone_lost,
            "robot_loss": bool(self.rover_lost or self.drone_lost),
            "end_reason": self.end_reason,
            "rover_loss_reason": self.r_tele.get("loss_reason") if self.rover_lost else None,
            "drone_loss_reason": self.d_tele.get("loss_reason") if self.drone_lost else None,
            "rover_lost_t": self.rover_lost_t,
            "rover_loss_target_kind": self.loss_kind,
            "exposure_by_kind": {k: round(v, 2) for k, v in self.exposure_by_kind.items()},
            "drone_lost_t": self.drone_lost_t,
            "aftershocks": self.aftershocks,
            "plans": len(self.plans),
            "reflex_calls": self.reflex_calls,
            "llm_mode": dict(self.plan_modes) if self.plan_modes else None,
            "final_weights": self.weights.as_dict(),
        }

    def debug_report(self) -> str:
        lines = [
            f"rover telemetry keys: {sorted(self.r_tele.keys())}",
            f"drone telemetry keys: {sorted(self.d_tele.keys())}",
            f"event types: {dict(self.event_types)}",
            f"acoustic bearing relative-to-heading: {self._ac_rel} votes(world,rel)={self._ac_votes}",
            f"tracks: {[t.to_dict() for t in self.tracks.values()]}",
            "survivors: " + str([(s.sid, round(s.x, 1), round(s.y, 1), round(s.vitality, 2), s.located, s.rescued) for s in self.world.survivors]),
            "last log:",
        ]
        lines += [f"  [{e['t']:>6}] {e['src']:<11} {e['msg']}" for e in self.log[-25:]]
        return "\n".join(lines)


def run_mission(
    policy: str = "adaptive",
    seed: int = 0,
    hazards: Optional[Dict[str, Any]] = None,
    num_survivors: int = cfg.DEFAULT_NUM_SURVIVORS,
    max_steps: Optional[int] = None,
    record: bool = False,
    orchestrator: Optional[NemotronOrchestrator] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    coord = MissionCoordinator(
        policy=policy, seed=seed, hazards=hazards, num_survivors=num_survivors,
        max_steps=max_steps, orchestrator=orchestrator, record=record,
    )
    metrics = coord.run()
    if verbose:
        print(coord.debug_report())
    if record:
        metrics["frames"] = coord.frames
    return metrics


def _main() -> None:
    ap = argparse.ArgumentParser(description="Run one RescueSim mission")
    ap.add_argument("--policy", choices=POLICIES + ("all",), default="all")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    for pol in (POLICIES if args.policy == "all" else (args.policy,)):
        coord = MissionCoordinator(policy=pol, seed=args.seed)
        print(json.dumps(coord.run(), indent=2, default=str))
        if args.debug:
            print(coord.debug_report())


if __name__ == "__main__":
    _main()