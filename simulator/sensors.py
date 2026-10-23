"""simulator/sensors.py

Signal-physics sensor models for RescueSim-USAR.

LiDAR     36-ray raycast, Beer-Lambert smoke attenuation  T = exp(-integral sigma_smoke ds)
Thermal   inverse-square law x smoke transmittance x rubble attenuation -> SNR (dB)
Acoustic  6-mic piezo ring (rover only): spherical spreading + rubble loss + multipath,
          bearing from a time-difference-of-arrival (TDOA) plane-wave fit, and a
          multi-position bearing intersection (`triangulate_bearings`).

Ground-truth fields named `truth_*` exist only for logging and scoring. Planning
agents must not read them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import settings as cfg
from simulator.disaster_world import DisasterWorld

THERMAL_MAX_RANGE_M: float = 25.0
THERMAL_FOV_DEG: float = 120.0
THERMAL_RUBBLE_E_FOLD_M: float = 0.5     # thermal signal e-folds every 0.5 m of rubble overburden
ACOUSTIC_ASSUMED_SOURCE_DB: float = 62.0  # mid-range prior used for the coarse level-based range
ACOUSTIC_TAPPING_PROB: float = 0.6        # tapping is bursty: probability a given scan hears it
ACOUSTIC_MAX_RANGE_M: float = 30.0
MULTIPATH_DELAY_STD_PER_M: float = 6.0e-5  # extra arrival-time scatter (s) per metre of rubble


# --------------------------------------------------------------------------
# Pure physics helpers (unit-testable)
# --------------------------------------------------------------------------
def beer_lambert_transmittance(optical_depth: float) -> float:
    """T = exp(-tau) where tau = integral(sigma ds)."""
    return float(math.exp(-max(0.0, optical_depth)))


def thermal_signal(delta_t_c: float, distance_m: float, smoke_tau: float, overburden_m: float = 0.0) -> float:
    """Noise-free thermal signal: ref * dT * (r0/r)^2 * exp(-tau_smoke) * exp(-overburden/e_fold)."""
    s = cfg.SENSORS
    d = max(distance_m, 0.1)
    return (s.thermal_ref_signal * max(delta_t_c, 0.0) * (s.thermal_ref_distance_m / d) ** 2
            * math.exp(-max(smoke_tau, 0.0)) * math.exp(-max(overburden_m, 0.0) / THERMAL_RUBBLE_E_FOLD_M))


def snr_db_from_signal(signal: float, noise_floor: Optional[float] = None) -> float:
    nf = cfg.SENSORS.thermal_noise_floor if noise_floor is None else noise_floor
    return 20.0 * math.log10(max(signal / nf, 1e-6))


def acoustic_received_db(source_db: float, distance_m: float, rubble_len_m: float) -> float:
    """Level at the array: L0 - 20*log10(r/r0) - rubble_loss_db_per_m * rubble_length."""
    s = cfg.SENSORS
    spread = s.acoustic_spreading_exponent * math.log10(max(distance_m, s.acoustic_ref_distance_m) / s.acoustic_ref_distance_m)
    return source_db - spread - s.acoustic_rubble_loss_db_per_m * max(rubble_len_m, 0.0)


def rover_self_noise_db(speed_mps: float) -> float:
    """Extra acoustic noise floor (dB) caused by the rover's own gait/motors."""
    return 10.0 * math.log10(1.0 + (abs(speed_mps) / 0.3) ** 2)


def triangulate_bearings(
    observations: Sequence[Tuple[float, float, float]],
    weights: Optional[Sequence[float]] = None,
    min_spread_deg: float = 10.0,
) -> Optional[Tuple[Tuple[float, float], float]]:
    """Least-squares intersection of bearing lines.

    observations: (origin_x, origin_y, bearing_rad) from different rover positions.
    Returns ((x, y), rms_residual_m) or None when the geometry is degenerate
    (fewer than two lines or nearly parallel bearings).
    """
    if len(observations) < 2:
        return None
    th = np.array([o[2] for o in observations], dtype=float)
    spread = np.degrees(np.ptp(np.mod(th, math.pi)))
    spread = min(spread, 180.0 - spread) if spread > 90.0 else spread
    if spread < min_spread_deg:
        return None
    A = np.column_stack([-np.sin(th), np.cos(th)])
    b = np.array([-math.sin(o[2]) * o[0] + math.cos(o[2]) * o[1] for o in observations])
    w = np.ones(len(b)) if weights is None else np.asarray(weights, dtype=float)
    sw = np.sqrt(w)
    sol, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
    rms = float(np.sqrt(np.mean((A @ sol - b) ** 2)))
    return (float(sol[0]), float(sol[1])), rms


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------
@dataclass
class LidarScan:
    origin: np.ndarray
    angles: np.ndarray                 # world-frame azimuth of each ray (rad)
    pitch_rad: float
    ranges: np.ndarray                 # metres, NaN where there was no valid return
    clear_range: np.ndarray            # distance each ray is known free of obstacles
    hit: np.ndarray                    # bool: valid surface return
    two_way_transmittance: np.ndarray  # T^2 at the surface (or at max range if no hit)
    smoke_limited: np.ndarray          # bool: beam died in smoke before reaching a surface

    def hit_points(self) -> np.ndarray:
        d = np.stack([np.cos(self.pitch_rad) * np.cos(self.angles),
                      np.cos(self.pitch_rad) * np.sin(self.angles),
                      np.full_like(self.angles, math.sin(self.pitch_rad))], axis=1)
        pts = self.origin[None, :] + d * np.nan_to_num(self.ranges, nan=0.0)[:, None]
        return pts[self.hit]

    def summary(self) -> Dict[str, Any]:
        r = self.ranges[self.hit]
        return {
            "returns": int(self.hit.sum()), "rays": int(self.hit.size),
            "smoke_limited_rays": int(self.smoke_limited.sum()),
            "mean_range_m": round(float(r.mean()), 2) if r.size else None,
            "min_range_m": round(float(r.min()), 2) if r.size else None,
        }


@dataclass
class ThermalDetection:
    range_m: float
    bearing_rad: float
    snr_db: float
    est_temp_c: float                  # range+smoke corrected; NOT corrected for rubble
    est_position: Tuple[float, float, float]
    smoke_transmittance: float
    truth_kind: str = ""               # scoring only: "survivor" | "electrical_fire" | "hot_pipe"
    truth_id: int = -1                 # scoring only

    def to_dict(self) -> Dict[str, Any]:
        return {
            "range_m": round(self.range_m, 2), "bearing_deg": round(math.degrees(self.bearing_rad), 1),
            "snr_db": round(self.snr_db, 1), "est_temp_c": round(self.est_temp_c, 1),
            "est_position": [round(v, 2) for v in self.est_position],
            "smoke_transmittance": round(self.smoke_transmittance, 3),
        }


@dataclass
class AcousticDetection:
    level_db: float
    snr_db: float
    bearing_rad: float
    range_est_m: float                 # coarse, level-based
    tapping: bool
    truth_id: int = -1                 # scoring only

    def to_dict(self) -> Dict[str, Any]:
        return {"level_db": round(self.level_db, 1), "snr_db": round(self.snr_db, 1),
                "bearing_deg": round(math.degrees(self.bearing_rad), 1),
                "range_est_m": round(self.range_est_m, 1), "tapping": self.tapping}


# --------------------------------------------------------------------------
# Sensor suite
# --------------------------------------------------------------------------
class SensorSuite:
    def __init__(self, world: DisasterWorld, rng: Optional[np.random.Generator] = None) -> None:
        self.world = world
        # separate stream so sensor noise never perturbs world reproducibility
        self.rng = rng if rng is not None else np.random.default_rng(world.seed + 101)

    # ---------------- LiDAR ----------------
    def lidar_scan(self, origin: Sequence[float], heading_rad: float = 0.0, pitch_deg: float = 0.0) -> LidarScan:
        s = cfg.SENSORS
        n, step = s.lidar_num_rays, s.lidar_step_m
        o = np.asarray(origin, dtype=float)
        ang = heading_rad + np.arange(n) * (2.0 * math.pi / n)
        pr = math.radians(pitch_deg)
        dirs = np.stack([math.cos(pr) * np.cos(ang), math.cos(pr) * np.sin(ang),
                         np.full(n, math.sin(pr))], axis=1)
        dist = np.arange(1, int(round(s.lidar_range_m / step)) + 1) * step
        m = dist.size
        pts = (o[None, None, :] + dirs[:, None, :] * dist[None, :, None]).reshape(-1, 3)

        solid = self.world.solid_at_points(pts).reshape(n, m)
        sigma = s.smoke_sigma_per_unit * self.world.smoke_at_points(pts).reshape(n, m)
        tau = np.cumsum(sigma, axis=1) * step                  # integral(sigma ds) along each ray
        t2 = np.exp(-2.0 * tau)                                # two-way Beer-Lambert transmittance

        hit_any = solid.any(axis=1)
        hit_idx = np.where(hit_any, solid.argmax(axis=1), m)
        lost = t2 < s.lidar_min_transmittance
        lost_idx = np.where(lost.any(axis=1), lost.argmax(axis=1), m)

        valid = hit_any & (hit_idx < lost_idx)
        stop = np.minimum(hit_idx, lost_idx)
        clear = np.where(stop > 0, dist[np.clip(stop - 1, 0, m - 1)], 0.0)

        true_range = dist[np.clip(hit_idx, 0, m - 1)] - 0.5 * step
        noise = self.rng.normal(0.0, s.lidar_range_noise_std_m * (1.0 + true_range / s.lidar_range_m))
        ranges = np.where(valid, np.maximum(true_range + noise, 0.0), np.nan)

        t_at = t2[np.arange(n), np.clip(np.minimum(hit_idx, m - 1), 0, m - 1)]
        smoke_limited = lost.any(axis=1) & (lost_idx <= hit_idx)
        return LidarScan(o, ang, pr, ranges, clear, valid, t_at, smoke_limited)

    # ---------------- segment sampling ----------------
    def _segment(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        d = float(np.linalg.norm(b - a))
        n = max(3, int(math.ceil(d / cfg.SENSORS.lidar_step_m)) + 1)
        t = np.linspace(0.0, 1.0, n)
        pts = a[None, :] + t[:, None] * (b - a)[None, :]
        return d, d / (n - 1), self.world.solid_at_points(pts), self.world.smoke_at_points(pts)

    # ---------------- Thermal IR ----------------
    def thermal_scan(
        self,
        origin: Sequence[float],
        heading_rad: float = 0.0,
        fov_deg: float = THERMAL_FOV_DEG,
        max_range_m: float = THERMAL_MAX_RANGE_M,
        min_snr_db: Optional[float] = None,
    ) -> List[ThermalDetection]:
        """Detect heat signatures (survivors AND decoys) inside the camera cone."""
        s, w = cfg.SENSORS, self.world
        thr = s.thermal_detect_snr_db if min_snr_db is None else min_snr_db
        o = np.asarray(origin, dtype=float)
        ambient = w.ambient_temp_c

        targets: List[Tuple[str, int, np.ndarray, float, float]] = []   # kind, id, point, temp, overburden
        for sv in w.survivors:
            if sv.rescued or not sv.alive:
                continue
            surf = max(sv.z, w.elevation_at(sv.x, sv.y)) + 0.3
            targets.append(("survivor", sv.sid, np.array([sv.x, sv.y, surf]), sv.signature_temp_c(),
                            w.survivor_overburden(sv)))
        for h in w.heat_sources:
            targets.append((h.kind, h.hid, np.array([h.x, h.y, h.z]), h.temp_c, 0.0))

        half_fov = math.radians(fov_deg) / 2.0
        out: List[ThermalDetection] = []
        for kind, tid, p, temp, over in targets:
            vec = p - o
            d, ds, solid, smoke = self._segment(o, p)
            if d < 0.2 or d > max_range_m:
                continue
            bearing = math.atan2(vec[1], vec[0])
            off = abs((bearing - heading_rad + math.pi) % (2.0 * math.pi) - math.pi)
            if fov_deg < 360.0 and off > half_fov:
                continue
            if solid[1:-1].any():                       # line of sight blocked by rubble/pillar
                continue
            tau = float(s.smoke_sigma_per_unit * smoke.sum() * ds)
            sig = thermal_signal(temp - ambient, d, tau, over)
            meas = max(0.0, sig + self.rng.normal(0.0, s.thermal_noise_floor))
            snr = snr_db_from_signal(meas)
            if snr < thr:
                continue
            t_smoke = beer_lambert_transmittance(tau)
            denom = s.thermal_ref_signal * (s.thermal_ref_distance_m / d) ** 2 * t_smoke
            est_temp = ambient + meas / max(denom, 1e-9)
            sigma_pos = 0.25 + 0.03 * d
            est = (float(p[0] + self.rng.normal(0, sigma_pos)), float(p[1] + self.rng.normal(0, sigma_pos)),
                   float(w.elevation_at(p[0], p[1])))
            out.append(ThermalDetection(d, bearing, snr, est_temp, est, t_smoke, kind, tid))
        out.sort(key=lambda r: -r.snr_db)
        return out

    # ---------------- Acoustic piezo array (rover only) ----------------
    def acoustic_scan(
        self,
        rover_pos: Sequence[float],
        heading_rad: float = 0.0,
        own_noise_db: float = 0.0,
        tapping_prob: float = ACOUSTIC_TAPPING_PROB,
    ) -> List[AcousticDetection]:
        """Listen through rubble; bearing from a TDOA plane-wave fit over the mic ring."""
        s, w = cfg.SENSORS, self.world
        o = np.asarray(rover_pos, dtype=float)
        floor_db = s.acoustic_noise_floor_db + own_noise_db
        phis = heading_rad + np.arange(s.acoustic_num_mics) * (2.0 * math.pi / s.acoustic_num_mics)
        rel = s.acoustic_array_radius_m * np.stack([np.cos(phis), np.sin(phis)], axis=1)     # (M, 2)
        mics = np.column_stack([o[0] + rel[:, 0], o[1] + rel[:, 1], np.full(len(phis), o[2])])

        out: List[AcousticDetection] = []
        for sv in w.survivors:
            if sv.rescued or not sv.alive:
                continue
            src = np.array([sv.x, sv.y, sv.z])
            d, ds, solid, _ = self._segment(src, o)
            if d > ACOUSTIC_MAX_RANGE_M:
                continue
            tapping = bool(self.rng.random() < tapping_prob)
            if not tapping:
                continue
            rubble_len = float(solid.sum() * ds)
            level = acoustic_received_db(sv.acoustic_emission_db(), d, rubble_len) \
                + float(self.rng.normal(0.0, s.acoustic_multipath_std_db))
            snr = level - floor_db
            if snr < s.acoustic_detect_snr_db:
                continue

            dists = np.linalg.norm(mics - src[None, :], axis=1)
            sigma_t = math.hypot(s.acoustic_time_jitter_s, MULTIPATH_DELAY_STD_PER_M * rubble_len)
            toa = dists / s.acoustic_speed_mps + self.rng.normal(0.0, sigma_t, size=dists.shape)
            # plane wave: t_i = t0 - (u/c) . m_i  ->  solve [1, -mx, -my] @ [t0, ux', uy'] = t
            A = np.column_stack([np.ones(len(toa)), -rel[:, 0], -rel[:, 1]])
            sol, *_ = np.linalg.lstsq(A, toa, rcond=None)
            bearing = math.atan2(sol[2], sol[1])

            r_est = float(np.clip(
                s.acoustic_ref_distance_m * 10.0 ** ((ACOUSTIC_ASSUMED_SOURCE_DB - level) / s.acoustic_spreading_exponent),
                1.0, ACOUSTIC_MAX_RANGE_M))
            out.append(AcousticDetection(level, snr, bearing, r_est, True, sv.sid))
        out.sort(key=lambda r: -r.snr_db)
        return out