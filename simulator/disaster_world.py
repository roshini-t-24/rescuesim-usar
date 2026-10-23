"""simulator/disaster_world.py

Collapsed-building world model for RescueSim-USAR.

Geometry: 60 m x 60 m x 15 m interior on a 1 m grid.

State held here (all NumPy):
    elevation_map[x, y]          rubble height in metres (0.0 .. 6.5)
    structural_instability[x, y] probability of secondary collapse (0 .. 1)
    smoke_density[x, y, z]       optical extinction field (unit-less, see sensors.py)
    ceiling_map[x, y]            underside height of the remaining ceiling slab
    pillars                      standing structural columns (2x2 cell footprint)
    survivors / heat_sources     trapped people and thermal decoys

"Cosmos hazard synthesis" note: this module reproduces the *kind* of environmental
conditioning a world foundation model such as NVIDIA Cosmos would provide
(randomised smoke plumes, haze, wind, lighting-free thermal clutter) with
procedural Gaussian/OU processes. It does NOT call Cosmos weights. The parameters
used for each world are exposed in `domain_randomization` so the UI can display them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, label

from config import settings as cfg

# --------------------------------------------------------------------------
# Module constants
# --------------------------------------------------------------------------
ENTRY_XY: Tuple[float, float] = (2.5, 2.5)   # rescue entry point (rover/drone start)
PILLAR_SPACING_M: int = 12
PILLAR_SIZE_CELLS: int = 2
PILLAR_COLLAPSE_PROB: float = 0.25           # pillars already gone at t = 0

SMOKE_PEAK: float = 2.6                      # peak plume density before hazard scaling
SMOKE_TAU_S: float = 90.0                    # ventilation / settling time constant
SMOKE_UPDATE_EVERY: int = 4                  # diffuse every N steps (speed)
SMOKE_MAX: float = 4.0

# The mission clock is compressed: 600 s of sim time stands for ~80 min of the
# "golden hour" of vital decline, so decay is visible within one demo run.
VITAL_DECAY_TIME_SCALE: float = 8.0


# --------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------
@dataclass
class Survivor:
    sid: int
    x: float
    y: float
    z: float                      # body altitude (fixed; rubble may pile on top)
    core_temp_c: float
    acoustic_db: float            # tapping / voice level at 1 m
    vital_decay_rate: float       # fraction of vitality lost per (clinical) minute
    vitality: float = 1.0
    located: bool = False
    rescued: bool = False

    @property
    def alive(self) -> bool:
        return self.vitality > 0.0

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def signature_temp_c(self) -> float:
        """Apparent thermal signature; cools slightly as vitality drops."""
        return self.core_temp_c - (1.0 - self.vitality) * 2.0

    def acoustic_emission_db(self) -> float:
        """Source level at 1 m; weakens as vitality drops (20*log10 amplitude law)."""
        return self.acoustic_db + 20.0 * math.log10(max(self.vitality, 0.05))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sid": self.sid, "x": round(self.x, 2), "y": round(self.y, 2), "z": round(self.z, 2),
            "core_temp_c": round(self.core_temp_c, 2), "acoustic_db": round(self.acoustic_db, 1),
            "vitality": round(self.vitality, 3), "located": self.located,
            "rescued": self.rescued, "alive": self.alive,
        }


@dataclass
class HeatSource:
    """Non-human heat source (decoy). Electrical fires also emit smoke."""
    hid: int
    kind: str                     # "electrical_fire" | "hot_pipe"
    x: float
    y: float
    z: float
    temp_c: float
    is_smoke_source: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"hid": self.hid, "kind": self.kind, "x": round(self.x, 2), "y": round(self.y, 2),
                "z": round(self.z, 2), "temp_c": round(self.temp_c, 1)}


# --------------------------------------------------------------------------
# World
# --------------------------------------------------------------------------
class DisasterWorld:
    def __init__(
        self,
        seed: int = 0,
        num_survivors: int = cfg.DEFAULT_NUM_SURVIVORS,
        hazard_modifiers: Optional[Dict[str, Any]] = None,
        dynamic_smoke: bool = True,
        auto_aftershocks: bool = True,
    ) -> None:
        self.seed = int(seed)
        self.nx, self.ny, self.nz = cfg.WORLD_X_M, cfg.WORLD_Y_M, cfg.WORLD_Z_M
        self.cell = cfg.CELL_SIZE_M
        self.entry_xy = ENTRY_XY
        self.dynamic_smoke = dynamic_smoke
        self.auto_aftershocks = auto_aftershocks

        self.time_s: float = 0.0
        self.step_count: int = 0
        self.terrain_version: int = 0
        self.events: List[Dict[str, Any]] = []

        self.hazards: Dict[str, Any] = dict(cfg.DEFAULT_HAZARD_MODIFIERS)
        self.ambient_temp_c: float = float(self.hazards["ambient_temp_c"])
        self.fema_margin_m: float = float(self.hazards["fema_safety_margin_m"])
        self._instability_factor: float = 1.0
        self._smoke_factor: float = self._smoke_factor_for(float(self.hazards["smoke_toxicity_index"]))

        self._xx, self._yy = np.meshgrid(
            (np.arange(self.nx) + 0.5) * self.cell,
            (np.arange(self.ny) + 0.5) * self.cell,
            indexing="ij",
        )

        self.survivors: List[Survivor] = []
        self.heat_sources: List[HeatSource] = []
        self.pillars: List[Tuple[int, int]] = []
        self._collapsed_pillars: List[Tuple[int, int]] = []

        placed = False
        for attempt in range(8):
            self.rng = np.random.default_rng(self.seed + 7919 * attempt)
            self._generate_terrain()
            self._generate_ceiling()
            self._generate_pillars()
            self._refresh_derived()
            self._generate_instability()
            self._place_survivors(num_survivors)
            if self.survivors:
                placed = True
                break
        if not placed:  # extremely unlikely; flatten the terrain so a survivor fits
            self.elevation_map = np.minimum(self.elevation_map, 1.0)
            self._refresh_derived()
            self._generate_instability()
            self._place_survivors(num_survivors)

        self._generate_heat_sources()
        self._generate_smoke()
        self._init_wind()

        self.domain_randomization: Dict[str, Any] = {
            "seed": self.seed,
            "terrain_roughness_iterations": self._terrain_iters,
            "pillars_standing": len(self.pillars),
            "pillars_collapsed_at_start": len(self._collapsed_pillars),
            "smoke_sources": sum(1 for h in self.heat_sources if h.is_smoke_source),
            "smoke_plume_sigma_m": [5.0, 5.0, 2.5],
            "haze_amplitude": 0.25,
            "ceiling_sag_max_m": 5.0,
            "wind_heading_deg": round(math.degrees(math.atan2(self._wind_dir[1], self._wind_dir[0])) % 360, 1),
            "survivor_overburden_m": [round(self.survivor_overburden(s), 2) for s in self.survivors],
            "method": "procedural (Gaussian/OU) Cosmos-style conditioning; no Cosmos weights invoked",
        }

        if hazard_modifiers:
            self.apply_hazard_modifiers(hazard_modifiers)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate_terrain(self) -> None:
        rng, nx, ny = self.rng, self.nx, self.ny
        field = np.zeros((nx, ny))
        for sigma, amp in ((9.0, 1.0), (4.5, 0.55), (2.2, 0.25)):
            n = gaussian_filter(rng.standard_normal((nx, ny)), sigma, mode="reflect")
            field += amp * n / (n.std() + 1e-9)
        for _ in range(int(rng.integers(5, 9))):          # collapsed slab piles
            cx, cy = rng.uniform(8, nx - 8), rng.uniform(8, ny - 8)
            r, h = rng.uniform(4.0, 9.0), rng.uniform(1.0, 2.4)
            field += h * np.exp(-((self._xx - cx) ** 2 + (self._yy - cy) ** 2) / (2.0 * (r / 2.0) ** 2))
        field = (field - field.min()) / (np.ptp(field) + 1e-9)
        elev = cfg.MAX_RUBBLE_HEIGHT_M * field ** 1.4

        d_entry = np.hypot(self._xx - self.entry_xy[0], self._yy - self.entry_xy[1])
        elev *= np.clip((d_entry - 3.0) / 9.0, 0.0, 1.0) ** 2   # keep the entrance passable

        iters = 0
        for iters in range(1, 13):   # calibrate so ~6-22 % of cells exceed the rover slope limit
            gx, gy = np.gradient(elev, self.cell)
            frac = float(np.mean(np.hypot(gx, gy) > cfg.ROVER.max_slope))
            if frac > 0.22:
                elev = gaussian_filter(elev, 0.8, mode="nearest")
            elif frac < 0.06:
                elev = np.clip(elev * 1.12, 0.0, cfg.MAX_RUBBLE_HEIGHT_M)
            else:
                break
        self.elevation_map = np.clip(elev, 0.0, cfg.MAX_RUBBLE_HEIGHT_M)
        self._terrain_iters = iters

    def _generate_ceiling(self) -> None:
        sag = gaussian_filter(self.rng.standard_normal((self.nx, self.ny)), 8.0, mode="reflect")
        sag = (sag - sag.min()) / (np.ptp(sag) + 1e-9)
        ceiling = cfg.CEILING_HEIGHT_M - 5.0 * sag ** 1.5
        self.ceiling_map = np.maximum(ceiling, self.elevation_map + 3.0)

    def _generate_pillars(self) -> None:
        self.pillars, self._collapsed_pillars = [], []
        for px in range(6, self.nx - 4, PILLAR_SPACING_M):
            for py in range(6, self.ny - 4, PILLAR_SPACING_M):
                if self.rng.random() < PILLAR_COLLAPSE_PROB:
                    self._collapsed_pillars.append((px, py))
                else:
                    self.pillars.append((px, py))

    def _build_pillar_mask(self) -> np.ndarray:
        mask = np.zeros((self.nx, self.ny), dtype=bool)
        for px, py in self.pillars:
            mask[px:px + PILLAR_SIZE_CELLS, py:py + PILLAR_SIZE_CELLS] = True
        return mask

    def _refresh_derived(self) -> None:
        """Recompute slope, voxel occupancy and rover traversability from the height fields."""
        self._gx, self._gy = np.gradient(self.elevation_map, self.cell)
        self.slope_map = np.hypot(self._gx, self._gy)
        self.pillar_mask = self._build_pillar_mask()
        zc = (np.arange(self.nz) + 0.5) * self.cell
        self.solid = (
            (zc[None, None, :] < self.elevation_map[:, :, None])
            | (zc[None, None, :] >= self.ceiling_map[:, :, None])
            | self.pillar_mask[:, :, None]
        )
        self._traversable = (self.slope_map <= cfg.ROVER.max_slope) & ~self.pillar_mask

    def _generate_instability(self) -> None:
        rng = self.rng
        rough = gaussian_filter(rng.standard_normal((self.nx, self.ny)), 3.0, mode="reflect")
        rough = rough / (rough.std() + 1e-9)
        base = (
            0.55 * np.clip(self.slope_map, 0.0, 1.0)
            + 0.25 * (self.elevation_map / cfg.MAX_RUBBLE_HEIGHT_M)
            + 0.20 / (1.0 + np.exp(-rough))
        )
        for px, py in self._collapsed_pillars:   # missing column => loaded, cracked slab around it
            d2 = (self._xx - (px + 1.0)) ** 2 + (self._yy - (py + 1.0)) ** 2
            base += 0.30 * np.exp(-d2 / (2.0 * 2.5 ** 2))
        self._instability_base = np.clip(gaussian_filter(base, 1.0, mode="nearest"), 0.0, 1.0)
        self._apply_instability_factor()

    def _apply_instability_factor(self) -> None:
        self.structural_instability = np.clip(self._instability_base * self._instability_factor, 0.0, 1.0)

    def _place_survivors(self, n: int) -> None:
        self.survivors = []
        lab, _ = label(self._traversable, structure=np.ones((3, 3)))
        ei, ej = self._ij(*self.entry_xy)
        if lab[ei, ej] == 0:
            return
        reach = lab == lab[ei, ej]
        d_entry = np.hypot(self._xx - self.entry_xy[0], self._yy - self.entry_xy[1])
        cand = (
            reach
            & (self.elevation_map > 0.5) & (self.elevation_map < 5.2)
            & (self.slope_map < 0.40) & (self.structural_instability < 0.75)
            & (d_entry >= 10.0)
        )
        cells = np.argwhere(cand)
        if len(cells) == 0:
            return
        self.rng.shuffle(cells)
        for min_sep in (10.0, 6.0, 3.0):
            for i, j in cells:
                if len(self.survivors) >= n:
                    break
                x, y = (i + 0.5) * self.cell, (j + 0.5) * self.cell
                if all(math.hypot(x - s.x, y - s.y) >= min_sep for s in self.survivors):
                    cover = float(self.rng.uniform(0.2, 1.8))
                    self.survivors.append(Survivor(
                        sid=len(self.survivors), x=x, y=y,
                        z=max(0.3, float(self.elevation_map[i, j]) - cover),
                        core_temp_c=float(self.rng.uniform(*cfg.SURVIVOR_CORE_TEMP_RANGE_C)),
                        acoustic_db=float(self.rng.uniform(*cfg.SURVIVOR_ACOUSTIC_DB_RANGE)),
                        vital_decay_rate=float(self.rng.uniform(*cfg.SURVIVOR_VITAL_DECAY_PER_MIN)),
                    ))
            if len(self.survivors) >= n:
                break

    def _generate_heat_sources(self) -> None:
        d_entry = np.hypot(self._xx - self.entry_xy[0], self._yy - self.entry_xy[1])
        ok = (self.elevation_map < 4.0) & (~self.pillar_mask) & (d_entry >= 8.0)
        cells = np.argwhere(ok)
        self.rng.shuffle(cells)
        specs = [("electrical_fire", float(self.rng.uniform(80, 140)), True),
                 ("electrical_fire", float(self.rng.uniform(80, 140)), True),
                 ("hot_pipe", float(self.rng.uniform(36.5, 38.5)), False)]   # human-like decoy
        self.heat_sources = []
        for hid, (kind, temp, smoke) in enumerate(specs):
            if hid >= len(cells):
                break
            i, j = cells[hid]
            self.heat_sources.append(HeatSource(
                hid=hid, kind=kind, x=(i + 0.5) * self.cell, y=(j + 0.5) * self.cell,
                z=float(self.elevation_map[i, j]) + 0.5, temp_c=temp, is_smoke_source=smoke))

    @staticmethod
    def _smoke_factor_for(toxicity: float) -> float:
        return 0.7 + 0.8 * toxicity

    def _generate_smoke(self) -> None:
        src = np.zeros((self.nx, self.ny, self.nz))
        for h in self.heat_sources:
            if h.is_smoke_source:
                i, j = self._ij(h.x, h.y)
                src[i, j, int(np.clip(h.z / self.cell, 0, self.nz - 1))] = 1.0
        plume = gaussian_filter(src, sigma=(5.0, 5.0, 2.5), mode="constant")
        plume = plume / max(float(plume.max()), 1e-12)
        zfrac = (np.arange(self.nz) + 0.5) / self.nz
        buoyancy = 0.5 + 0.9 * zfrac                      # smoke pools toward the ceiling
        haze = gaussian_filter(self.rng.standard_normal((self.nx, self.ny, self.nz)),
                               sigma=(6.0, 6.0, 3.0), mode="reflect")
        haze = (haze - haze.min()) / (np.ptp(haze) + 1e-9)
        self._smoke_base = SMOKE_PEAK * plume * buoyancy[None, None, :] + 0.25 * haze * (0.3 + 0.7 * zfrac)[None, None, :]
        self._smoke_target = self._smoke_base * self._smoke_factor
        self.smoke_density = np.where(self.solid, 0.0, self._smoke_target)

    def _init_wind(self) -> None:
        ang = float(self.rng.uniform(0.0, 2.0 * math.pi))
        self._wind_dir = np.array([math.cos(ang), math.sin(ang), 0.0])
        self._wind_state = self._wind_mean().copy()

    def _wind_mean(self) -> np.ndarray:
        return self._wind_dir * 0.35 * float(self.hazards["wind_turbulence_mps"])

    # ------------------------------------------------------------------
    # Coordinate helpers & field queries
    # ------------------------------------------------------------------
    def _ij(self, x: float, y: float) -> Tuple[int, int]:
        return (int(np.clip(math.floor(x / self.cell), 0, self.nx - 1)),
                int(np.clip(math.floor(y / self.cell), 0, self.ny - 1)))

    def in_bounds(self, x: float, y: float, z: Optional[float] = None) -> bool:
        ok = 0.0 <= x < self.nx * self.cell and 0.0 <= y < self.ny * self.cell
        if z is not None:
            ok = ok and 0.0 <= z < self.nz * self.cell
        return ok

    def _bilinear(self, grid: np.ndarray, x: float, y: float) -> float:
        fx = float(np.clip(x / self.cell - 0.5, 0.0, self.nx - 1.000001))
        fy = float(np.clip(y / self.cell - 0.5, 0.0, self.ny - 1.000001))
        i0, j0 = int(fx), int(fy)
        tx, ty = fx - i0, fy - j0
        return float(grid[i0, j0] * (1 - tx) * (1 - ty) + grid[i0 + 1, j0] * tx * (1 - ty)
                     + grid[i0, j0 + 1] * (1 - tx) * ty + grid[i0 + 1, j0 + 1] * tx * ty)

    def elevation_at(self, x: float, y: float) -> float:
        return self._bilinear(self.elevation_map, x, y)

    def slope_at(self, x: float, y: float) -> float:
        return self._bilinear(self.slope_map, x, y)

    def gradient_at(self, x: float, y: float) -> Tuple[float, float]:
        return self._bilinear(self._gx, x, y), self._bilinear(self._gy, x, y)

    def instability_at(self, x: float, y: float) -> float:
        return self._bilinear(self.structural_instability, x, y)

    def ceiling_at(self, x: float, y: float) -> float:
        return self._bilinear(self.ceiling_map, x, y)

    def is_solid(self, x: float, y: float, z: float) -> bool:
        if not self.in_bounds(x, y, z):
            return True
        i, j = self._ij(x, y)
        return bool(self.solid[i, j, int(np.clip(math.floor(z / self.cell), 0, self.nz - 1))])

    def smoke_at(self, x: float, y: float, z: float) -> float:
        if not self.in_bounds(x, y, z):
            return 0.0
        i, j = self._ij(x, y)
        return float(self.smoke_density[i, j, int(np.clip(math.floor(z / self.cell), 0, self.nz - 1))])

    def _point_indices(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        idx = np.floor(pts / self.cell).astype(int)
        inb = ((idx[:, 0] >= 0) & (idx[:, 0] < self.nx) & (idx[:, 1] >= 0) & (idx[:, 1] < self.ny)
               & (idx[:, 2] >= 0) & (idx[:, 2] < self.nz))
        ii = np.clip(idx[:, 0], 0, self.nx - 1)
        jj = np.clip(idx[:, 1], 0, self.ny - 1)
        kk = np.clip(idx[:, 2], 0, self.nz - 1)
        return ii, jj, kk, inb

    def solid_at_points(self, pts: np.ndarray) -> np.ndarray:
        """Vectorised occupancy for ray marching. Out-of-bounds counts as solid."""
        ii, jj, kk, inb = self._point_indices(pts)
        return np.where(inb, self.solid[ii, jj, kk], True)

    def smoke_at_points(self, pts: np.ndarray) -> np.ndarray:
        """Vectorised smoke density for Beer-Lambert integration. Out-of-bounds is 0."""
        ii, jj, kk, inb = self._point_indices(pts)
        return np.where(inb, self.smoke_density[ii, jj, kk], 0.0)

    def wind_at(self, x: float, y: float, z: float) -> np.ndarray:
        """Wind vector (m/s) at a position: Ornstein-Uhlenbeck turbulence, stronger at altitude."""
        alt = 0.4 + 0.6 * float(np.clip(z / 12.0, 0.0, 1.2))
        return self._wind_state * alt

    def traversable_mask(self) -> np.ndarray:
        """True where the rover can stand (slope within limit, not a pillar). Do not mutate."""
        return self._traversable

    def survivor_overburden(self, s: Survivor) -> float:
        """Rubble thickness above the survivor (m); grows when aftershocks pile more debris."""
        return max(0.0, self.elevation_at(s.x, s.y) - s.z)

    # ------------------------------------------------------------------
    # Survivor bookkeeping
    # ------------------------------------------------------------------
    def survivor_by_id(self, sid: int) -> Optional[Survivor]:
        return next((s for s in self.survivors if s.sid == sid), None)

    def mark_located(self, sid: int) -> bool:
        s = self.survivor_by_id(sid)
        if s is None or s.located:
            return False
        s.located = True
        self._log({"type": "survivor_located", "sid": sid})
        return True

    def rescue_survivor(self, sid: int) -> bool:
        s = self.survivor_by_id(sid)
        if s is None or s.rescued or not s.alive:
            return False
        s.rescued = True
        self._log({"type": "survivor_rescued", "sid": sid, "vitality": round(s.vitality, 3)})
        return True

    # ------------------------------------------------------------------
    # Hazard coupling (Tavily -> physics)
    # ------------------------------------------------------------------
    def physics_snapshot(self) -> Dict[str, float]:
        free = ~self.solid
        sm = self.smoke_density[free]
        return {
            "seismic_aftershock_prob": round(float(self.hazards["seismic_aftershock_prob"]), 3),
            "wind_turbulence_mps": round(float(self.hazards["wind_turbulence_mps"]), 2),
            "ambient_temp_c": round(self.ambient_temp_c, 1),
            "fema_safety_margin_m": round(self.fema_margin_m, 2),
            "mean_structural_instability": round(float(self.structural_instability.mean()), 3),
            "high_risk_fraction": round(float((self.structural_instability > 0.6).mean()), 3),
            "mean_smoke_density": round(float(sm.mean()) if sm.size else 0.0, 3),
            "peak_smoke_density": round(float(sm.max()) if sm.size else 0.0, 3),
        }

    def apply_hazard_modifiers(self, mods: Dict[str, Any]) -> Dict[str, Any]:
        """Clamp and apply PhysicalHazardModifiers. Returns before/after physics for the UI."""
        before = self.physics_snapshot()
        clean: Dict[str, Any] = dict(cfg.DEFAULT_HAZARD_MODIFIERS)
        for key, (lo, hi) in cfg.HAZARD_BOUNDS.items():
            raw = mods.get(key)
            if raw is None:
                continue
            try:
                v = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                clean[key] = float(np.clip(v, lo, hi))
        urls = mods.get("source_urls") or []
        clean["source_urls"] = [str(u) for u in urls if isinstance(u, str)]

        old_smoke_factor = self._smoke_factor
        self.hazards = clean
        self.ambient_temp_c = float(clean["ambient_temp_c"])
        self.fema_margin_m = float(clean["fema_safety_margin_m"])

        self._instability_factor = float(np.clip(0.5 + 5.0 * clean["seismic_aftershock_prob"], 0.4, 1.6))
        self._apply_instability_factor()

        self._smoke_factor = self._smoke_factor_for(float(clean["smoke_toxicity_index"]))
        ratio = self._smoke_factor / old_smoke_factor
        self._smoke_target = self._smoke_base * self._smoke_factor
        self.smoke_density = np.clip(self.smoke_density * ratio, 0.0, SMOKE_MAX)

        self._wind_state = self._wind_mean().copy()
        self._log({"type": "hazard_update", "modifiers": {k: v for k, v in clean.items() if k != "source_urls"}})
        return {"before": before, "after": self.physics_snapshot(), "modifiers": clean}

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def _log(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        ev.setdefault("time_s", round(self.time_s, 2))
        self.events.append(ev)
        return ev

    def _step_wind(self, dt: float) -> None:
        theta = 0.8
        sigma = 0.5 * float(self.hazards["wind_turbulence_mps"])
        noise = self.rng.standard_normal(3) * np.array([1.0, 1.0, 0.5])
        self._wind_state = (self._wind_state - theta * (self._wind_state - self._wind_mean()) * dt
                            + sigma * math.sqrt(dt) * noise)

    def _update_smoke(self, dt_eff: float) -> None:
        blurred = gaussian_filter(self.smoke_density, sigma=(0.8, 0.8, 0.5), mode="nearest")
        s = 0.85 * self.smoke_density + 0.15 * blurred      # gentle diffusion; keeps plume cores
        a = 1.0 - math.exp(-dt_eff / SMOKE_TAU_S)           # relax toward plume target (dust decays away)
        s = s + a * (self._smoke_target - s)
        s[self.solid] = 0.0
        self.smoke_density = np.clip(s, 0.0, SMOKE_MAX)

    def step(self, dt: float = cfg.DT_S) -> List[Dict[str, Any]]:
        """Advance the world by dt seconds. Returns events that happened this step."""
        events: List[Dict[str, Any]] = []
        self.time_s += dt
        self.step_count += 1
        self._step_wind(dt)

        for s in self.survivors:
            if s.rescued or not s.alive:
                continue
            s.vitality = max(0.0, s.vitality - s.vital_decay_rate * VITAL_DECAY_TIME_SCALE * dt / 60.0)
            if s.vitality <= 0.0:
                events.append(self._log({"type": "survivor_lost", "sid": s.sid}))

        if self.dynamic_smoke and self.step_count % SMOKE_UPDATE_EVERY == 0:
            self._update_smoke(dt * SMOKE_UPDATE_EVERY)

        if self.auto_aftershocks:
            p_mission = float(self.hazards["seismic_aftershock_prob"])
            mission_s = cfg.MAX_MISSION_STEPS * cfg.DT_S
            p_step = 1.0 - (1.0 - p_mission) ** (dt / mission_s)
            if self.rng.random() < p_step:
                events.append(self.trigger_aftershock(float(self.rng.triangular(4.0, 4.8, 6.2))))
        return events

    def trigger_aftershock(self, magnitude: float, epicenter: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Collapse rubble around an epicentre, closing corridors and forcing replanning.

        Effects: elevation rises (debris), ceiling drops, some pillars fail, nearby cells
        become less stable, dust raises smoke density, buried survivors take vitality damage.
        """
        mag = float(np.clip(magnitude, 2.0, 8.0))
        scale = float(np.clip((mag - 3.0) / 3.5, 0.1, 1.0))
        before_trav = self._traversable.copy()

        if epicenter is None:
            p = (self.structural_instability ** 2) * (~self.pillar_mask)
            p = p.ravel() / max(float(p.sum()), 1e-12)
            flat = int(self.rng.choice(p.size, p=p))
            ei, ej = divmod(flat, self.ny)
            ex, ey = (ei + 0.5) * self.cell, (ej + 0.5) * self.cell
        else:
            ex, ey = float(epicenter[0]), float(epicenter[1])

        radius = 3.0 + 7.0 * scale
        d = np.hypot(self._xx - ex, self._yy - ey)
        w = np.exp(-(d ** 2) / (2.0 * (radius / 2.0) ** 2))
        w[d > radius] = 0.0

        noise = gaussian_filter(self.rng.standard_normal((self.nx, self.ny)), 2.0, mode="reflect")
        noise = noise / (noise.std() + 1e-9)
        delta = np.clip(w * scale * (2.0 + 1.2 * noise), 0.0, None)

        pillars_lost = []
        for px, py in list(self.pillars):
            if math.hypot(px + 1.0 - ex, py + 1.0 - ey) <= radius and scale > 0.5 \
                    and self.rng.random() < 0.3 * scale:
                self.pillars.remove((px, py))
                self._collapsed_pillars.append((px, py))
                pillars_lost.append((px, py))
                delta[px:px + PILLAR_SIZE_CELLS, py:py + PILLAR_SIZE_CELLS] += 2.0 * scale

        self.elevation_map = np.clip(self.elevation_map + delta, 0.0, cfg.MAX_RUBBLE_HEIGHT_M)
        self.ceiling_map = np.maximum(self.ceiling_map - 0.8 * delta, self.elevation_map + 2.0)

        ring = np.exp(-((d - radius) ** 2) / (2.0 * (radius / 3.0) ** 2)) * (d > 0.6 * radius)
        self._instability_base = np.clip(self._instability_base + 0.3 * scale * w + 0.2 * scale * ring, 0.0, 1.0)
        self._apply_instability_factor()

        self._refresh_derived()
        self.smoke_density = np.clip(self.smoke_density + 0.8 * scale * w[:, :, None]
                                     * np.exp(-np.arange(self.nz) / 4.0)[None, None, :], 0.0, SMOKE_MAX)
        self.smoke_density[self.solid] = 0.0

        affected = []
        for s in self.survivors:
            if s.rescued or not s.alive:
                continue
            wi = float(w[self._ij(s.x, s.y)])
            if wi > 0.05:
                s.vitality = max(0.0, s.vitality - 0.15 * scale * wi)
                affected.append(s.sid)

        self.terrain_version += 1
        newly_blocked = int((before_trav & ~self._traversable).sum())
        return self._log({
            "type": "aftershock", "magnitude": round(mag, 2),
            "epicenter_xy": [round(ex, 1), round(ey, 1)], "radius_m": round(radius, 1),
            "newly_blocked_cells": newly_blocked,
            "newly_blocked_fraction": round(newly_blocked / float(before_trav.sum() or 1), 4),
            "max_elevation_delta_m": round(float(delta.max()), 2),
            "pillars_collapsed": [list(p) for p in pillars_lost],
            "survivors_affected": affected, "terrain_version": self.terrain_version,
        })

    # ------------------------------------------------------------------
    # Summaries for the orchestrator / UI
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        free = ~self.solid
        sm = self.smoke_density[free]
        return {
            "time_s": round(self.time_s, 1),
            "terrain_version": self.terrain_version,
            "traversable_fraction": round(float(self._traversable.mean()), 3),
            "mean_slope": round(float(self.slope_map.mean()), 3),
            "max_slope": round(float(self.slope_map.max()), 3),
            "mean_instability": round(float(self.structural_instability.mean()), 3),
            "high_risk_fraction": round(float((self.structural_instability > 0.6).mean()), 3),
            "mean_smoke": round(float(sm.mean()) if sm.size else 0.0, 3),
            "peak_smoke": round(float(sm.max()) if sm.size else 0.0, 3),
            "survivors_total": len(self.survivors),
            "survivors_located": sum(s.located for s in self.survivors),
            "survivors_rescued": sum(s.rescued for s in self.survivors),
            "survivors_alive": sum(s.alive and not s.rescued for s in self.survivors),
            "aftershocks": sum(1 for e in self.events if e["type"] == "aftershock"),
            "fire_sources": sum(1 for h in self.heat_sources if h.is_smoke_source),
            "hazards": {k: v for k, v in self.hazards.items() if k != "source_urls"},
        }

    def snapshot(self, stride: int = 2) -> Dict[str, Any]:
        """Down-sampled terrain/entity payload for the browser 3D viewport."""
        st = max(1, int(stride))
        return {
            "stride": st, "size_m": [self.nx, self.ny, self.nz],
            "terrain_version": self.terrain_version, "time_s": round(self.time_s, 1),
            "elevation": np.round(self.elevation_map[::st, ::st], 2).tolist(),
            "instability": np.round(self.structural_instability[::st, ::st], 2).tolist(),
            "ceiling": np.round(self.ceiling_map[::st, ::st], 1).tolist(),
            "pillars": [list(p) for p in self.pillars],
            "survivors": [s.to_dict() for s in self.survivors],
            "heat_sources": [h.to_dict() for h in self.heat_sources],
        }

    def smoke_points(self, threshold: float = 0.4, stride: int = 2, max_points: int = 1500) -> List[List[float]]:
        """Sparse [x, y, z, density] samples of dense smoke for a translucent 3D scatter."""
        st = max(1, int(stride))
        sub = self.smoke_density[::st, ::st, ::st]
        idx = np.argwhere(sub > threshold)
        if len(idx) == 0:
            return []
        dens = sub[idx[:, 0], idx[:, 1], idx[:, 2]]
        if len(idx) > max_points:
            keep = np.argsort(dens)[-max_points:]
            idx, dens = idx[keep], dens[keep]
        pos = (idx * st + 0.5 * st) * self.cell
        return [[round(float(p[0]), 1), round(float(p[1]), 1), round(float(p[2]), 1), round(float(dv), 2)]
                for p, dv in zip(pos, dens)]