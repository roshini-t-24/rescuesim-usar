"""Central configuration for RescueSim-USAR.

All physics constants, model IDs, endpoints and default policy weights live
here so the simulator, agents, batch evaluator and API server share one
source of truth. Nothing in this module performs network I/O.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# --------------------------------------------------------------------------
# Nebius Token Factory (OpenAI-compatible) + NVIDIA model IDs
# --------------------------------------------------------------------------
NEBIUS_BASE_URL: str = os.environ.get("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1/")

# Verify exact IDs against the Token Factory model catalog; override via env
# without touching code if your catalog lists a different NVIDIA variant.
PLANNER_MODEL_ID: str = os.environ.get(
    "NEBIUS_PLANNER_MODEL", "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"
)
REFLEX_MODEL_ID: str = os.environ.get(
    "NEBIUS_REFLEX_MODEL", "nvidia/Nemotron-Mini-4B-Instruct"
)

LLM_TIMEOUT_S: float = 45.0
LLM_MAX_TOKENS_PLANNER: int = 2500
LLM_MAX_TOKENS_REFLEX: int = 1200


def get_nebius_key() -> str | None:
    """Read at call time so tests can monkeypatch the environment."""
    key = os.environ.get("NEBIUS_API_KEY", "").strip()
    return key or None


def get_tavily_key() -> str | None:
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    return key or None


def nebius_online() -> bool:
    return get_nebius_key() is not None


def tavily_online() -> bool:
    return get_tavily_key() is not None


def mode_banner() -> Dict[str, str]:
    """Human-readable mode flags the UI shows so judges always know what ran."""
    return {
        "nebius": "LIVE (Nebius Token Factory)" if nebius_online() else "OFFLINE (deterministic heuristic)",
        "tavily": "LIVE (Tavily search)" if tavily_online() else "OFFLINE (cached hazard profile)",
    }


# --------------------------------------------------------------------------
# World geometry  (60 m x 60 m x 15 m building interior, 1 m voxels)
# --------------------------------------------------------------------------
WORLD_X_M: int = 60
WORLD_Y_M: int = 60
WORLD_Z_M: int = 15
CELL_SIZE_M: float = 1.0

MAX_RUBBLE_HEIGHT_M: float = 6.5
CEILING_HEIGHT_M: float = 15.0
SMOKE_Z_LAYERS: int = WORLD_Z_M  # one smoke layer per metre of height

# Survivor generation ranges (from blueprint)
SURVIVOR_CORE_TEMP_RANGE_C: Tuple[float, float] = (35.0, 37.5)
SURVIVOR_ACOUSTIC_DB_RANGE: Tuple[float, float] = (45.0, 80.0)
SURVIVOR_VITAL_DECAY_PER_MIN: Tuple[float, float] = (0.002, 0.012)  # fraction of vitality lost / minute
DEFAULT_NUM_SURVIVORS: int = 5

# --------------------------------------------------------------------------
# Simulation clock
# --------------------------------------------------------------------------
DT_S: float = 0.5               # simulation timestep
MAX_MISSION_STEPS: int = 1200   # 600 s of mission time
REPLAN_COOLDOWN_STEPS: int = 20


# --------------------------------------------------------------------------
# Quadruped rover (ANYmal / Spot class)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RoverConfig:
    max_slope: float = 0.55            # ||grad(elevation)|| limit (~28 deg)
    max_speed_mps: float = 0.9
    max_yaw_rate_rps: float = 1.2
    max_accel_mps2: float = 0.8
    max_yaw_accel_rps2: float = 2.5
    radius_m: float = 0.45
    battery_capacity_wh: float = 600.0
    k1_wh_per_m: float = 0.55          # dE = k1*|v|dt + k2*max(0, dz)
    k2_wh_per_m_climb: float = 4.0
    idle_power_w: float = 35.0
    dwa_horizon_s: float = 2.0
    dwa_v_samples: int = 7
    dwa_w_samples: int = 11
    payload_pods: int = 3              # medical supply drop pods


# --------------------------------------------------------------------------
# Micro drone (sub-250 g indoor UAV)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DroneConfig:
    z_min_m: float = 1.5
    z_max_m: float = 12.0
    max_speed_mps: float = 2.0
    max_accel_mps2: float = 3.0
    radius_m: float = 0.15
    battery_capacity_wh: float = 6.0
    hover_power_w: float = 14.0
    k_speed_w_per_mps: float = 3.0
    drag_coeff: float = 0.35           # velocity damping (1/s)
    wind_gain: float = 0.12            # turbulence force per m/s of wind
    # Artificial potential field gains
    k_attract_frontier: float = 1.0
    k_attract_thermal: float = 1.8
    k_repel_obstacle: float = 4.0
    repel_influence_m: float = 3.0
    k_ceiling_repel: float = 2.0


# --------------------------------------------------------------------------
# Sensors
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SensorConfig:
    # LiDAR
    lidar_num_rays: int = 36           # 10 deg resolution
    lidar_range_m: float = 15.0
    lidar_step_m: float = 0.25
    lidar_range_noise_std_m: float = 0.03
    lidar_min_transmittance: float = 0.05   # return is dropped below this
    smoke_sigma_per_unit: float = 0.12      # extinction coeff (1/m) per unit smoke density
    # Thermal IR
    thermal_ref_distance_m: float = 1.0
    thermal_ref_signal: float = 40.0        # signal units at 1 m for dT = 1 C
    thermal_noise_floor: float = 0.35
    thermal_detect_snr_db: float = 9.0
    ambient_temp_default_c: float = 18.0
    # Acoustic piezo array (rover only)
    acoustic_array_radius_m: float = 0.30
    acoustic_num_mics: int = 6
    acoustic_noise_floor_db: float = 32.0
    acoustic_ref_distance_m: float = 1.0
    acoustic_rubble_loss_db_per_m: float = 3.5   # per metre of rubble along path
    acoustic_spreading_exponent: float = 20.0    # 20*log10(r/r0) spherical spreading
    acoustic_multipath_std_db: float = 2.5
    acoustic_detect_snr_db: float = 6.0
    acoustic_speed_mps: float = 340.0
    acoustic_time_jitter_s: float = 2.0e-5


ROVER = RoverConfig()
DRONE = DroneConfig()
SENSORS = SensorConfig()


# --------------------------------------------------------------------------
# Risk-aware A*  cost:  C = d + alpha*slope + beta*instability  (+ gamma*smoke)
# --------------------------------------------------------------------------
@dataclass
class PolicyWeights:
    alpha_slope: float = 4.0
    beta_collapse: float = 6.0
    gamma_smoke: float = 1.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "alpha_slope": self.alpha_slope,
            "beta_collapse": self.beta_collapse,
            "gamma_smoke": self.gamma_smoke,
        }


# Policy presets used by the live sim and the 100-episode batch evaluator
POLICY_BASELINE_GREEDY = PolicyWeights(alpha_slope=0.0, beta_collapse=0.0, gamma_smoke=0.0)
POLICY_STATIC_CONSERVATIVE = PolicyWeights(alpha_slope=4.0, beta_collapse=6.0, gamma_smoke=1.0)

# Hard bounds the orchestrator output is clamped to (guards against LLM garbage)
WEIGHT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "alpha_slope": (0.0, 12.0),
    "beta_collapse": (0.0, 20.0),
    "gamma_smoke": (0.0, 6.0),
}


# --------------------------------------------------------------------------
# Tavily + hazard modifiers (defaults == "no live data" baseline)
# --------------------------------------------------------------------------
TAVILY_DEFAULT_QUERY: str = (
    "latest earthquake aftershock structural hazard USAR weather telemetry"
)
TAVILY_MAX_RESULTS: int = 5
TAVILY_SEARCH_DEPTH: str = "advanced"

DEFAULT_HAZARD_MODIFIERS: Dict[str, object] = {
    "seismic_aftershock_prob": 0.10,
    "wind_turbulence_mps": 1.0,
    "ambient_temp_c": 18.0,
    "smoke_toxicity_index": 0.30,
    "fema_safety_margin_m": 2.0,
    "source_urls": [],
}

# Physical-validity clamps applied to anything parsed from an LLM / web text
HAZARD_BOUNDS: Dict[str, Tuple[float, float]] = {
    "seismic_aftershock_prob": (0.0, 1.0),
    "wind_turbulence_mps": (0.0, 15.0),
    "ambient_temp_c": (-30.0, 55.0),
    "smoke_toxicity_index": (0.0, 1.0),
    "fema_safety_margin_m": (0.5, 10.0),
}

# Offline cached profile used when TAVILY_API_KEY is absent (a realistic
# mid-magnitude urban collapse scenario, so the offline demo still couples
# hazard data into physics).
OFFLINE_HAZARD_PROFILE: Dict[str, object] = {
    "seismic_aftershock_prob": 0.18,
    "wind_turbulence_mps": 4.2,
    "ambient_temp_c": 14.5,
    "smoke_toxicity_index": 0.65,
    "fema_safety_margin_m": 2.5,
    "source_urls": ["offline://cached-usar-hazard-profile"],
}

# --------------------------------------------------------------------------
# Server / batch
# --------------------------------------------------------------------------
SERVER_HOST: str = os.environ.get("RESCUESIM_HOST", "0.0.0.0")
SERVER_PORT: int = int(os.environ.get("RESCUESIM_PORT", "8000"))
BATCH_DEFAULT_EPISODES: int = 100
BATCH_DEFAULT_SEED: int = 2026