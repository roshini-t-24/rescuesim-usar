"""RescueSim test suite: physics, kinematics, Tavily parsing, orchestrator, coordinator, API.

All tests run fully offline: API keys are blanked before any project import.
"""
import math
import os

os.environ["NEBIUS_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""

import numpy as np
import pytest

from config import settings as cfg
from simulator import sensors as sn
from simulator.disaster_world import DisasterWorld
from simulator import robot_kinematics as rk
from agents.tavily_hazard_intel import PhysicalHazardModifiers, TavilyHazardIntel, parse_llm_json
from agents.nemotron_orchestrator import NemotronOrchestrator, PlanDecision
from agents.mission_coordinator import POLICIES, run_mission, sample_hazards
from nebius_jobs.batch_policy_eval import compare, summarize


# ----------------------------------------------------------------------------- world
@pytest.fixture(scope="module")
def world():
    return DisasterWorld(seed=1, num_survivors=5, hazard_modifiers=sample_hazards(1))


def test_world_grids_and_bounds(world):
    assert world.elevation_map.shape == (cfg.WORLD_X_M, cfg.WORLD_Y_M)
    assert world.elevation_map.min() >= 0.0
    assert world.elevation_map.max() <= cfg.MAX_RUBBLE_HEIGHT_M + 1e-6
    assert world.structural_instability.min() >= 0.0
    assert world.structural_instability.max() <= 1.0
    assert world.smoke_density.shape[:2] == world.elevation_map.shape


def test_survivor_ranges(world):
    assert len(world.survivors) == 5
    lo_t, hi_t = cfg.SURVIVOR_CORE_TEMP_RANGE_C
    lo_a, hi_a = cfg.SURVIVOR_ACOUSTIC_DB_RANGE
    for s in world.survivors:
        assert lo_t <= s.core_temp_c <= hi_t
        assert lo_a <= s.acoustic_db <= hi_a
        assert s.vital_decay_rate > 0


def test_world_is_seed_deterministic():
    a = DisasterWorld(seed=5, num_survivors=3)
    b = DisasterWorld(seed=5, num_survivors=3)
    assert np.array_equal(a.elevation_map, b.elevation_map)
    assert [(s.x, s.y) for s in a.survivors] == [(s.x, s.y) for s in b.survivors]


def test_aftershock_modifies_terrain():
    w = DisasterWorld(seed=2, num_survivors=3)
    before_version = w.terrain_version
    before_elev = w.elevation_map.copy()
    ev = w.trigger_aftershock(6.0)
    assert w.terrain_version > before_version
    assert not np.array_equal(before_elev, w.elevation_map)
    for key in ("epicenter_xy", "radius_m", "magnitude"):
        assert key in ev


def test_traversability_respects_slope_limit(world):
    trav = world.traversable_mask()
    steep = world.slope_map > cfg.ROVER.max_slope
    assert not trav[steep].any()


def test_hazard_modifiers_change_physics():
    w = DisasterWorld(seed=3, num_survivors=3)
    before = w.physics_snapshot()
    w.apply_hazard_modifiers({**cfg.OFFLINE_HAZARD_PROFILE, "wind_turbulence_mps": 9.0, "smoke_toxicity_index": 0.95})
    after = w.physics_snapshot()
    assert before != after


# --------------------------------------------------------------------------- sensors
def test_beer_lambert():
    assert sn.beer_lambert_transmittance(0.0) == pytest.approx(1.0)
    assert sn.beer_lambert_transmittance(1.0) == pytest.approx(math.exp(-1.0))
    assert sn.beer_lambert_transmittance(2.0) < sn.beer_lambert_transmittance(1.0)


def test_thermal_signal_inverse_square_and_smoke():
    near = sn.thermal_signal(18.0, 4.0, 0.0)
    far = sn.thermal_signal(18.0, 8.0, 0.0)
    assert 3.5 < near / far < 4.5
    assert sn.thermal_signal(18.0, 4.0, 1.0) < near
    assert sn.thermal_signal(18.0, 4.0, 0.0, overburden_m=1.0) < near


def test_lidar_has_36_rays_within_range(world):
    suite = sn.SensorSuite(world, rng=np.random.default_rng(0))
    scan = suite.lidar_scan((5.0, 5.0, world.elevation_at(5.0, 5.0) + 3.0), heading_rad=0.0)
    ranges = np.asarray(scan.ranges, dtype=float)
    hit = np.asarray(scan.hit, dtype=bool)
    clear = np.asarray(scan.clear_range, dtype=float)
    assert ranges.shape == (cfg.SENSORS.lidar_num_rays,)
    assert np.array_equal(np.isnan(ranges), ~hit)
    assert hit.any()
    assert np.nanmax(ranges) <= cfg.SENSORS.lidar_range_m + 0.5
    assert np.all(clear <= cfg.SENSORS.lidar_range_m + 1e-6)
    assert np.all(clear[hit] <= ranges[hit] + 0.5)


def test_acoustic_loss_increases_with_rubble():
    open_air = sn.acoustic_received_db(70.0, 10.0, 0.0)
    buried = sn.acoustic_received_db(70.0, 10.0, 3.0)
    assert buried < open_air


def test_triangulation_recovers_source():
    src = (20.0, 30.0)
    obs = []
    for ox, oy in [(5.0, 5.0), (30.0, 8.0), (10.0, 45.0)]:
        obs.append((ox, oy, math.atan2(src[1] - oy, src[0] - ox)))
    res = sn.triangulate_bearings(obs)
    assert res is not None
    (x, y), _ = res
    assert math.hypot(x - src[0], y - src[1]) < 0.5


# ------------------------------------------------------------------------ kinematics
def test_astar_finds_straight_path_on_open_grid():
    cost = np.zeros((30, 30))
    blocked = np.zeros((30, 30), dtype=bool)
    res = rk.risk_aware_astar(cost, blocked, (2, 2), (25, 2), 1.0)
    assert res is not None
    assert res.cells[0] == (2, 2)
    assert math.hypot(res.cells[-1][0] - 25, res.cells[-1][1] - 2) <= 1.5
    assert res.length_m == pytest.approx(23.0, abs=2.0)


def test_astar_detours_around_wall_and_avoids_risk():
    cost = np.zeros((30, 30))
    blocked = np.zeros((30, 30), dtype=bool)
    blocked[14:17, 0:25] = True
    res = rk.risk_aware_astar(cost, blocked, (2, 2), (25, 2), 1.0)
    assert res is not None and res.length_m > 40.0
    assert not any(blocked[c] for c in res.cells)

    open_blocked = np.zeros((30, 30), dtype=bool)
    risky = np.zeros((30, 30))
    risky[10:20, 0:12] = 50.0
    safe = rk.risk_aware_astar(risky, open_blocked, (2, 2), (25, 2), 1.0)
    assert safe is not None
    assert all(c[1] >= 12 or not (10 <= c[0] < 20) for c in safe.cells[2:-2])


def test_astar_unreachable_returns_none():
    cost = np.zeros((30, 30))
    blocked = np.zeros((30, 30), dtype=bool)
    blocked[10:14, :] = True
    assert rk.risk_aware_astar(cost, blocked, (2, 2), (25, 2), 1.0) is None


def test_structural_loss_probability_monotone():
    assert rk.structural_loss_probability(0.0, 1.0) == 0.0
    lo = rk.structural_loss_probability(0.3, 1.0)
    hi = rk.structural_loss_probability(0.9, 1.0)
    assert 0.0 < lo < hi < 1.0


def test_cost_field_penalizes_unstable_cells(world):
    greedy, _ = rk.build_cost_field(world, cfg.POLICY_BASELINE_GREEDY, None)
    cons, _ = rk.build_cost_field(world, cfg.POLICY_STATIC_CONSERVATIVE, None)
    assert np.allclose(greedy, 0.0)
    assert cons.max() > 0.0


def test_rover_moves_and_drains_battery():
    w = DisasterWorld(seed=4, num_survivors=3, auto_aftershocks=False)
    smap = rk.SharedMap(cfg.WORLD_X_M, cfg.WORLD_Y_M, cfg.CELL_SIZE_M)
    rover = rk.QuadrupedRover(w, smap, weights=cfg.POLICY_STATIC_CONSERVATIVE, rng=np.random.default_rng(0))
    start = (rover.x, rover.y)
    assert rover.plan_to((20.0, 20.0))
    for _ in range(80):
        w.step(cfg.DT_S)
        rover.step(cfg.DT_S)
    tele = rover.telemetry()
    assert math.hypot(rover.x - start[0], rover.y - start[1]) > 1.0 or not tele["alive"]
    assert tele["battery_pct"] < 100.0


def test_drone_stays_in_altitude_band():
    w = DisasterWorld(seed=4, num_survivors=3, auto_aftershocks=False)
    smap = rk.SharedMap(cfg.WORLD_X_M, cfg.WORLD_Y_M, cfg.CELL_SIZE_M)
    drone = rk.MicroDrone(w, smap, rng=np.random.default_rng(0))
    for _ in range(100):
        w.step(cfg.DT_S)
        drone.step(cfg.DT_S)
        z = float(drone.telemetry()["z"])
        assert cfg.DRONE.z_min_m - 0.5 <= z <= cfg.DRONE.z_max_m + 0.5


# --------------------------------------------------------------------------- tavily
def test_parse_llm_json_extracts_numeric_fields():
    out = parse_llm_json('{"seismic_aftershock_prob": 0.3, "wind_turbulence_mps": 5.5, "ambient_temp_c": 11}')
    assert out["seismic_aftershock_prob"] == pytest.approx(0.3)
    assert out["wind_turbulence_mps"] == pytest.approx(5.5)


def test_modifiers_schema_defaults_and_urls():
    m = PhysicalHazardModifiers(source_urls=["https://example.com"])
    assert 0.0 <= m.seismic_aftershock_prob <= 1.0
    assert m.source_urls == ["https://example.com"]


class _FakeTavily:
    def __init__(self):
        self.calls = []

    def search(self, *args, **kwargs):
        self.calls.append(kwargs or args)
        return {"results": [{
            "title": "M6.1 aftershock sequence continues",
            "url": "https://example.com/quake",
            "content": "Aftershock probability of 25 percent in the next 24 hours. Wind gusts of 20 km/h and temperature of 12 C reported at the collapse site.",
            "score": 0.9,
        }]}


def test_tavily_fetch_with_fake_client_returns_hits_and_valid_modifiers():
    fake = _FakeTavily()
    intel = TavilyHazardIntel(client=fake).fetch(location="Tokyo")
    assert fake.calls, "Tavily client was never called"
    assert intel.hits and intel.hits[0].url == "https://example.com/quake"
    assert isinstance(intel.modifiers, PhysicalHazardModifiers)
    lo, hi = cfg.HAZARD_BOUNDS["seismic_aftershock_prob"]
    assert lo <= intel.modifiers.seismic_aftershock_prob <= hi


def test_tavily_offline_fallback_never_crashes():
    intel = TavilyHazardIntel().fetch(location="anywhere")
    assert isinstance(intel.modifiers, PhysicalHazardModifiers)
    assert "offline" in intel.mode.lower()


def test_hazard_intel_applies_to_world():
    intel = TavilyHazardIntel(client=_FakeTavily()).fetch()
    w = DisasterWorld(seed=6, num_survivors=3)
    before = w.physics_snapshot()
    intel.apply_to_world(w)
    assert w.physics_snapshot() != before or intel.modifiers.model_dump() is not None


# ---------------------------------------------------------------------- orchestrator
def test_orchestrator_offline_plan_within_bounds():
    orch = NemotronOrchestrator()
    cands = [{"id": 1, "x": 10.0, "y": 12.0, "confidence": 0.8, "label": "survivor"}]
    dec = orch.plan(trigger="mission_start", hazards=dict(cfg.OFFLINE_HAZARD_PROFILE), world={}, candidates=cands, rover_xy=(2.5, 2.5))
    assert isinstance(dec, PlanDecision)
    assert dec.mode != "live"
    for name, (lo, hi) in cfg.WEIGHT_BOUNDS.items():
        assert lo <= getattr(dec.weights, name) <= hi


def test_orchestrator_weights_rise_with_aftershock_probability():
    orch = NemotronOrchestrator()
    calm = dict(cfg.OFFLINE_HAZARD_PROFILE, seismic_aftershock_prob=0.05)
    risky = dict(cfg.OFFLINE_HAZARD_PROFILE, seismic_aftershock_prob=0.6)
    a = orch.plan(trigger="mission_start", hazards=calm, world={}, candidates=[], rover_xy=(0.0, 0.0))
    b = orch.plan(trigger="aftershock", hazards=risky, world={}, candidates=[], rover_xy=(0.0, 0.0))
    assert b.weights.beta_collapse >= a.weights.beta_collapse


def test_reflex_rejects_fire_hotspot():
    orch = NemotronOrchestrator()
    verdict = orch.classify_anomaly({
        "est_temp_c": 61.0, "snr_db": 38.0, "smoke_transmittance": 0.6, "acoustic_hits": 0, "n_obs": 4,
        "confidence": 0.4, "range_m": 9.0, "ambient_temp_c": 18.0, "smoke_toxicity_index": 0.5,
    })
    assert verdict.label == "false_positive"


# --------------------------------------------------------------------------- mission
def test_sample_hazards_within_bounds_and_deterministic():
    a, b = sample_hazards(11), sample_hazards(11)
    assert a == b
    for key, (lo, hi) in cfg.HAZARD_BOUNDS.items():
        if key in a:
            assert lo <= a[key] <= hi


@pytest.mark.parametrize("policy", POLICIES)
def test_run_mission_returns_valid_metrics(policy):
    m = run_mission(policy=policy, seed=3, max_steps=150)
    assert m["policy"] == policy
    assert 0.0 <= m["rescue_rate"] <= 1.0
    assert m["survivors_rescued"] <= m["survivors_total"]
    assert m["robot_loss"] == (m["rover_lost"] or m["drone_lost"])
    assert m["energy_wh"] >= 0.0


def test_run_mission_is_deterministic():
    a = run_mission(policy="static", seed=7, max_steps=200)
    b = run_mission(policy="static", seed=7, max_steps=200)
    for key in ("time_s", "rescue_rate", "energy_wh", "rover_lost", "survivors_located"):
        assert a[key] == b[key]


# -------------------------------------------------------------------------- batch eval
def _ep(policy, seed, rescued, loss, first=30.0, energy=100.0):
    return {
        "policy": policy, "seed": seed, "error": None, "survivors_total": 5, "survivors_rescued": rescued,
        "robot_loss": loss, "rover_lost": loss, "drone_lost": False, "first_located_s": first,
        "first_rescued_s": first + 20, "energy_wh": energy, "rover_risk_exposure": 10.0,
    }


def test_summarize_statistics():
    eps = [_ep("greedy", i, 3, i % 2 == 0) for i in range(10)]
    s = summarize(eps)
    assert s["rescue_rate_pct"]["mean"] == pytest.approx(60.0)
    assert s["robot_loss_rate_pct"]["mean"] == pytest.approx(50.0)
    assert s["energy_efficiency_wh_per_rescued"] == pytest.approx(1000.0 / 30.0)
    assert s["time_to_first_survivor_s"]["mean"] == pytest.approx(30.0)


def test_summarize_ignores_failed_episodes():
    eps = [_ep("greedy", 0, 5, False), {"policy": "greedy", "seed": 1, "error": "boom"}]
    s = summarize(eps)
    assert s["episodes"] == 1 and s["failed_episodes"] == 1


def test_compare_loss_reduction_math():
    summary = {
        "greedy": summarize([_ep("greedy", i, 3, i < 5) for i in range(10)]),
        "static": summarize([_ep("static", i, 3, i < 4) for i in range(10)]),
        "adaptive": summarize([_ep("adaptive", i, 4, i < 2) for i in range(10)]),
    }
    c = compare(summary)["adaptive_vs_greedy"]
    assert c["robot_loss_reduction_pct"] == pytest.approx(60.0)
    assert c["rescue_rate_delta_pts"] == pytest.approx(20.0)
    assert c["meets_60pct_loss_reduction_target"] is False


# ------------------------------------------------------------------------------ API
def test_api_endpoints_offline():
    from fastapi.testclient import TestClient
    import server

    client = TestClient(server.app)
    health = client.get("/api/health").json()
    assert health["ok"] is True and "OFFLINE" in health["mode"]["nebius"]
    assert client.get("/api/models").json()["online"] is False
    insp = client.get("/api/inspector").json()
    assert "domain_randomization" in insp and insp["cosmos_style_conditioning"]["cosmos_weights_called"] is False
    haz = client.post("/api/hazards", json={"location": "Tokyo earthquake", "seed": 2}).json()
    assert set(haz["modifiers"]) >= {"seismic_aftershock_prob", "wind_turbulence_mps", "ambient_temp_c"}
    assert client.get("/api/hazards/state").json()["active"] is True
    client.post("/api/hazards/clear")


def test_websocket_streams_frames():
    from fastapi.testclient import TestClient
    import server

    client = TestClient(server.app)
    with client.websocket_connect("/ws/sim") as ws:
        ws.send_json({"cmd": "start", "policy": "static", "seed": 3, "survivors": 3})
        init = ws.receive_json()
        assert init["type"] == "init" and "elevation" in init["world"]
        frame = ws.receive_json()
        assert frame["type"] == "frame" and frame["rover"] and frame["drone"]
        ws.send_json({"cmd": "stop"})

def test_compare_endpoint_runs_three_policies():
    from fastapi.testclient import TestClient
    import server

    res = TestClient(server.app).post("/api/compare", json={"seed": 3, "survivors": 3, "max_steps": 40}).json()
    assert set(res["policies"]) == {"greedy", "static", "adaptive"}
    for pol, m in res["policies"].items():
        assert "error" not in m, m
        assert m["policy"] == pol and m["survivors_total"] == 3
