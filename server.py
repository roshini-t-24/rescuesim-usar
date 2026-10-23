"""FastAPI command-center backend: REST + WebSocket telemetry for the RescueSim dashboard."""
from __future__ import annotations

import asyncio
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.mission_coordinator import POLICIES, MissionCoordinator, _clean, _ev_type, sample_hazards
from agents.nemotron_orchestrator import NemotronOrchestrator
from agents.tavily_hazard_intel import TavilyHazardIntel
from config import settings as cfg
from simulator import disaster_world as dw
from simulator.disaster_world import DisasterWorld

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
RESULTS = ROOT / "results"
STATIC.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
BENCH_FILE = RESULTS / "serverless_eval.json"
UI_BENCH_FILE = RESULTS / "ui_eval.json"
UI_BENCH_LOG = RESULTS / "ui_eval.log"

ORCH = NemotronOrchestrator()
HAZ: Dict[str, Any] = {"modifiers": None, "intel": None, "ts": None, "location": None}
BENCH: Dict[str, Any] = {"proc": None, "started": None, "episodes": 0, "live": False, "log_fh": None}

app = FastAPI(title="RescueSim USAR Command Center", version="1.0.0")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pick_hazards(seed: int, use_live: bool) -> Dict[str, Any]:
    if use_live and HAZ["modifiers"]:
        return dict(HAZ["modifiers"])
    return sample_hazards(seed)


def _world_payload(coord: MissionCoordinator) -> Dict[str, Any]:
    w = coord.world
    st = 2
    cell = float(getattr(w, "cell", cfg.CELL_SIZE_M))
    elev = np.asarray(w.elevation_map, dtype=float)
    inst = np.asarray(w.structural_instability, dtype=float)
    pil = np.asarray(w.pillar_mask, dtype=float)
    nx, ny = elev.shape
    return _clean({
        "x": (np.arange(0, nx, st) + 0.5) * cell,
        "y": (np.arange(0, ny, st) + 0.5) * cell,
        "elevation": elev.T[::st, ::st],
        "instability": inst.T[::st, ::st],
        "pillars": pil.T[::st, ::st],
        "smoke_points": w.smoke_points(threshold=0.4, stride=2, max_points=1500),
        "size_m": [cfg.WORLD_X_M, cfg.WORLD_Y_M, cfg.WORLD_Z_M],
        "terrain_version": getattr(w, "terrain_version", 0),
    })


def _advance(coord: MissionCoordinator, n_steps: int, quake: Optional[float]) -> Dict[str, Any]:
    quake_event = None
    if quake is not None:
        ev = coord.world.trigger_aftershock(float(quake))
        ev = dict(ev) if isinstance(ev, dict) else {"raw": str(ev)}
        if "aftershock" not in _ev_type(ev):
            ev["type"] = "aftershock"
        ev.setdefault("magnitude", float(quake))
        coord._on_world_event(ev)
        quake_event = _clean(ev)
    for _ in range(n_steps):
        if coord.done:
            break
        coord.step()
    fr = coord.frame()
    fr["plans_count"] = len(coord.plans)
    fr["last_plan"] = coord.plans[-1] if coord.plans else None
    fr["reflex_calls"] = coord.reflex_calls
    if quake_event is not None:
        fr["quake"] = quake_event
    return fr


def _build_session(policy: str, seed: int, use_live: bool, survivors: int) -> MissionCoordinator:
    return MissionCoordinator(
        policy=policy,
        seed=seed,
        hazards=_pick_hazards(seed, use_live),
        num_survivors=survivors,
        orchestrator=ORCH if policy == "adaptive" else None,
    )


# --------------------------------------------------------------------------- #
# Health, mode, models
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "mode": cfg.mode_banner(),
        "orchestrator": _clean(ORCH.status()),
        "policies": list(POLICIES),
        "planner_model": cfg.PLANNER_MODEL_ID,
        "reflex_model": cfg.REFLEX_MODEL_ID,
        "hazards_active": HAZ["modifiers"] is not None,
    }


@app.get("/api/models")
def list_models() -> Dict[str, Any]:
    """Lists model IDs served by Nebius Token Factory so PLANNER/REFLEX IDs can be verified."""
    key = cfg.get_nebius_key()
    if not key:
        return {"online": False, "error": "NEBIUS_API_KEY not set", "models": []}
    try:
        from openai import OpenAI

        client = OpenAI(base_url=cfg.NEBIUS_BASE_URL, api_key=key, timeout=cfg.LLM_TIMEOUT_S)
        ids = sorted(m.id for m in client.models.list())
    except Exception as exc:  # noqa: BLE001
        return {"online": True, "error": f"{type(exc).__name__}: {exc}", "models": []}
    nv = [i for i in ids if "nemotron" in i.lower() or "nvidia" in i.lower()]
    return {
        "online": True,
        "configured": {"planner": cfg.PLANNER_MODEL_ID, "reflex": cfg.REFLEX_MODEL_ID},
        "planner_found": cfg.PLANNER_MODEL_ID in ids,
        "reflex_found": cfg.REFLEX_MODEL_ID in ids,
        "nvidia_models": nv,
        "total_models": len(ids),
        "models": ids,
    }


# --------------------------------------------------------------------------- #
# Tavily hazard intelligence (Tab 2)
# --------------------------------------------------------------------------- #
class HazardRequest(BaseModel):
    location: Optional[str] = None
    query: Optional[str] = None
    seed: int = 3


@app.post("/api/hazards")
async def fetch_hazards(req: HazardRequest) -> Dict[str, Any]:
    def work() -> Dict[str, Any]:
        extractor = ORCH.as_llm_extractor() if cfg.nebius_online() else None
        intel = TavilyHazardIntel(llm_extractor=extractor).fetch(location=req.location, query=req.query)
        world = DisasterWorld(seed=req.seed, num_survivors=cfg.DEFAULT_NUM_SURVIVORS)
        before = _clean(world.physics_snapshot())
        applied = _clean(intel.apply_to_world(world))
        after = _clean(world.physics_snapshot())
        mods = _clean(intel.modifiers.model_dump())
        HAZ.update(modifiers=mods, intel=_clean(intel.to_dict()), ts=time.time(), location=req.location or req.query)
        return {
            "intel": _clean(intel.to_dict()),
            "modifiers": mods,
            "physics_before": before,
            "physics_after": after,
            "applied": applied,
            "active": True,
        }

    return await asyncio.to_thread(work)


@app.get("/api/hazards/state")
def hazards_state() -> Dict[str, Any]:
    return {
        "active": HAZ["modifiers"] is not None,
        "modifiers": HAZ["modifiers"],
        "location": HAZ["location"],
        "fetched_at": HAZ["ts"],
        "mode": cfg.mode_banner(),
    }


@app.post("/api/hazards/clear")
def hazards_clear() -> Dict[str, Any]:
    HAZ.update(modifiers=None, intel=None, ts=None, location=None)
    return {"active": False}


# --------------------------------------------------------------------------- #
# Benchmark (Tab 3): subprocess so the server's live/offline mode is never altered
# --------------------------------------------------------------------------- #
class BenchRequest(BaseModel):
    episodes: int = Field(10, ge=2, le=100)
    live: bool = False


def _bench_running() -> bool:
    proc = BENCH["proc"]
    return proc is not None and proc.poll() is None


@app.post("/api/benchmark/run")
def benchmark_run(req: BenchRequest) -> Dict[str, Any]:
    if _bench_running():
        raise HTTPException(409, "a benchmark is already running")
    cmd = [
        sys.executable, "-m", "nebius_jobs.batch_policy_eval",
        "--episodes", str(req.episodes), "--workers", "3", "--output", str(UI_BENCH_FILE),
    ]
    if req.live:
        cmd.append("--live")
    if BENCH["log_fh"]:
        BENCH["log_fh"].close()
    BENCH["log_fh"] = open(UI_BENCH_LOG, "w", encoding="utf-8")
    BENCH["proc"] = subprocess.Popen(cmd, cwd=str(ROOT), stdout=BENCH["log_fh"], stderr=subprocess.STDOUT)
    BENCH.update(started=time.time(), episodes=req.episodes, live=req.live)
    return {"started": True, "episodes": req.episodes, "tasks": req.episodes * len(POLICIES)}


@app.get("/api/benchmark/status")
def benchmark_status() -> Dict[str, Any]:
    proc = BENCH["proc"]
    if proc is None:
        return {"state": "idle"}
    lines: List[str] = []
    if UI_BENCH_LOG.exists():
        lines = UI_BENCH_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    progress = None
    for ln in reversed(lines):
        if ln.startswith("[") and "/" in ln.split("]")[0]:
            a, b = ln[1:].split("]")[0].split("/")
            progress = {"done": int(a), "total": int(b)}
            break
    rc = proc.poll()
    return {
        "state": "running" if rc is None else ("finished" if rc == 0 else "failed"),
        "returncode": rc,
        "progress": progress,
        "elapsed_s": round(time.time() - BENCH["started"], 1),
        "episodes": BENCH["episodes"],
        "live": BENCH["live"],
        "log_tail": lines[-8:],
    }


@app.get("/api/benchmark/result")
def benchmark_result(source: str = "serverless", full: bool = False) -> Dict[str, Any]:
    path = BENCH_FILE if source == "serverless" else UI_BENCH_FILE
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found; run the benchmark first")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not full:
        data.pop("episodes", None)
    data["source_file"] = path.name
    return data


# --------------------------------------------------------------------------- #
# Architecture inspector (Tab 4)
# --------------------------------------------------------------------------- #
def _randomization() -> Dict[str, Any]:
    samples = [sample_hazards(s) for s in range(200)]
    out: Dict[str, Any] = {}
    for key, val in samples[0].items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            vals = [float(s[key]) for s in samples]
            out[key] = {"min": min(vals), "max": max(vals), "mean": statistics.fmean(vals)}
    return out


def _latency_summary(log: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[float]] = {}
    for e in log:
        lat = e.get("latency_ms")
        if lat is None:
            continue
        label = str(e.get("role") or e.get("model") or "call")
        groups.setdefault(f"{label}|{e.get('mode', '')}", []).append(float(lat))
    return {
        k: {
            "calls": len(v),
            "mean_ms": round(statistics.fmean(v), 1),
            "p95_ms": round(sorted(v)[max(0, int(0.95 * len(v)) - 1)], 1),
            "max_ms": round(max(v), 1),
        }
        for k, v in groups.items()
    }


@app.get("/api/inspector")
def inspector(limit: int = 12) -> Dict[str, Any]:
    log = _clean(ORCH.payload_log(limit=200))
    return {
        "mode": cfg.mode_banner(),
        "orchestrator": _clean(ORCH.status()),
        "models": {
            "base_url": cfg.NEBIUS_BASE_URL,
            "planner": cfg.PLANNER_MODEL_ID,
            "reflex": cfg.REFLEX_MODEL_ID,
            "timeout_s": cfg.LLM_TIMEOUT_S,
            "max_tokens": {"planner": cfg.LLM_MAX_TOKENS_PLANNER, "reflex": cfg.LLM_MAX_TOKENS_REFLEX},
        },
        "payloads": log[-limit:],
        "payload_count": len(log),
        "latency": _latency_summary(log),
        "domain_randomization": _randomization(),
        "hazard_bounds": _clean(cfg.HAZARD_BOUNDS),
        "offline_hazard_profile": _clean(cfg.OFFLINE_HAZARD_PROFILE),
        "cosmos_style_conditioning": {
            "implementation": "procedural Gaussian / Ornstein-Uhlenbeck smoke, wind and instability fields",
            "cosmos_weights_called": False,
            "smoke_peak": dw.SMOKE_PEAK,
            "smoke_tau_s": dw.SMOKE_TAU_S,
            "smoke_max": dw.SMOKE_MAX,
            "pillar_collapse_prob": dw.PILLAR_COLLAPSE_PROB,
            "pillar_spacing_m": dw.PILLAR_SPACING_M,
        },
    }


@app.post("/api/inspector/probe")
async def inspector_probe() -> Dict[str, Any]:
    cases = {
        "electrical_fire_hotspot": {
            "est_temp_c": 61.0, "snr_db": 38.0, "smoke_transmittance": 0.6, "acoustic_hits": 0,
            "n_obs": 4, "confidence": 0.4, "range_m": 9.0, "ambient_temp_c": 18.0, "smoke_toxicity_index": 0.5,
        },
        "human_survivor_36_8C": {
            "est_temp_c": 36.8, "snr_db": 24.0, "smoke_transmittance": 0.8, "acoustic_hits": 2,
            "n_obs": 5, "confidence": 0.45, "range_m": 7.0, "ambient_temp_c": 18.0, "smoke_toxicity_index": 0.5,
        },
    }

    def work() -> Dict[str, Any]:
        out = {}
        for name, feat in cases.items():
            out[name] = {"features": feat, "verdict": _clean(ORCH.classify_anomaly(feat).to_dict())}
        return out

    return await asyncio.to_thread(work)


# --------------------------------------------------------------------------- #
# Live simulation WebSocket (Tab 1)
# --------------------------------------------------------------------------- #
@app.websocket("/ws/sim")
async def ws_sim(ws: WebSocket) -> None:
    await ws.accept()
    ctl: Dict[str, Any] = {"start": None, "paused": False, "stop": False, "speed": 1.0, "quake": None, "gone": False}

    async def reader() -> None:
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                cmd = msg.get("cmd")
                if cmd == "start":
                    ctl.update(start=msg, paused=False, stop=True if ctl["start"] is not None else False)
                    ctl["start"] = msg
                    ctl["stop"] = False
                    ctl["restart"] = True
                elif cmd == "pause":
                    ctl["paused"] = True
                elif cmd == "resume":
                    ctl["paused"] = False
                elif cmd == "stop":
                    ctl["stop"] = True
                elif cmd == "speed":
                    ctl["speed"] = float(min(max(msg.get("value", 1.0), 0.25), 8.0))
                elif cmd == "aftershock":
                    ctl["quake"] = float(min(max(msg.get("magnitude", 5.0), 3.0, ), 7.5))
        except (WebSocketDisconnect, RuntimeError):
            ctl["gone"] = True
            ctl["stop"] = True
        except Exception:  # noqa: BLE001
            ctl["gone"] = True
            ctl["stop"] = True

    reader_task = asyncio.create_task(reader())
    try:
        while not ctl["gone"]:
            if ctl["start"] is None or not ctl.get("restart"):
                await asyncio.sleep(0.1)
                continue
            req = ctl["start"]
            ctl["restart"] = False
            ctl["stop"] = False
            policy = req.get("policy", "adaptive")
            if policy not in POLICIES:
                await ws.send_json({"type": "error", "message": f"unknown policy {policy}"})
                continue
            seed = int(req.get("seed", 3))
            try:
                coord = await asyncio.to_thread(
                    _build_session, policy, seed, bool(req.get("use_live_hazards", False)),
                    int(req.get("survivors", cfg.DEFAULT_NUM_SURVIVORS)),
                )
                await ws.send_json({
                    "type": "init",
                    "policy": policy,
                    "seed": seed,
                    "hazards": _clean(coord.hazards),
                    "mode": cfg.mode_banner(),
                    "world": await asyncio.to_thread(_world_payload, coord),
                    "frame": coord.frame(),
                })
                frame_no = 0
                while not ctl["stop"] and not ctl["gone"] and not ctl.get("restart"):
                    if ctl["paused"]:
                        await asyncio.sleep(0.1)
                        continue
                    t0 = time.perf_counter()
                    quake, ctl["quake"] = ctl["quake"], None
                    n_steps = max(1, int(round(4 * ctl["speed"])))
                    fr = await asyncio.to_thread(_advance, coord, n_steps, quake)
                    frame_no += 1
                    if frame_no % 15 == 0:
                        fr["smoke"] = _clean(coord.world.smoke_points(threshold=0.4, stride=2, max_points=1500))
                    msg = {"type": "frame", **fr}
                    if quake is not None:
                        msg["world"] = await asyncio.to_thread(_world_payload, coord)
                    await ws.send_json(msg)
                    if coord.done:
                        await ws.send_json({
                            "type": "done",
                            "metrics": _clean(coord.metrics()),
                            "orchestrator": _clean(ORCH.status()),
                        })
                        break
                    await asyncio.sleep(max(0.0, 0.08 - (time.perf_counter() - t0)))
                if not ctl.get("restart"):
                    ctl["start"] = None
            except WebSocketDisconnect:
                break
            except Exception as exc:  # noqa: BLE001
                try:
                    await ws.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                except Exception:  # noqa: BLE001
                    break
                ctl["start"] = None
    finally:
        reader_task.cancel()


# --------------------------------------------------------------------------- #
# Static dashboard
# --------------------------------------------------------------------------- #
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def index():
    page = STATIC / "index.html"
    if page.exists():
        return FileResponse(page)
    return JSONResponse({"detail": "static/index.html not generated yet", "api_docs": "/docs"})


if __name__ == "__main__":
    print(f"RescueSim command center: http://localhost:{cfg.SERVER_PORT}", flush=True)
    uvicorn.run("server:app", host=cfg.SERVER_HOST, port=cfg.SERVER_PORT, reload=False)