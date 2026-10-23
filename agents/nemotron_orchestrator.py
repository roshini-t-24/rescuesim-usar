"""agents/nemotron_orchestrator.py

Dual-model routing on Nebius Token Factory (OpenAI-compatible API):

    Strategic Mission Planner  (cfg.PLANNER_MODEL_ID, Nemotron Ultra/70B class)
        called at mission start and after an aftershock / hazard update. Outputs Risk-Aware A*
        weights (alpha_slope, beta_collapse, gamma_smoke) and the rover/drone task allocation.
    Tactical Reflex Agent      (cfg.REFLEX_MODEL_ID, Nemotron-Mini-4B class)
        called on a sensor anomaly: survivor vs. false-positive heat source.

Graceful offline fallback: with no NEBIUS_API_KEY, or on any API/parse error, a deterministic
heuristic optimizer / physics classifier answers instead. Every decision carries mode
("nemotron" | "heuristic"), latency and any error, and every call is logged for the UI inspector.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from config import settings as cfg

log = logging.getLogger("rescuesim.nemotron")

PLANNER_SYSTEM_PROMPT = (
    "You are the strategic mission planner for a search-and-rescue robot team in a collapsed building: "
    "one quadruped rover and one micro-drone. The rover plans with Risk-Aware A* where edge cost = "
    "distance + alpha_slope*slope + beta_collapse*structural_instability + gamma_smoke*smoke_density. "
    "Raise beta_collapse when aftershock risk or high-risk area is large; raise alpha_slope when terrain "
    "is steep or the traversable fraction is low; raise gamma_smoke when smoke is heavy. Reply with ONE "
    "JSON object only, with keys: alpha_slope (0-12), beta_collapse (0-20), gamma_smoke (0-6), "
    "rover_target_order (list of candidate ids, best first), drone_mode (\"scout\" or \"relay_hold\"), "
    "drone_max_altitude_m (number), rationale (one sentence). Use only ids from the candidate list."
)

REFLEX_SYSTEM_PROMPT = (
    "You are the tactical reflex agent of a search-and-rescue robot. Classify one thermal/acoustic "
    "anomaly. Human survivors under rubble read cooler than 37.5 C because rubble attenuates the signal, "
    "electrical fires read far hotter than 45 C, and warm pipes or machinery can read 36-39 C, so a "
    "human-like temperature without acoustic corroboration is only 'uncertain'. Reply with ONE JSON "
    "object only: {\"label\": \"survivor\" | \"false_positive\" | \"uncertain\", \"confidence\": 0-1, "
    "\"rationale\": \"one sentence\"}."
)


class LLMUnavailable(RuntimeError):
    """No key, no client, network error or timeout."""


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------
@dataclass
class PlanDecision:
    weights: cfg.PolicyWeights
    rover_target_order: List[int]
    drone_mode: str
    drone_max_altitude_m: float
    rationale: str
    mode: str                                   # "nemotron" | "heuristic"
    trigger: str = "mission_start"
    model: str = "heuristic"
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"weights": self.weights.as_dict(), "rover_target_order": self.rover_target_order,
                "drone_mode": self.drone_mode, "drone_max_altitude_m": round(self.drone_max_altitude_m, 1),
                "rationale": self.rationale, "mode": self.mode, "trigger": self.trigger,
                "model": self.model, "latency_ms": round(self.latency_ms, 1), "error": self.error}


@dataclass
class ReflexVerdict:
    label: str                                  # "survivor" | "false_positive" | "uncertain"
    confidence: float
    rationale: str
    mode: str
    model: str = "heuristic"
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "confidence": round(self.confidence, 3), "rationale": self.rationale,
                "mode": self.mode, "model": self.model, "latency_ms": round(self.latency_ms, 1),
                "error": self.error}


# --------------------------------------------------------------------------
# Deterministic fallbacks (also used by the headless batch evaluator)
# --------------------------------------------------------------------------
def _clamp_weight(name: str, v: float) -> float:
    lo, hi = cfg.WEIGHT_BOUNDS[name]
    return float(min(max(v, lo), hi))


def heuristic_weights(hazards: Dict[str, Any], world: Dict[str, Any], trigger: str = "mission_start") -> cfg.PolicyWeights:
    """Map hazard telemetry + map statistics to A* risk weights."""
    p = float(hazards.get("seismic_aftershock_prob", 0.1))
    tox = float(hazards.get("smoke_toxicity_index", 0.3))
    hr = float(world.get("high_risk_fraction", 0.0))
    mean_slope = float(world.get("mean_slope", 0.07))
    trav = float(world.get("traversable_fraction", 0.9))
    after = int(world.get("aftershocks", 0))
    peak_smoke = float(world.get("peak_smoke", 0.0))
    beta = 4.0 + 14.0 * p + 8.0 * hr + 2.0 * min(after, 3) + (3.0 if trigger == "aftershock" else 0.0)
    alpha = 3.0 + 8.0 * mean_slope + 6.0 * (1.0 - trav)
    gamma = 0.5 + 2.5 * tox + 0.4 * peak_smoke
    return cfg.PolicyWeights(_clamp_weight("alpha_slope", alpha), _clamp_weight("beta_collapse", beta),
                             _clamp_weight("gamma_smoke", gamma))


def heuristic_allocation(hazards: Dict[str, Any], candidates: Sequence[Dict[str, Any]],
                         rover_xy: Tuple[float, float], robots: Optional[Dict[str, Any]] = None
                         ) -> Tuple[List[int], str, float]:
    wind = float(hazards.get("wind_turbulence_mps", 1.0))
    drone_batt = float((robots or {}).get("drone", {}).get("battery_pct", 100.0))
    mode = "relay_hold" if (wind > 10.0 or drone_batt < 15.0) else "scout"
    max_alt = 12.0 if wind < 6.0 else (9.0 if wind < 9.0 else 7.0)

    def utility(c: Dict[str, Any]) -> float:
        d = math.hypot(float(c["x"]) - rover_xy[0], float(c["y"]) - rover_xy[1])
        return (float(c.get("confidence", 0.5)) * float(c.get("vitality", 1.0))
                / (1.0 + d / 20.0 + 2.0 * float(c.get("risk", 0.3))))

    order = [int(c["id"]) for c in sorted(candidates, key=utility, reverse=True)]
    return order, mode, max_alt


def heuristic_reflex(features: Dict[str, Any]) -> ReflexVerdict:
    """Physics-based classification of a thermal anomaly.

    features: est_temp_c, ambient_temp_c, acoustic_corroborated (bool), smoke_transmittance (0-1)
    """
    t = float(features.get("est_temp_c", 0.0))
    amb = float(features.get("ambient_temp_c", 18.0))
    acoustic = bool(features.get("acoustic_corroborated", False))
    t_smoke = float(features.get("smoke_transmittance", 1.0))
    lo = amb + 4.0                                   # rubble-attenuated humans read well below 37 C
    if t >= 45.0:
        return ReflexVerdict("false_positive", min(0.99, 0.75 + (t - 45.0) / 120.0),
                             f"{t:.0f} C is far above human range: electrical or fire hotspot", "heuristic")
    if lo <= t <= 39.5:
        if acoustic:
            return ReflexVerdict("survivor", 0.92, f"{t:.1f} C in human band with acoustic tapping corroboration",
                                 "heuristic")
        conf = 0.45 + 0.1 * t_smoke
        return ReflexVerdict("uncertain", conf, f"{t:.1f} C is human-like but pipes/machinery overlap; "
                             "needs acoustic confirmation", "heuristic")
    if 39.5 < t < 45.0:
        return ReflexVerdict("uncertain", 0.5, f"{t:.1f} C is above normal human temperature but below fire range",
                             "heuristic")
    if acoustic:
        return ReflexVerdict("uncertain", 0.5, f"{t:.1f} C is cold but tapping was heard; rubble may mask heat",
                             "heuristic")
    return ReflexVerdict("false_positive", 0.6, f"{t:.1f} C is near ambient: no human heat signature", "heuristic")


# --------------------------------------------------------------------------
# JSON helper
# --------------------------------------------------------------------------
def parse_json_object(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        raise ValueError("empty reply")
    txt = re.sub(r"```(?:json)?", "", text).strip()
    a, b = txt.find("{"), txt.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object in reply")
    obj = json.loads(txt[a:b + 1])
    if not isinstance(obj, dict):
        raise ValueError("reply is not an object")
    return obj


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
class NemotronOrchestrator:
    LOG_CAP = 60

    def __init__(self, client: Any = None) -> None:
        self._client = client
        self.calls: List[Dict[str, Any]] = []

    # ---- client / transport ----
    def _get_client(self) -> Any:
        if self._client is None:
            key = cfg.get_nebius_key()
            if key is None:
                raise LLMUnavailable("NEBIUS_API_KEY not set")
            from openai import OpenAI
            self._client = OpenAI(base_url=cfg.NEBIUS_BASE_URL, api_key=key,
                                  timeout=cfg.LLM_TIMEOUT_S, max_retries=1)
        return self._client

    def _chat(self, role: str, system: str, user: str, max_tokens: int,
              temperature: float) -> Tuple[str, float, str]:
        model = cfg.PLANNER_MODEL_ID if role == "planner" else cfg.REFLEX_MODEL_ID
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        entry: Dict[str, Any] = {"t": time.time(), "role": role, "model": model, "endpoint": cfg.NEBIUS_BASE_URL,
                                 "request": {"model": model, "temperature": temperature, "max_tokens": max_tokens,
                                             "messages": [{"role": m["role"], "content": m["content"][:6000]}
                                                          for m in messages]},
                                 "response": None, "latency_ms": None, "error": None}
        self._record(entry)
        t0 = time.perf_counter()
        try:
            client = self._get_client()
            resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                                  max_tokens=max_tokens)
            text = (resp.choices[0].message.content or "").strip()
        except LLMUnavailable as exc:
            entry["error"] = str(exc)
            raise
        except Exception as exc:
            entry["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise LLMUnavailable(entry["error"]) from exc
        ms = (time.perf_counter() - t0) * 1000.0
        entry["latency_ms"], entry["response"] = round(ms, 1), text[:4000]
        return text, ms, model

    def _record(self, entry: Dict[str, Any]) -> None:
        self.calls.append(entry)
        if len(self.calls) > self.LOG_CAP:
            del self.calls[0:len(self.calls) - self.LOG_CAP]

    # ---- strategic planner ----
    def plan(self, trigger: str, hazards: Dict[str, Any], world: Dict[str, Any],
             candidates: Optional[Sequence[Dict[str, Any]]] = None,
             rover_xy: Tuple[float, float] = (0.0, 0.0),
             current_weights: Optional[cfg.PolicyWeights] = None,
             robots: Optional[Dict[str, Any]] = None) -> PlanDecision:
        cands = list(candidates or [])
        h_weights = heuristic_weights(hazards, world, trigger)
        h_order, h_mode, h_alt = heuristic_allocation(hazards, cands, rover_xy, robots)
        fallback = PlanDecision(
            h_weights, h_order, h_mode, h_alt,
            "Deterministic optimizer: beta rises with aftershock probability and high-risk area, "
            "alpha with slope, gamma with smoke.", "heuristic", trigger)
        if not cfg.nebius_online() and self._client is None:
            return fallback

        payload = {"trigger": trigger, "hazards": {k: v for k, v in hazards.items() if k != "source_urls"},
                   "map": {k: world[k] for k in ("traversable_fraction", "mean_slope", "high_risk_fraction",
                                                 "mean_instability", "mean_smoke", "peak_smoke", "aftershocks",
                                                 "survivors_total", "survivors_located", "survivors_rescued")
                           if k in world},
                   "candidates": cands, "rover_xy": [round(rover_xy[0], 1), round(rover_xy[1], 1)],
                   "current_weights": (current_weights.as_dict() if current_weights else None),
                   "robots": robots or {}}
        try:
            text, ms, model = self._chat("planner", PLANNER_SYSTEM_PROMPT, json.dumps(payload),
                                         cfg.LLM_MAX_TOKENS_PLANNER, 0.2)
            obj = parse_json_object(text)
            got = {k: float(obj[k]) for k in ("alpha_slope", "beta_collapse", "gamma_smoke")
                   if k in obj and obj[k] is not None and math.isfinite(float(obj[k]))}
            if not got:
                raise ValueError("no weights in reply")
            weights = cfg.PolicyWeights(
                _clamp_weight("alpha_slope", got.get("alpha_slope", h_weights.alpha_slope)),
                _clamp_weight("beta_collapse", got.get("beta_collapse", h_weights.beta_collapse)),
                _clamp_weight("gamma_smoke", got.get("gamma_smoke", h_weights.gamma_smoke)))
            valid_ids = {int(c["id"]) for c in cands}
            order: List[int] = []
            for i in obj.get("rover_target_order", []) or []:
                try:
                    i = int(i)
                except (TypeError, ValueError):
                    continue
                if i in valid_ids and i not in order:
                    order.append(i)
            order += [i for i in h_order if i not in order]
            mode = obj.get("drone_mode") if obj.get("drone_mode") in ("scout", "relay_hold") else h_mode
            alt = float(obj.get("drone_max_altitude_m", h_alt))
            alt = float(min(max(alt, cfg.DRONE.z_min_m + 1.0), cfg.DRONE.z_max_m)) if math.isfinite(alt) else h_alt
            return PlanDecision(weights, order, mode, alt, str(obj.get("rationale", ""))[:300], "nemotron",
                                trigger, model, ms)
        except Exception as exc:
            log.warning("Planner fell back to heuristic: %s", exc)
            fallback.error = f"{type(exc).__name__}: {exc}"
            return fallback

    # ---- tactical reflex ----
    def classify_anomaly(self, features: Dict[str, Any]) -> ReflexVerdict:
        fallback = heuristic_reflex(features)
        if not cfg.nebius_online() and self._client is None:
            return fallback
        try:
            text, ms, model = self._chat("reflex", REFLEX_SYSTEM_PROMPT, json.dumps(features),
                                         cfg.LLM_MAX_TOKENS_REFLEX, 0.0)
            obj = parse_json_object(text)
            label = str(obj.get("label", "")).strip().lower()
            if label not in ("survivor", "false_positive", "uncertain"):
                raise ValueError(f"bad label {label!r}")
            conf = float(obj.get("confidence", 0.5))
            if not math.isfinite(conf):
                raise ValueError("bad confidence")
            return ReflexVerdict(label, min(max(conf, 0.0), 1.0), str(obj.get("rationale", ""))[:240],
                                 "nemotron", model, ms)
        except Exception as exc:
            log.warning("Reflex fell back to heuristic: %s", exc)
            fallback.error = f"{type(exc).__name__}: {exc}"
            return fallback

    # ---- hook for TavilyHazardIntel(llm_extractor=...) ----
    def as_llm_extractor(self) -> Callable[[str], Optional[str]]:
        from agents.tavily_hazard_intel import EXTRACTION_SYSTEM_PROMPT

        def extractor(prompt: str) -> Optional[str]:
            if not cfg.nebius_online() and self._client is None:
                return None
            text, _, _ = self._chat("planner", EXTRACTION_SYSTEM_PROMPT, prompt, 300, 0.0)
            return text
        return extractor

    # ---- telemetry for the UI inspector ----
    def payload_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self.calls[-limit:]

    def status(self) -> Dict[str, Any]:
        done = [c for c in self.calls if c["latency_ms"] is not None and not c["error"]]
        lat = [c["latency_ms"] for c in done]
        return {"modes": cfg.mode_banner(), "planner_model": cfg.PLANNER_MODEL_ID,
                "reflex_model": cfg.REFLEX_MODEL_ID, "endpoint": cfg.NEBIUS_BASE_URL,
                "calls": len(self.calls), "ok_calls": len(done),
                "errors": sum(1 for c in self.calls if c["error"]),
                "mean_latency_ms": round(sum(lat) / len(lat), 1) if lat else None}