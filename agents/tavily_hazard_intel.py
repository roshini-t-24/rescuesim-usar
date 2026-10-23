"""agents/tavily_hazard_intel.py

Runtime Tavily search -> PhysicalHazardModifiers that directly change the physics engine.

Pipeline
    1. TavilyClient(api_key=...).search(...)  for seismic/structural and weather context
    2. extraction:  Nemotron JSON extractor (if an llm_extractor is supplied)
                    + deterministic regex/unit-conversion extractor (always available)
    3. strict Pydantic schema with physical clamps (cfg.HAZARD_BOUNDS)
    4. DisasterWorld.apply_hazard_modifiers(...)  -> before/after physics

Offline behaviour: with no TAVILY_API_KEY (or on any network/library error) the cached profile
cfg.OFFLINE_HAZARD_PROFILE is used and the mode is reported as "offline", so nothing crashes
and the UI never presents offline values as live search results.
"""
from __future__ import annotations

import json
import logging
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from config import settings as cfg

log = logging.getLogger("rescuesim.tavily")

LLMExtractor = Callable[[str], Union[str, Dict[str, Any], None]]

EXTRACTION_SYSTEM_PROMPT = (
    "You convert disaster-response news and weather snippets into physics parameters for a "
    "search-and-rescue simulator. Reply with ONE JSON object and nothing else, with exactly these keys: "
    "seismic_aftershock_prob (0-1, chance of a damaging aftershock during the next hour), "
    "wind_turbulence_mps (m/s), ambient_temp_c (degrees C), smoke_toxicity_index (0-1), "
    "fema_safety_margin_m (metres of stand-off from unstable structure, typically 1.5-4). "
    "Omit a key if the snippets give no basis for it. Never invent numbers."
)


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
class PhysicalHazardModifiers(BaseModel):
    seismic_aftershock_prob: float = 0.10
    wind_turbulence_mps: float = 1.0
    ambient_temp_c: float = 18.0
    smoke_toxicity_index: float = 0.30
    fema_safety_margin_m: float = 2.0
    source_urls: List[str] = Field(default_factory=list)

    @field_validator("seismic_aftershock_prob", "wind_turbulence_mps", "ambient_temp_c",
                     "smoke_toxicity_index", "fema_safety_margin_m", mode="before")
    @classmethod
    def _finite_float(cls, v: Any) -> float:
        f = float(v)
        if not math.isfinite(f):
            raise ValueError("non-finite value")
        return f

    @field_validator("seismic_aftershock_prob", "wind_turbulence_mps", "ambient_temp_c",
                     "smoke_toxicity_index", "fema_safety_margin_m", mode="after")
    @classmethod
    def _clamp(cls, v: float, info: Any) -> float:
        lo, hi = cfg.HAZARD_BOUNDS[info.field_name]
        return float(min(max(v, lo), hi))

    @field_validator("source_urls", mode="before")
    @classmethod
    def _urls(cls, v: Any) -> List[str]:
        return [str(u) for u in (v or []) if isinstance(u, str)]


NUMERIC_FIELDS = ("seismic_aftershock_prob", "wind_turbulence_mps", "ambient_temp_c",
                  "smoke_toxicity_index", "fema_safety_margin_m")


@dataclass
class SearchHit:
    title: str
    url: str
    content: str
    score: float = 0.0

    def to_dict(self, max_chars: int = 300) -> Dict[str, Any]:
        return {"title": self.title, "url": self.url, "score": round(self.score, 3),
                "snippet": self.content[:max_chars]}


@dataclass
class HazardIntel:
    queries: List[str]
    mode: str                                   # "live" | "offline"
    extraction: str                             # "nemotron+heuristic" | "heuristic" | "offline_profile" ...
    modifiers: PhysicalHazardModifiers
    hits: List[SearchHit] = field(default_factory=list)
    evidence: Dict[str, str] = field(default_factory=dict)       # field -> short source-derived note
    latency_ms: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"queries": self.queries, "mode": self.mode, "extraction": self.extraction,
                "modifiers": self.modifiers.model_dump(), "evidence": self.evidence,
                "hits": [h.to_dict() for h in self.hits], "latency_ms": self.latency_ms,
                "error": self.error}

    def apply_to_world(self, world: Any) -> Dict[str, Any]:
        """Push the modifiers into the physics engine; returns intel + before/after physics."""
        return {"intel": self.to_dict(), "physics": world.apply_hazard_modifiers(self.modifiers.model_dump())}


# --------------------------------------------------------------------------
# Deterministic extraction (regex + unit conversion)
# --------------------------------------------------------------------------
_NUM = r"(\d+(?:\.\d+)?)"
_WIND_UNITS = r"(km/h|kph|kmh|mph|m/s|mps|knots?|kts?)"
_RE_WIND_A = re.compile(rf"(?:winds?|gusts?)[^.\n]{{0,60}}?{_NUM}\s?{_WIND_UNITS}\b", re.I)
_RE_WIND_B = re.compile(rf"{_NUM}\s?{_WIND_UNITS}[^.\n]{{0,30}}?(?:winds?|gusts?)", re.I)
_RE_TEMP = re.compile(r"(-?\d+(?:\.\d+)?)\s?(?:°|º|degrees?)\s?(?:of\s)?([CF]|celsius|fahrenheit)\b", re.I)
_RE_PCT_BEFORE = re.compile(rf"{_NUM}\s?%[^.\n]{{0,80}}?aftershock", re.I)
_RE_PCT_AFTER = re.compile(rf"aftershock[^.\n]{{0,80}}?{_NUM}\s?%", re.I)
_RE_MAG = re.compile(rf"(?:magnitude|\bM)\s?{_NUM}", re.I)
_RE_AQI = re.compile(r"\bAQI\b[^0-9]{0,12}(\d{2,3})", re.I)
_SMOKE_WORDS = ("smoke", "fire", "gas leak", "toxic", "fumes", "dust", "asbestos", "chemical")


def _to_mps(value: float, unit: str) -> float:
    u = unit.lower()
    if u in ("km/h", "kph", "kmh"):
        return value / 3.6
    if u == "mph":
        return value * 0.44704
    if u.startswith("kn") or u.startswith("kt"):
        return value * 0.514444
    return value


def _note(text: str, start: int, end: int, width: int = 50) -> str:
    lo = max(0, start - width)
    s = re.sub(r"\s+", " ", text[lo:end + width]).strip()
    if lo > 0 and " " in s:
        s = s.split(" ", 1)[1]                      # drop a partial leading word
    return s[:140]


def extract_fields(text: str) -> Dict[str, Tuple[float, str]]:
    """Return {field: (value, evidence)} for whatever the text supports. Never guesses missing fields."""
    out: Dict[str, Tuple[float, str]] = {}
    low = text.lower()

    winds: List[Tuple[float, str]] = []
    for rx in (_RE_WIND_A, _RE_WIND_B):
        for m in rx.finditer(text):
            winds.append((_to_mps(float(m.group(1)), m.group(2)), _note(text, m.start(), m.end())))
    if winds:
        out["wind_turbulence_mps"] = (statistics.mean(v for v, _ in winds), winds[0][1])

    temps: List[Tuple[float, str]] = []
    for m in _RE_TEMP.finditer(text):
        v = float(m.group(1))
        c = (v - 32.0) * 5.0 / 9.0 if m.group(2).lower().startswith("f") else v
        if -30.0 <= c <= 55.0:
            temps.append((c, _note(text, m.start(), m.end())))
    if temps:
        out["ambient_temp_c"] = (statistics.median(v for v, _ in temps), temps[0][1])

    pcts = [(float(m.group(1)) / 100.0, _note(text, m.start(), m.end()))
            for rx in (_RE_PCT_BEFORE, _RE_PCT_AFTER) for m in rx.finditer(text) if 0 <= float(m.group(1)) <= 100]
    if pcts:
        best = max(pcts, key=lambda p: p[0])
        out["seismic_aftershock_prob"] = (best[0], best[1])
    elif any(w in low for w in ("earthquake", "quake", "seismic")):
        mags = [(float(m.group(1)), m) for m in _RE_MAG.finditer(text) if 4.0 <= float(m.group(1)) <= 9.5]
        if mags and "aftershock" in low:
            mag, m = max(mags, key=lambda t: t[0])
            p = min(max(0.04 + 0.07 * (mag - 5.0), 0.04), 0.5)      # heuristic: larger mainshock, more aftershocks
            out["seismic_aftershock_prob"] = (p, f"magnitude-based heuristic from M{mag:g}: " + _note(text, m.start(), m.end(), 30))

    aqi = _RE_AQI.search(text)
    if aqi:
        out["smoke_toxicity_index"] = (min(float(aqi.group(1)) / 300.0, 1.0), _note(text, aqi.start(), aqi.end()))
    else:
        hits = [w for w in _SMOKE_WORDS if w in low]
        if hits:
            out["smoke_toxicity_index"] = (min(0.25 + 0.12 * len(hits), 0.95), "keywords: " + ", ".join(hits))

    margin = None
    if any(k in low for k in ("high-rise", "high rise", "tower", "unreinforced masonry")):
        margin = 3.0
    elif any(k in low for k in ("multi-story", "multistory", "multi-storey", "apartment", "storey", "story building")):
        margin = 2.5
    elif "reinforced concrete" in low:
        margin = 2.0
    if margin is not None:
        out["fema_safety_margin_m"] = (margin, "structure-type heuristic (collapse-zone stand-off)")
    return out


def aggregate_fields(per_hit: List[Tuple[SearchHit, Dict[str, Tuple[float, str]]]]
                     ) -> Tuple[Dict[str, float], Dict[str, str], List[str]]:
    values: Dict[str, float] = {}
    evidence: Dict[str, str] = {}
    used_urls: List[str] = []
    for name in NUMERIC_FIELDS:
        cands = [(h, f[name]) for h, f in per_hit if name in f]
        if not cands:
            continue
        nums = [c[1][0] for c in cands]
        if name in ("seismic_aftershock_prob", "smoke_toxicity_index", "fema_safety_margin_m"):
            v = max(nums)
        elif name == "ambient_temp_c":
            v = statistics.median(nums)
        else:
            v = statistics.mean(nums)
        values[name] = v
        top = max(cands, key=lambda c: c[1][0]) if name != "wind_turbulence_mps" else cands[0]
        evidence[name] = f"{top[1][1]}  [{top[0].url}]"
        for h, _ in cands:
            if h.url not in used_urls:
                used_urls.append(h.url)
    return values, evidence, used_urls


# --------------------------------------------------------------------------
# LLM JSON helpers
# --------------------------------------------------------------------------
def build_extraction_prompt(hits: List[SearchHit], max_chars: int = 700) -> str:
    blocks = [f"[{i + 1}] {h.title}\n{h.url}\n{h.content[:max_chars]}" for i, h in enumerate(hits)]
    return "Snippets:\n\n" + "\n\n".join(blocks) + "\n\nReturn the JSON object now."


def parse_llm_json(raw: Union[str, Dict[str, Any], None]) -> Dict[str, float]:
    """Parse and validate an LLM reply into numeric modifier fields. Raises ValueError on garbage."""
    if raw is None:
        raise ValueError("empty reply")
    obj = raw
    if isinstance(raw, str):
        txt = re.sub(r"```(?:json)?", "", raw).strip()
        a, b = txt.find("{"), txt.rfind("}")
        if a < 0 or b <= a:
            raise ValueError("no JSON object in reply")
        obj = json.loads(txt[a:b + 1])
    if not isinstance(obj, dict):
        raise ValueError("reply is not an object")
    picked = {k: obj[k] for k in NUMERIC_FIELDS if k in obj and obj[k] is not None}
    if not picked:
        raise ValueError("no recognised keys")
    checked = PhysicalHazardModifiers(**picked)          # validates + clamps
    return {k: getattr(checked, k) for k in picked}


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
class TavilyHazardIntel:
    def __init__(self, client: Any = None, llm_extractor: Optional[LLMExtractor] = None) -> None:
        self._client = client
        self.llm_extractor = llm_extractor
        self.last: Optional[HazardIntel] = None

    @staticmethod
    def build_queries(location: Optional[str] = None, query: Optional[str] = None) -> List[str]:
        if query:
            return [query]
        if location:
            loc = location.strip()
            return [f"{loc} earthquake aftershock structural collapse USAR hazard",
                    f"{loc} weather wind speed temperature today"]
        return [cfg.TAVILY_DEFAULT_QUERY]

    def _get_client(self) -> Any:
        if self._client is None:
            key = cfg.get_tavily_key()
            if key is None:
                return None
            from tavily import TavilyClient            # imported lazily so offline runs need no package
            self._client = TavilyClient(api_key=key)
        return self._client

    def offline(self, queries: List[str], error: Optional[str] = None) -> HazardIntel:
        mods = PhysicalHazardModifiers(**cfg.OFFLINE_HAZARD_PROFILE)
        intel = HazardIntel(queries=queries, mode="offline", extraction="offline_profile", modifiers=mods,
                            evidence={k: "cached offline profile (no live search performed)" for k in NUMERIC_FIELDS},
                            latency_ms={"search_ms": 0.0, "extract_ms": 0.0}, error=error)
        self.last = intel
        return intel

    def fetch(self, location: Optional[str] = None, query: Optional[str] = None) -> HazardIntel:
        queries = self.build_queries(location, query)
        try:
            client = self._get_client()
        except Exception as exc:                        # missing package, bad key format, ...
            log.warning("Tavily client unavailable: %s", exc)
            return self.offline(queries, error=f"client init failed: {exc}")
        if client is None:
            return self.offline(queries)

        t0 = time.perf_counter()
        hits: List[SearchHit] = []
        seen = set()
        per_query = max(2, cfg.TAVILY_MAX_RESULTS if len(queries) == 1 else (cfg.TAVILY_MAX_RESULTS + 1) // 2 + 1)
        try:
            for q in queries:
                resp = client.search(query=q, search_depth=cfg.TAVILY_SEARCH_DEPTH,
                                     max_results=per_query, include_answer=False)
                for r in (resp or {}).get("results", []):
                    url = str(r.get("url", ""))
                    if url and url not in seen:
                        seen.add(url)
                        hits.append(SearchHit(str(r.get("title", "")), url, str(r.get("content", "")),
                                              float(r.get("score", 0.0) or 0.0)))
        except Exception as exc:
            log.warning("Tavily search failed: %s", exc)
            return self.offline(queries, error=f"search failed: {exc}")
        search_ms = (time.perf_counter() - t0) * 1000.0
        if not hits:
            return self.offline(queries, error="search returned no results")

        t1 = time.perf_counter()
        per_hit = [(h, extract_fields(f"{h.title}. {h.content}")) for h in hits]
        h_vals, evidence, used = aggregate_fields(per_hit)

        llm_vals: Dict[str, float] = {}
        method = "heuristic"
        if self.llm_extractor is not None:
            try:
                llm_vals = parse_llm_json(self.llm_extractor(build_extraction_prompt(hits)))
                method = "nemotron+heuristic"
            except Exception as exc:
                log.warning("LLM extraction failed (%s); using heuristic only", exc)
        merged = {**h_vals, **llm_vals}
        for k in llm_vals:
            evidence[k] = "Nemotron extraction from retrieved snippets"
        if not merged:
            intel = self.offline(queries, error="snippets contained no extractable hazard values")
            intel.hits = hits
            return intel

        urls = used or [h.url for h in hits[:3]]
        mods = PhysicalHazardModifiers(**{**{k: cfg.DEFAULT_HAZARD_MODIFIERS[k] for k in NUMERIC_FIELDS},
                                          **merged, "source_urls": urls})
        for k in NUMERIC_FIELDS:
            evidence.setdefault(k, "not found in sources; default value used")
        intel = HazardIntel(queries=queries, mode="live", extraction=method, modifiers=mods, hits=hits,
                            evidence=evidence,
                            latency_ms={"search_ms": round(search_ms, 1),
                                        "extract_ms": round((time.perf_counter() - t1) * 1000.0, 1)})
        self.last = intel
        return intel