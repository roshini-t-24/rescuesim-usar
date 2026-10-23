"""Headless Monte Carlo policy evaluator for Nebius Serverless Jobs.

Usage:
    python -m nebius_jobs.batch_policy_eval --episodes 100 --output results/serverless_eval.json
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

POLICIES = ("greedy", "static", "adaptive")
POLICY_LABELS = {
    "greedy": "Baseline Greedy Navigation",
    "static": "Static Conservative A*",
    "adaptive": "Nemotron + Cosmos Adaptive Policy",
}
_KEY_VARS = ("NEBIUS_API_KEY", "TAVILY_API_KEY")


# --------------------------------------------------------------------------- #
# Episode execution (runs inside worker processes)
# --------------------------------------------------------------------------- #
def _to_dict(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return dict(result)
    if isinstance(result, (tuple, list)):
        for item in result:
            try:
                return _to_dict(item)
            except TypeError:
                continue
        raise TypeError("run_mission returned a sequence without a metrics object")
    for attr in ("metrics", "to_dict", "model_dump", "dict"):
        obj = getattr(result, attr, None)
        if obj is None:
            continue
        val = obj() if callable(obj) else obj
        if isinstance(val, dict):
            return dict(val)
    if hasattr(result, "__dict__"):
        return dict(vars(result))
    raise TypeError(f"Cannot convert {type(result)!r} to a metrics dict")


def _sample_hazards(mc: Any, seed: int) -> Any:
    fn = getattr(mc, "sample_hazards", None)
    if fn is None:
        cls = getattr(mc, "MissionCoordinator", None)
        fn = getattr(cls, "sample_hazards", None) if cls is not None else None
    if fn is None:
        return None
    params = [p for p in inspect.signature(fn).parameters.values() if p.name not in ("self", "cls")]
    if not params:
        return fn()
    first = params[0].name.lower()
    arg = np.random.default_rng(seed) if "rng" in first or "random" in first else seed
    return fn(arg)


def _invoke_run_mission(mc: Any, policy: str, seed: int) -> Dict[str, Any]:
    fn = getattr(mc, "run_mission", None)
    owner = None
    if fn is None:
        owner = mc.MissionCoordinator
        fn = owner.run_mission
    sig = inspect.signature(fn)
    names = set(sig.parameters)
    kwargs: Dict[str, Any] = {}
    if "policy" in names:
        kwargs["policy"] = policy
    if "seed" in names:
        kwargs["seed"] = seed
    for hz_name in ("hazards", "hazard_modifiers", "modifiers"):
        if hz_name in names:
            kwargs[hz_name] = _sample_hazards(mc, seed)
            break
    for flag in ("verbose", "record", "record_frames", "keep_frames", "log"):
        if flag in names:
            kwargs[flag] = False
    if owner is not None:
        init_names = set(inspect.signature(owner.__init__).parameters)
        init_kwargs = {k: v for k, v in (("policy", policy), ("seed", seed)) if k in init_names}
        return _to_dict(getattr(owner(**init_kwargs), "run_mission")(**{k: v for k, v in kwargs.items() if k not in init_kwargs}))
    return _to_dict(fn(**kwargs))


def _force_offline() -> None:
    for key in _KEY_VARS:
        os.environ[key] = ""
    from config import settings as _cfg
    _cfg.get_nebius_key = lambda: None
    _cfg.get_tavily_key = lambda: None
    _cfg.nebius_online = lambda: False
    _cfg.tavily_online = lambda: False


def run_episode(task: Dict[str, Any]) -> Dict[str, Any]:
    policy, seed, live = task["policy"], task["seed"], task["live"]
    if not live:
        _force_offline()
    t0 = time.perf_counter()
    try:
        mc = importlib.import_module("agents.mission_coordinator")
        metrics = _invoke_run_mission(mc, policy, seed)
        metrics["policy"] = policy
        metrics["seed"] = seed
        metrics["wall_s"] = round(time.perf_counter() - t0, 3)
        metrics["error"] = None
        return metrics
    except Exception as exc:  # noqa: BLE001 - recorded per episode
        return {
            "policy": policy,
            "seed": seed,
            "wall_s": round(time.perf_counter() - t0, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _rescue_pct(ep: Dict[str, Any]) -> Optional[float]:
    total = _num(ep.get("survivors_total"))
    rescued = _num(ep.get("survivors_rescued"))
    if total and rescued is not None:
        return 100.0 * rescued / total
    rate = _num(ep.get("rescue_rate"))
    if rate is None:
        return None
    return rate * 100.0 if rate <= 1.0 else rate


def _bootstrap_ci(values: Sequence[float], seed: int = 7, n_boot: int = 2000) -> List[Optional[float]]:
    if len(values) < 2:
        return [None, None]
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _wilson_ci(k: int, n: int, z: float = 1.96) -> List[Optional[float]]:
    if n == 0:
        return [None, None]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [100.0 * max(0.0, centre - half), 100.0 * min(1.0, centre + half)]


def summarize(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [e for e in episodes if not e.get("error")]
    n = len(ok)
    out: Dict[str, Any] = {"episodes": n, "failed_episodes": len(episodes) - n}
    if n == 0:
        return out

    rescue = [r for r in (_rescue_pct(e) for e in ok) if r is not None]
    losses = [bool(e.get("robot_loss")) for e in ok]
    rover_lost = sum(bool(e.get("rover_lost")) for e in ok)
    drone_lost = sum(bool(e.get("drone_lost")) for e in ok)
    ttfs = [t for t in (_num(e.get("first_located_s")) for e in ok) if t is not None]
    ttfr = [t for t in (_num(e.get("first_rescued_s")) for e in ok) if t is not None]
    energy = [x for x in (_num(e.get("energy_wh")) for e in ok) if x is not None]
    rescued_total = sum(int(_num(e.get("survivors_rescued")) or 0) for e in ok)
    risk = [x for x in (_num(e.get("rover_risk_exposure")) for e in ok) if x is not None]

    out.update(
        {
            "rescue_rate_pct": {
                "mean": float(np.mean(rescue)) if rescue else None,
                "std": float(np.std(rescue, ddof=1)) if len(rescue) > 1 else 0.0,
                "ci95": _bootstrap_ci(rescue),
            },
            "robot_loss_rate_pct": {
                "mean": 100.0 * sum(losses) / n,
                "ci95": _wilson_ci(sum(losses), n),
                "rover_lost_pct": 100.0 * rover_lost / n,
                "drone_lost_pct": 100.0 * drone_lost / n,
            },
            "time_to_first_survivor_s": {
                "mean": float(np.mean(ttfs)) if ttfs else None,
                "median": float(np.median(ttfs)) if ttfs else None,
                "ci95": _bootstrap_ci(ttfs),
                "episodes_with_detection": len(ttfs),
                "episodes_without_detection": n - len(ttfs),
            },
            "time_to_first_rescue_s": {
                "mean": float(np.mean(ttfr)) if ttfr else None,
                "episodes_with_rescue": len(ttfr),
            },
            "energy_efficiency_wh_per_rescued": (sum(energy) / rescued_total) if rescued_total and energy else None,
            "energy_wh_per_episode_mean": float(np.mean(energy)) if energy else None,
            "survivors_rescued_total": rescued_total,
            "rover_risk_exposure_mean": float(np.mean(risk)) if risk else None,
            "wall_s_mean": float(np.mean([e.get("wall_s", 0.0) for e in ok])),
        }
    )
    return out


def paired_tests(by_policy: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    try:
        from scipy.stats import wilcoxon
    except Exception:  # noqa: BLE001
        return {}
    results: Dict[str, Any] = {}
    adaptive = {e["seed"]: e for e in by_policy.get("adaptive", []) if not e.get("error")}
    for other in ("greedy", "static"):
        base = {e["seed"]: e for e in by_policy.get(other, []) if not e.get("error")}
        seeds = sorted(set(adaptive) & set(base))
        entry: Dict[str, Any] = {"paired_episodes": len(seeds)}
        for name, fn in (
            ("rescue_rate_pct", _rescue_pct),
            ("robot_loss", lambda e: 1.0 if e.get("robot_loss") else 0.0),
        ):
            a = [fn(adaptive[s]) for s in seeds]
            b = [fn(base[s]) for s in seeds]
            pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            if len(pairs) < 5 or all(x == y for x, y in pairs):
                entry[name] = {"p_value": None}
                continue
            stat, p = wilcoxon([x for x, _ in pairs], [y for _, y in pairs])
            entry[name] = {"statistic": float(stat), "p_value": float(p)}
        results[f"adaptive_vs_{other}"] = entry
    return results


def compare(summary: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    comp: Dict[str, Any] = {}
    adp = summary.get("adaptive", {})
    for other in ("greedy", "static"):
        base = summary.get(other, {})
        if not adp.get("rescue_rate_pct") or not base.get("rescue_rate_pct"):
            continue
        a_loss = adp["robot_loss_rate_pct"]["mean"]
        b_loss = base["robot_loss_rate_pct"]["mean"]
        reduction = 100.0 * (b_loss - a_loss) / b_loss if b_loss > 0 else None
        a_res = adp["rescue_rate_pct"]["mean"]
        b_res = base["rescue_rate_pct"]["mean"]
        a_t = adp["time_to_first_survivor_s"]["mean"]
        b_t = base["time_to_first_survivor_s"]["mean"]
        a_e = adp["energy_efficiency_wh_per_rescued"]
        b_e = base["energy_efficiency_wh_per_rescued"]
        comp[f"adaptive_vs_{other}"] = {
            "robot_loss_reduction_pct": reduction,
            "meets_60pct_loss_reduction_target": bool(reduction is not None and reduction > 60.0),
            "rescue_rate_delta_pts": (a_res - b_res) if a_res is not None and b_res is not None else None,
            "time_to_first_survivor_delta_s": (a_t - b_t) if a_t is not None and b_t is not None else None,
            "energy_per_rescued_delta_wh": (a_e - b_e) if a_e is not None and b_e is not None else None,
        }
    return comp


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate(
    episodes: int,
    policies: Sequence[str] = POLICIES,
    seed_base: int = 1000,
    workers: int = 1,
    live: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    if not live:
        _force_offline()
    tasks = [
        {"policy": p, "seed": seed_base + i, "live": live}
        for i in range(episodes)
        for p in policies
    ]
    t0 = time.perf_counter()
    results: List[Dict[str, Any]] = []
    done = 0

    def _tick(res: Dict[str, Any]) -> None:
        nonlocal done
        results.append(res)
        done += 1
        if progress and (done % max(1, len(tasks) // 20) == 0 or done == len(tasks)):
            print(f"[{done}/{len(tasks)}] episodes complete ({time.perf_counter() - t0:.1f}s)", flush=True)

    if workers <= 1:
        for task in tasks:
            _tick(run_episode(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_episode, t) for t in tasks]
            for fut in as_completed(futures):
                _tick(fut.result())

    by_policy: Dict[str, List[Dict[str, Any]]] = {p: [] for p in policies}
    for res in results:
        by_policy[res["policy"]].append(res)
    for eps in by_policy.values():
        eps.sort(key=lambda e: e["seed"])

    summary = {p: summarize(by_policy[p]) for p in policies}
    return {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "episodes_per_policy": episodes,
            "policies": list(policies),
            "policy_labels": {p: POLICY_LABELS[p] for p in policies},
            "seed_base": seed_base,
            "paired_seeds": True,
            "llm_mode": "live" if live else "offline-heuristic",
            "workers": workers,
            "duration_s": round(time.perf_counter() - t0, 2),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "summary": summary,
        "comparison": compare(summary),
        "paired_tests": paired_tests(by_policy) if "adaptive" in policies else {},
        "episodes": [e for p in policies for e in by_policy[p]],
    }


def _print_table(report: Dict[str, Any]) -> None:
    header = f"{'policy':<10} {'rescue %':>9} {'loss %':>8} {'TTFS s':>9} {'Wh/rescued':>11} {'failed':>7}"
    print("\n" + header)
    print("-" * len(header))
    for pol, s in report["summary"].items():
        if "rescue_rate_pct" not in s:
            print(f"{pol:<10} {'n/a':>9}")
            continue

        def f(v: Optional[float], w: int, d: int = 1) -> str:
            return f"{v:>{w}.{d}f}" if v is not None else f"{'n/a':>{w}}"

        print(
            f"{pol:<10} {f(s['rescue_rate_pct']['mean'], 9)} {f(s['robot_loss_rate_pct']['mean'], 8)} "
            f"{f(s['time_to_first_survivor_s']['mean'], 9)} {f(s['energy_efficiency_wh_per_rescued'], 11, 2)} "
            f"{s['failed_episodes']:>7}"
        )
    for name, c in report["comparison"].items():
        red = c["robot_loss_reduction_pct"]
        print(f"{name}: loss reduction = {'n/a' if red is None else f'{red:.1f}%'}, "
              f"rescue delta = {c['rescue_rate_delta_pts']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="RescueSim 100-episode policy benchmark")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--output", type=str, default="results/serverless_eval.json")
    ap.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--live", action="store_true", help="use Nebius/Tavily keys from the environment")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.live:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

    report = evaluate(
        episodes=args.episodes,
        policies=args.policies,
        seed_base=args.seed_base,
        workers=args.workers,
        live=args.live,
        progress=not args.quiet,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    _print_table(report)
    print(f"\nWrote {out} ({report['meta']['duration_s']}s)")

    failed = sum(s.get("failed_episodes", 0) for s in report["summary"].values())
    total = args.episodes * len(args.policies)
    if failed:
        first = next(e["error"] for e in report["episodes"] if e.get("error"))
        print(f"WARNING: {failed}/{total} episodes failed; first error: {first}", file=sys.stderr)
    return 2 if failed == total else 0


if __name__ == "__main__":
    raise SystemExit(main())