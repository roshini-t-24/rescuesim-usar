"""Run one mission for one policy with the offline planner and print its metrics as JSON.

Used by the dashboard's "compare policies on this seed" button.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["greedy", "static", "adaptive"])
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--survivors", type=int, default=5)
    ap.add_argument("--hazards-json", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    from nebius_jobs.batch_policy_eval import _force_offline

    _force_offline()
    from agents.mission_coordinator import run_mission

    hazards = json.loads(args.hazards_json) if args.hazards_json else None
    metrics = run_mission(
        policy=args.policy, seed=args.seed, hazards=hazards,
        num_survivors=args.survivors, max_steps=args.max_steps,
    )
    print("RESULT_JSON:" + json.dumps(metrics, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
