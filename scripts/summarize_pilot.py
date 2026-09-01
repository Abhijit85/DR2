#!/usr/bin/env python3
"""Summarize a 3-arm HotpotQA pilot and emit trigger calibration scores."""
from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    out = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def pct(numer: int, denom: int) -> float:
    return 0.0 if denom == 0 else 100.0 * numer / denom


def mean(xs: list[float]) -> float:
    return 0.0 if not xs else float(sum(xs) / len(xs))


def median(xs: list[float]) -> float:
    return 0.0 if not xs else float(stats.median(xs))


def arm_stats(rows: list[dict]) -> dict:
    n = len(rows)
    em = mean([float(r.get("em", 0.0)) for r in rows])
    f1 = mean([float(r.get("f1", 0.0)) for r in rows])
    wall = median([float(r.get("wall_s", 0.0)) for r in rows])
    n_retr = [float(r.get("n_retrievals", 0.0)) for r in rows]
    finish = sum(r.get("terminated_by") == "finish" for r in rows)
    gap_q = 0
    rollback_q = 0
    cap_hits = 0
    all_cycles = 0
    for r in rows:
        cycles = r.get("cycles", [])
        all_cycles += len(cycles)
        gap_q += int(any(bool(c.get("gap_fire")) for c in cycles))
        rollback_q += int(any(int(c.get("rollback_n", 0)) > 0 for c in cycles))
        for c in cycles:
            cap_hits += int(bool(c.get("rollback_capped")))
    return {
        "n": n,
        "em": em,
        "f1": f1,
        "finish_pct": pct(finish, n),
        "gap_fire_pct": pct(gap_q, n),
        "rollback_q": 0.0 if n == 0 else rollback_q / n,
        "rollback_cap_hit_pct": pct(cap_hits, max(all_cycles, 1)),
        "retrievals_per_q": mean(n_retr),
        "median_wall_s": wall,
    }


def collect_null_scores(rows: list[dict]) -> list[float]:
    scores = []
    for r in rows:
        em = float(r.get("em", 0.0))
        f1 = float(r.get("f1", 0.0))
        # Keep only clearly-correct or strong-partial answers in the null set.
        if not (em == 1.0 or f1 >= 0.6):
            continue
        for c in r.get("cycles", []):
            if bool(c.get("gap_fire")):
                continue
            for s in c.get("gap_all_scores", []):
                scores.append(float(s))
    return scores


def format_arm(label: str, s: dict) -> str:
    return (
        f"{label:>14}  n={s['n']:3d}  EM={s['em']:.3f}  F1={s['f1']:.3f}  "
        f"gap={s['gap_fire_pct']:.1f}%  rollback/q={s['rollback_q']:.2f}  "
        f"cap_hit={s['rollback_cap_hit_pct']:.1f}%  finish={s['finish_pct']:.1f}%  "
        f"retr/q={s['retrievals_per_q']:.2f}  med_wall={s['median_wall_s']:.1f}s"
    )


def health_messages(s: dict) -> list[str]:
    msgs = []
    gap = s["gap_fire_pct"]
    if gap == 0.0:
        msgs.append("gap trigger dead: 0% of questions fired")
    elif gap >= 99.0:
        msgs.append("gap trigger likely miscalibrated: ~100% of questions fired")
    elif not (5.0 <= gap <= 85.0):
        msgs.append(f"gap trigger outside target band: {gap:.1f}%")
    if s["rollback_q"] <= 0.0:
        msgs.append("rollback inactive: 0 questions had any rollback")
    if s["rollback_cap_hit_pct"] >= 30.0:
        msgs.append(f"rollback cap hit too often: {s['rollback_cap_hit_pct']:.1f}%")
    if s["finish_pct"] <= 50.0:
        msgs.append(f"finish termination low: {s['finish_pct']:.1f}%")
    if not (0.5 <= s["retrievals_per_q"] <= 3.5):
        msgs.append(f"retrievals/q outside target band: {s['retrievals_per_q']:.2f}")
    if s["median_wall_s"] >= 180.0:
        msgs.append(f"median wall clock high: {s['median_wall_s']:.1f}s")
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_a")
    ap.add_argument("arm_b")
    ap.add_argument("arm_c")
    ap.add_argument("--calibration-out", required=True)
    args = ap.parse_args()

    rows_a = load_jsonl(args.arm_a)
    rows_b = load_jsonl(args.arm_b)
    rows_c = load_jsonl(args.arm_c)
    stats_a = arm_stats(rows_a)
    stats_b = arm_stats(rows_b)
    stats_c = arm_stats(rows_c)

    print("Pilot Arm Table")
    print(format_arm("Arm A", stats_a))
    print(format_arm("Arm B", stats_b))
    print(format_arm("Arm C", stats_c))

    print("\nHealth Checks")
    msgs = health_messages(stats_c)
    if msgs:
        for msg in msgs:
            print(f"- {msg}")
    else:
        print("- all Arm C health targets passed")

    delta = stats_c["f1"] - max(stats_a["f1"], stats_b["f1"])
    print("\nHypothesis Verdict")
    print(f"C - max(A,B) F1 = {delta:.3f}")

    null_scores = collect_null_scores(rows_c)
    cal_out = Path(args.calibration_out)
    cal_out.parent.mkdir(parents=True, exist_ok=True)
    cal_out.write_text(json.dumps({
        "null_scores": null_scores,
        "source": str(Path(args.arm_c)),
        "n_null_scores": len(null_scores),
    }, indent=2))
    print("\nCalibration")
    print(f"wrote {len(null_scores)} null scores to {cal_out}")


if __name__ == "__main__":
    main()
