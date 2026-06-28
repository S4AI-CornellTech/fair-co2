#!/usr/bin/env python3
"""Attribute one server's embodied carbon across a schedule of co-located jobs, three ways.

For each job it computes a share under:
  RUP       proportional to CPU x runtime (the common industry split)
  Shapley   the exact fair share: a job's average marginal contribution to the peak demand the
            server was sized for (cost grows exponentially with the number of jobs)
  Fair-CO2  a cheap approximation of the Shapley share (cost grows linearly)

The three methods reproduce Fair-CO2's own code (baseline_attribution / ground_truth_shapley_attribution
/ temporal_shapley in monte-carlo-simulations/dynamic-demand/dynamic_demand_sim.py), built on the
hierarchical Shapley in forecast/emb_shapley_lib.py.

  python tutorial.py                               # the default schedule (exercises/workloads.json)
  python tutorial.py --workloads my_schedule.json  # your own schedule
  python tutorial.py --budget 2000                 # override the shared budget (kg)
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "forecast"))  # Fair-CO2's emb_shapley_lib
from emb_shapley_lib import shapley_attribution as h_shapley  # noqa: E402


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else HERE / p


def token(name):
    return name.split()[0].strip("(),").lower()


def demand_series(jobs, time):
    """Concurrent demand (cores) in each time slot."""
    d = [0] * time
    for t in range(time):
        for j in jobs:
            if j["start"] <= t < j["start"] + j["runtime"]:
                d[t] += j["cpu"]
    return d


def rup_attribution(jobs):
    """RUP: proportional to CPU x runtime."""
    raw = [j["cpu"] * j["runtime"] for j in jobs]
    s = sum(raw)
    return [x / s for x in raw] if s else [0] * len(jobs)


def shapley_exact(jobs, time):
    """Exact Shapley: a job's average marginal contribution to the peak of concurrent demand, over
    all coalitions of jobs."""
    labels = [j["name"] for j in jobs]
    by = {j["name"]: j for j in jobs}
    n = len(labels)
    combos = [set(c) for k in range(n + 1) for c in itertools.combinations(labels, k)]

    def peak(coal):
        return max(demand_series([by[l] for l in coal], time)) if coal else 0

    out = []
    for lab in labels:
        sv = 0.0
        for c in combos:
            if lab in c:
                without = c - {lab}
                sv += (peak(c) - peak(without)) / math.comb(n - 1, len(without))
        out.append(sv)
    s = sum(out)
    return [x / s for x in out] if s else [0] * n


def fairco2_temporal(jobs, time):
    """Fair-CO2's approximation: a hierarchical Shapley over the demand time-series gives a per-slot
    intensity; each job pays that intensity times its CPU over the slots it is active."""
    dem = demand_series(jobs, time)
    df = pd.DataFrame({"time": list(range(time)), "demand": dem})
    _shap, _peaks, ci_list, _rt = h_shapley(df, "time", "demand", [1],
                                            attribution_total=1, sampling_interval=1, offset=0)
    ci = ci_list[-1]
    out = []
    for j in jobs:
        a = sum(ci[t] * j["cpu"] for t in range(j["start"], min(j["start"] + j["runtime"], time)))
        out.append(a)
    s = sum(out)
    return [x / s for x in out] if s else [0] * len(jobs)


def attribution_chart(out_png, jobs, rup, shap, fco, budget):
    """Grouped bar chart of the three attributions (headless / Agg)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = [token(j["name"]) for j in jobs]
    series = {"RUP": [r * budget for r in rup],
              "Shapley": [s * budget for s in shap],
              "Fair-CO2": [f * budget for f in fco]}
    colors = ["#bb5566", "#3b7a57", "#ddaa33"]
    x = np.arange(len(groups))
    n = len(series)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
    for i, (name, vals) in enumerate(series.items()):
        bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=name, color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("attributed embodied carbon (kgCO2e)")
    ax.set_title("Embodied carbon attributed to each job")
    ax.legend(frameon=False, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description="Attribute a server's embodied carbon across co-located jobs")
    ap.add_argument("--workloads", default="exercises/workloads.json",
                    help="schedule JSON: {time, budget_kg, jobs:[{name,cpu,runtime,start}]}")
    ap.add_argument("--budget", type=float, default=None, help="override the shared budget in kg")
    ap.add_argument("--fig", action="store_true", help="also write the attribution chart to figures/")
    ap.add_argument("--expect", action="append", default=[], metavar="KEY=VAL",
                    help="assert a value: <job>_rup|_shapley|_fairco2 (repeatable; used by CI)")
    args = ap.parse_args()

    cfg = json.loads(_resolve(args.workloads).read_text())
    jobs, time = cfg["jobs"], int(cfg["time"])
    budget = args.budget if args.budget is not None else float(cfg.get("budget_kg", 1523.1))

    rup = rup_attribution(jobs)
    shap = shapley_exact(jobs, time)
    fco = fairco2_temporal(jobs, time)
    dem = demand_series(jobs, time)
    peak_t = max(range(time), key=lambda t: dem[t])

    results = {}
    print(f"Splitting {budget:,.1f} kgCO2e across {len(jobs)} co-located jobs over {time} time slots.")
    print(f"Concurrent demand peaks at {max(dem)} cores (slot {peak_t}).\n")
    print(f"  {'job':<24}{'RUP':>11}{'Shapley':>11}{'Fair-CO2':>11}{'RUP err':>9}{'Fair-CO2 err':>13}")
    for i, j in enumerate(jobs):
        r, s, f = rup[i] * budget, shap[i] * budget, fco[i] * budget
        rup_err = abs(r - s) / s * 100 if s else 0.0
        f_err = abs(f - s) / s * 100 if s else 0.0
        tok = token(j["name"])
        results[f"{tok}_rup"] = round(r, 1)
        results[f"{tok}_shapley"] = round(s, 1)
        results[f"{tok}_fairco2"] = round(f, 1)
        print(f"  {j['name']:<24}{r:>9,.1f}kg{s:>9,.1f}kg{f:>9,.1f}kg{rup_err:>8.0f}%{f_err:>12.0f}%")
    print("\nRUP charges by CPU x runtime. Shapley charges by a job's contribution to the peak the")
    print("server was sized for. Fair-CO2 approximates the Shapley share cheaply enough to bill per job.")
    print("The two error columns are each method's distance from the fair Shapley share.")

    if args.fig:
        attribution_chart(HERE / "figures" / "attribution.png", jobs, rup, shap, fco, budget)

    failures = []
    for spec in args.expect:
        key, _, raw = spec.partition("=")
        key = key.strip()
        got = results.get(key)
        if got is None:
            failures.append(f"{key}: not computed")
            continue
        want = float(raw)
        if abs(got - want) > max(0.2, 0.01 * abs(want)):
            failures.append(f"{key}: got {got}, expected {want}")
    if failures:
        for msg in failures:
            print(f"  EXPECT FAIL: {msg}", file=sys.stderr)
        sys.exit(1)
    if args.expect:
        print("  EXPECT OK")


if __name__ == "__main__":
    main()
