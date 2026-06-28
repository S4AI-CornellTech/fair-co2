#!/usr/bin/env python3
"""Fair-CO2 tutorial helper: attribute a shared carbon budget across a SCHEDULE of co-located
jobs three ways — the industry-default proportional (RUP) split, the exact fair **Shapley**
ground truth, and **Fair-CO2's** cheap Shapley approximation — to show why RUP is wrong and how
Fair-CO2 fixes it. Self-contained in this repo.

The three attribution methods reproduce Fair-CO2's own algorithms in
`../monte-carlo-simulations/dynamic-demand/dynamic_demand_sim.py` (`baseline_attribution`,
`ground_truth_shapley_attribution`, `temporal_shapley_shapley_attribution`); the hierarchical
Shapley they build on is imported directly from `../forecast/emb_shapley_lib.py`. No Monte-Carlo,
no multiprocessing — one editable schedule. Only Fair-CO2's committed data is read.

  ./tutorial.sh --workloads exercises/workloads.json     RUP vs Shapley vs Fair-CO2 on a schedule
  ./tutorial.sh --swing llama                            "your bill depends on your neighbor"
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
from emb_shapley_lib import shapley_attribution as h_shapley  # noqa: E402  Fair-CO2's hierarchical Shapley

# Fair-CO2's committed embodied-carbon colocation matrix (read-only).
EMB_MATRIX = HERE.parent / "colocation" / "ref-results" / "embodied_cf_colocation_matrix.csv"


def grouped_bars(out_png, groups, series, *, title, subtitle, ylabel, colors,
                 figsize=(7.8, 4.8), unit="%", ann_fmt="{:.1f}", legend_loc="upper left"):
    """Annotated grouped bar chart (headless / Agg). One bold headline (suptitle), one gray
    sub-line (axes title), value labels on each bar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(series.keys())
    x = np.arange(len(groups))
    n = len(names)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ymax = 0.0
    for i, name in enumerate(names):
        vals = series[name]
        ymax = max(ymax, max(vals))
        bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=name,
                      color=colors[i % len(colors)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, ann_fmt.format(v) + unit,
                    ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in groups])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_title(subtitle, fontsize=9.5, color="#555555")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc=legend_loc)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else HERE / p


def default_budget():
    """Shared embodied budget = ACT's R740 from segment 1 (committed handoff), in kg."""
    p = HERE / "inputs" / "from_01_act.json"
    if p.exists():
        return float(json.loads(p.read_text())["server_embodied_kgco2e"])
    return 1523.1


def token(name):
    return name.split()[0].strip("(),").lower()


def demand_series(jobs, time):
    """Concurrent demand (cores) at each unit time slot — Fair-CO2 get_demand_from_schedule."""
    d = [0] * time
    for t in range(time):
        for j in jobs:
            if j["start"] <= t < j["start"] + j["runtime"]:
                d[t] += j["cpu"]
    return d


def rup_attribution(jobs):
    """RUP: proportional to CPU x runtime (Fair-CO2 baseline_attribution)."""
    raw = [j["cpu"] * j["runtime"] for j in jobs]
    s = sum(raw)
    return [x / s for x in raw] if s else [0] * len(jobs)


def shapley_exact(jobs, time):
    """Exact fair Shapley over the schedule: each job's avg marginal contribution to the peak of
    *concurrent* demand, all coalitions (Fair-CO2 ground_truth_shapley_attribution)."""
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
    """Fair-CO2's cheap approximation: a hierarchical Shapley over the demand *time-series* yields a
    per-slot intensity; each job pays that intensity x its CPU over its active slots
    (Fair-CO2 temporal_shapley_shapley_attribution). Linear in jobs, not exponential."""
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


def main():
    ap = argparse.ArgumentParser(description="Fair-CO2 hands-on: RUP vs Shapley vs Fair-CO2 on a schedule")
    ap.add_argument("--workloads", default="exercises/workloads.json",
                    help="JSON: {time, jobs:[{name,cpu,runtime,start}], budget_source?}")
    ap.add_argument("--budget", type=float, default=None, help="shared embodied budget in kg (overrides config/default)")
    ap.add_argument("--swing", metavar="WORKLOAD", help="show a job's attributed-carbon swing across co-location partners")
    ap.add_argument("--fig", action="store_true", help="render the attribution chart into figures/tutorial/")
    ap.add_argument("--expect", action="append", default=[], metavar="KEY=VAL",
                    help="assert a value: <jobtoken>_rup|_shapley|_fairco2, or swing (repeatable)")
    args = ap.parse_args()

    results = {}

    if args.swing:
        emb = pd.read_csv(EMB_MATRIX).set_index("workload")
        if args.swing not in emb.index:
            print(f"ERROR: '{args.swing}' not in matrix. Available: {', '.join(emb.index)}", file=sys.stderr)
            sys.exit(1)
        row = emb.loc[args.swing]
        lo, hi = float(row.min()), float(row.max())
        results["swing"] = round(hi / lo, 2)
        print(f"[Fair-CO2 tutorial] '{args.swing}' attributed embodied carbon across co-location partners:")
        print(f"    isolated (nothing):  {float(row['nothing']):.4f} gCO2e")
        print(f"    sharing with spark:  {float(row['spark']):.4f} gCO2e")
        print(f"    min {lo:.4f} (w/ {row.idxmin()})    max {hi:.4f} (w/ {row.idxmax()})")
        print(f"  -> the same job's attributed bill swings {results['swing']}x just from its neighbor")
        print("\n  the Fair-CO2 data this read (a committed matrix you load yourself):")
        print("      import pandas as pd")
        print('      pd.read_csv("colocation/ref-results/embodied_cf_colocation_matrix.csv").set_index("workload")')
    else:
        cfg = json.loads(_resolve(args.workloads).read_text())
        jobs, time = cfg["jobs"], int(cfg["time"])
        if args.budget is not None:
            budget = args.budget
        elif cfg.get("budget_source") == "from_01_act":
            budget = default_budget()
        else:
            budget = float(cfg.get("budget_kg", default_budget()))

        rup = rup_attribution(jobs)
        shap = shapley_exact(jobs, time)
        fco = fairco2_temporal(jobs, time)
        dem = demand_series(jobs, time)
        peak_t = max(range(time), key=lambda t: dem[t])

        print(f"[Fair-CO2 tutorial] one {time}-slot schedule, splitting {budget:,.1f} kgCO2e "
              f"across {len(jobs)} co-located jobs")
        print(f"  concurrent demand peaks at {max(dem)} cores (slot {peak_t})")
        print(f"  {'job':<22}{'RUP':>10}{'Shapley(fair)':>15}{'Fair-CO2':>11}   {'RUP err':>8}{'F-CO2 err':>10}")
        for i, j in enumerate(jobs):
            r, s, f = rup[i] * budget, shap[i] * budget, fco[i] * budget
            r_err = abs(r - s) / s * 100 if s else 0.0
            f_err = abs(f - s) / s * 100 if s else 0.0
            tok = token(j["name"])
            results[f"{tok}_rup"] = round(r, 1)
            results[f"{tok}_shapley"] = round(s, 1)
            results[f"{tok}_fairco2"] = round(f, 1)
            print(f"  {j['name']:<22}{r:>8,.1f}kg{s:>13,.1f}kg{f:>9,.1f}kg   {r_err:>6.0f}% {f_err:>8.0f}%")
        print("  -> RUP bills by CPU x runtime; Shapley charges by who drives the PEAK the hardware was")
        print("     built for. Fair-CO2 approximates the fair Shapley share cheaply enough to run live.")
        if args.fig:
            grouped_bars(
                HERE / "figures" / "tutorial" / "attribution.png",
                groups=[token(j["name"]) for j in jobs],
                series={"RUP": [round(rup[i] * budget, 1) for i in range(len(jobs))],
                        "Shapley (fair)": [round(shap[i] * budget, 1) for i in range(len(jobs))],
                        "Fair-CO2": [round(fco[i] * budget, 1) for i in range(len(jobs))]},
                title="RUP vs Shapley vs Fair-CO2",
                subtitle="RUP misattributes; Fair-CO2 approximates the fair Shapley share",
                ylabel="attributed embodied carbon (kgCO₂e)",
                colors=["#bb5566", "#3b7a57", "#ddaa33"], unit="", ann_fmt="{:.0f}",
            )
        print("\n  the Fair-CO2 API behind this (a library, not a CLI — call it yourself):")
        print("      from emb_shapley_lib import shapley_attribution   # ../forecast/emb_shapley_lib.py")
        print("      # RUP      = each job's cpu*runtime, normalized            (Fair-CO2 baseline_attribution)")
        print("      # Shapley  = avg marginal contribution to peak-of-concurrent-demand, all coalitions")
        print("      # Fair-CO2 = shapley_attribution(demand_series) -> per-slot CI; job pays CI*cpu over its active slots")

    failures = []
    for spec in args.expect:
        key, _, raw = spec.partition("=")
        key = key.strip()
        got = results.get(key)
        if got is None:
            failures.append(f"{key}: not computed")
            continue
        want = float(raw)
        tol = max(0.2, 0.01 * abs(want))
        if abs(got - want) > tol:
            failures.append(f"{key}: got {got}, expected {want}")
    if failures:
        for f in failures:
            print(f"  EXPECT FAIL: {f}", file=sys.stderr)
        sys.exit(1)
    if args.expect:
        print("  EXPECT OK")


if __name__ == "__main__":
    main()
