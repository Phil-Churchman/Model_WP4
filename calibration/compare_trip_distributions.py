"""
Compare what a distribution-mode run actually produced against what it was asked
to produce.

Reads the agent tracks from <folder_name>/output/output_trips_time_queued and
the target distributions from trip_distributions.json, and plots the two against
each other -- trip distance and fare wait, as both per-bin share and cumulative
percentage. Writes a PNG and a CSV of the same numbers.

    python calibration/compare_trip_distributions.py --scenario scenario_nairobi.json

The simulation does not sample the target directly. It draws a distance BAND and
then looks for a road node that far away in a straight line, which can fail, and
it refuses destinations that would leave a vehicle unable to reach a swap
station. Every one of those is a place the realised distribution can drift away
from the target, and none of them announces itself -- the run completes and the
numbers look plausible. This is the check that says whether it drifted.


================================================================================
READING THE RESULT
================================================================================
Everything is binned onto the TARGET's own bin edges, so the comparison is
like-for-like and no binning choice of this script can create or hide a
discrepancy.

Three reasons a gap here may be expected rather than a defect:

1.  Range truncation. A vehicle near the end of its charge can only accept
    destinations it can still reach a swap station from, so long trips are
    refused near the end of a cycle and the realised distance distribution is
    pulled short. Raising max_total_distance_m reduces it.

2.  Road circuity. The distance distribution is a distribution of ROAD
    distances, but the sampler works in straight-line space and converts with
    deviation_factor. If that factor does not match the network, realised
    distances are systematically long or short -- a shift of the whole
    distribution rather than a change in its shape.

3.  Length bias in the fare wait. Over a fixed simulated window, a vehicle that
    draws long waits completes fewer cycles and so contributes fewer samples
    than one drawing short waits. The recorded waits therefore over-represent
    short ones, and the realised mean sits BELOW the target even when sampling
    is perfect. It is a property of observing a renewal process for a fixed
    time, not an error, and it shrinks as the window lengthens.

Simulated waits are also rounded up to a whole simulation_step_sec, so at a
30 s step the shortest bins cannot be reproduced exactly.

Total variation distance is the headline number: half the sum of the absolute
per-bin differences in share, i.e. the fraction of probability mass sitting in
the wrong bin. 0 is identical, 1 is no overlap. Chi-square is reported too, but
read it with care -- at tens of thousands of trips it rejects on differences too
small to matter.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import (load_scenario, add_scenario_argument, writable_path,
                             load_trip_distributions, simulation_mode, DISTRIBUTION)

TRIP_TYPES = ("pickup", "passenger")


def read_run(run_dir, max_agents=None):
    """
    Pull the realised trip distances and waits out of the agent tracks.

    A LineString feature is a trip and carries length_m; a Point feature is the
    wait served on arrival and carries duration_s. The wait is typed by the trip
    it follows, so a Point of type "passenger" is the fare wait that trip earned
    and a Point of type "pickup" is the boarding dwell.
    """
    if not os.path.isdir(run_dir):
        raise SystemExit(f"No run output at:\n  {run_dir}\nRun the simulation first.")

    files = sorted(f for f in os.listdir(run_dir) if f.endswith(".geojson"))
    if max_agents:
        files = files[:max_agents]
    if not files:
        raise SystemExit(f"No agent files in {run_dir}.")

    distances = {t: [] for t in TRIP_TYPES}
    fare_waits, pickup_waits = [], []
    counts = Counter()

    for name in files:
        with open(os.path.join(run_dir, name), "r", encoding="utf-8") as f:
            feats = json.load(f).get("features", [])
        for ft in feats:
            props = ft.get("properties", {})
            kind = props.get("type")
            counts[kind] += 1
            is_trip = ft.get("geometry", {}).get("type") == "LineString"
            if is_trip and kind in distances:
                distances[kind].append(float(props.get("length_m", 0.0)))
            elif not is_trip:
                if kind == "passenger":
                    fare_waits.append(float(props.get("duration_s", 0.0)))
                elif kind == "pickup":
                    pickup_waits.append(float(props.get("duration_s", 0.0)))

    print(f"Run: {len(files)} agent files")
    print(f"  features by type: {dict(counts)}")
    return distances, np.array(fare_waits), np.array(pickup_waits)


def binned_share(values, lows, highs):
    """
    Fraction of `values` falling in each target bin, plus the counts.

    Values above the top edge are folded into the last bin rather than dropped,
    so the shares still sum to 1 and an overshooting tail shows up as weight in
    the top bin instead of quietly vanishing.
    """
    values = np.asarray(values, dtype=float)
    edges = np.concatenate((lows, [highs[-1]]))
    idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(lows) - 1)
    counts = np.bincount(idx, minlength=len(lows)).astype(float)
    total = counts.sum()
    return (counts / total if total else counts), counts


def _target_median(lows, highs, target):
    """Median of the piecewise-uniform target, interpolated within its bin."""
    cum = np.cumsum(target)
    i = int(np.searchsorted(cum, 0.5))
    i = min(i, len(lows) - 1)
    below = cum[i - 1] if i else 0.0
    frac = (0.5 - below) / target[i] if target[i] > 0 else 0.5
    return float(lows[i] + frac * (highs[i] - lows[i]))


def compare(name, values, bands, scale, unit):
    """Target vs realised for one quantity, as shares over the target's bins."""
    lows = np.array(bands.lows) / scale
    highs = np.array(bands.highs) / scale
    weights = np.diff(np.concatenate(([0.0], bands.cum)))
    target = weights / weights.sum()

    values = np.asarray(values, dtype=float) / scale
    share, counts = binned_share(values, lows, highs)

    mids = (lows + highs) / 2.0
    tvd = 0.5 * np.abs(share - target).sum()

    # Chi-square against the target shares. Bins the target never draws from
    # would divide by zero, and they carry no information either way.
    live = target > 0
    expected = target[live] * counts.sum()
    chi2 = float((((counts[live] - expected) ** 2) / expected).sum()) if counts.sum() else float("nan")

    stats = {
        "name": name,
        "n": int(counts.sum()),
        "target_mean": float((target * mids).sum()),
        "actual_mean": float(values.mean()) if values.size else float("nan"),
        # Interpolated inside the crossing bin, not that bin's midpoint. The
        # target is piecewise UNIFORM, so its mean really is the weighted mean of
        # the midpoints, but its median is not -- and reporting a midpoint here
        # made the target median jump in bin-sized steps and look like a
        # discrepancy against the exact median of the realised sample.
        "target_median": _target_median(lows, highs, target),
        "actual_median": float(np.median(values)) if values.size else float("nan"),
        "tvd": float(tvd),
        "chi2": chi2,
        "dof": int(live.sum() - 1),
        "unit": unit,
    }
    return lows, highs, target, share, stats


def print_stats(s):
    d = s["actual_mean"] - s["target_mean"]
    pct = 100.0 * d / s["target_mean"] if s["target_mean"] else float("nan")
    print(f"\n{s['name']}  (n = {s['n']:,})")
    print(f"  mean    target {s['target_mean']:8.2f}{s['unit']}   "
          f"actual {s['actual_mean']:8.2f}{s['unit']}   "
          f"{d:+.2f}{s['unit']} ({pct:+.1f}%)")
    print(f"  median  target {s['target_median']:8.2f}{s['unit']}   "
          f"actual {s['actual_median']:8.2f}{s['unit']}")
    print(f"  total variation distance {s['tvd']:.3f}  "
          f"({100 * s['tvd']:.1f}% of the mass is in the wrong bin)")
    print(f"  chi-square {s['chi2']:,.0f} on {s['dof']} dof")


def panel_bars(ax, lows, highs, target, actual, title, unit):
    """Per-bin share, target beside realised, plotted against bin INDEX."""
    x = np.arange(len(lows))
    w = 0.42
    ax.bar(x - w / 2, 100 * target, w, label="target", color="#3498db")
    ax.bar(x + w / 2, 100 * actual, w, label="simulated", color="#e67e22")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("share of trips (%)")
    # Bins are deliberately unequal in width, so plotting against value would
    # make the wide tail bins look like the bulk of the distribution. Index
    # spacing keeps every bin equally readable; the edges are on the ticks.
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lo:g}" for lo in lows] , rotation=90, fontsize=7)
    ax.set_xlabel(f"bin lower edge ({unit.strip()})  -- bins are unequal width")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)


def panel_cdf(ax, lows, highs, target, actual, title, unit):
    """Cumulative percentage, both curves through the same bin edges."""
    edges = np.concatenate((lows, [highs[-1]]))
    ax.plot(edges, 100 * np.concatenate(([0.0], np.cumsum(target))),
            lw=2.5, color="#3498db", label="target", marker="o", ms=3)
    ax.plot(edges, 100 * np.concatenate(([0.0], np.cumsum(actual))),
            lw=2.5, color="#e67e22", label="simulated", marker="s", ms=3)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("cumulative (%)")
    ax.set_xlabel(unit.strip())
    ax.set_ylim(0, 101)
    # symlog so the zero edge is representable, but clamped to the positive
    # side: left to itself it draws a -10^0 .. -10^-1 branch for data that
    # cannot be negative, wasting half the panel.
    ax.set_xscale("symlog", linthresh=max(float(edges[1]), 1e-9))
    ax.set_xlim(0, float(edges[-1]))
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25)


def main():
    p = argparse.ArgumentParser(
        description="Plot a distribution-mode run against trip_distributions.json")
    add_scenario_argument(p)
    p.add_argument("--run-dir", default=None,
                   help="agent tracks (default: the scenario's "
                        "output/output_trips_time_queued)")
    p.add_argument("--distributions", default=None,
                   help="trip_distributions.json to compare against (default: "
                        "the one the simulation would resolve)")
    p.add_argument("--out", default=None, help="PNG to write")
    p.add_argument("--csv", default=None, help="CSV of the per-bin numbers")
    p.add_argument("--max-agents", type=int, default=None,
                   help="read only the first N agent files")
    p.add_argument("--split-legs", action="store_true",
                   help="compare the pickup and passenger legs separately as "
                        "well as combined")
    args = p.parse_args()

    scenario = load_scenario(args.scenario).require_folder()
    mode = simulation_mode(scenario)
    if mode != DISTRIBUTION:
        print(f"NOTE: {os.path.basename(scenario.path)} is in {mode!r} mode, not "
              f"{DISTRIBUTION!r}.\n  The run being read may not have drawn from "
              f"these distributions at all.\n")

    run_dir = args.run_dir or scenario.trips_time_dir
    out_png = args.out or os.path.join(scenario.output_dir,
                                       "trip_distribution_comparison.png")
    out_csv = args.csv or os.path.join(scenario.output_dir,
                                       "trip_distribution_comparison.csv")

    bands, src = load_trip_distributions(scenario, path=args.distributions)
    print(f"Scenario : {scenario.path}")
    print(f"Target   : {src}")
    print(f"Run      : {run_dir}\n")

    distances, fare_waits, pickup_waits = read_run(run_dir, args.max_agents)
    all_trips = np.array(distances["pickup"] + distances["passenger"])
    if all_trips.size == 0:
        raise SystemExit(
            "No pickup or passenger trips in the run. Distribution mode "
            "produces both; a hail-rank run has neither.")
    if fare_waits.size == 0:
        raise SystemExit("No fare waits in the run.")

    series = [("trip distance (both legs)", all_trips, bands["distance"], 1000.0, " km")]
    if args.split_legs:
        for leg in TRIP_TYPES:
            if distances[leg]:
                series.append((f"trip distance ({leg})", np.array(distances[leg]),
                               bands["distance"], 1000.0, " km"))
    series.append(("fare wait", fare_waits, bands["wait"], 60.0, " min"))

    results, rows = [], []
    for name, values, band, scale, unit in series:
        lows, highs, target, actual, stats = compare(name, values, band, scale, unit)
        print_stats(stats)
        results.append((name, lows, highs, target, actual, stats))
        for lo, hi, t, a in zip(lows, highs, target, actual):
            rows.append({"series": name, "unit": unit.strip(),
                         "bin_min": lo, "bin_max": hi,
                         "target_share": round(float(t), 6),
                         "actual_share": round(float(a), 6),
                         "difference": round(float(a - t), 6)})

    if pickup_waits.size:
        print(f"\npickup wait  (n = {pickup_waits.size:,})  "
              f"mean {pickup_waits.mean() / 60:.2f} min -- fixed at "
              f"pickup_wait_sec, not drawn, so it is reported but not compared.")

    # ---- figure ----
    plotted = [r for r in results if not r[0].startswith("trip distance (p")] \
        if not args.split_legs else results
    n = len(plotted)
    fig, axes = plt.subplots(n, 2, figsize=(15, 4.6 * n), squeeze=False)
    for i, (name, lows, highs, target, actual, stats) in enumerate(plotted):
        unit = stats["unit"]
        panel_bars(axes[i][0], lows, highs, target, actual,
                   f"{name} -- per bin", unit)
        panel_cdf(axes[i][1], lows, highs, target, actual,
                  f"{name} -- cumulative", unit)
        axes[i][0].text(
            0.98, 0.95,
            f"n = {stats['n']:,}\nmean {stats['actual_mean']:.2f} vs "
            f"{stats['target_mean']:.2f}{unit}\nTVD {stats['tvd']:.3f}",
            transform=axes[i][0].transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.85))

    fig.suptitle(f"Realised vs target distributions -- {scenario.name}",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_png = writable_path(out_png)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    out_csv = writable_path(out_csv)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nWritten:\n  {out_png}\n  {out_csv}")


if __name__ == "__main__":
    main()
