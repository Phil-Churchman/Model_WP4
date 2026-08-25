"""
Distribution of the distance discrepancy between recorded and routed trips.

Reads the per-trip analysis written by Simulation/captured_trips_to_geojson.py
and compares, for every trip, the distance the GPS logger actually recorded
against the distance the routing algorithm produced for the same origin and
destination. Writes a histogram figure and the histogram data as CSV into the
scenario's output folder.

    python utilities/distance_discrepancy.py --scenario scenario_nairobi.json

What the number means
---------------------
    discrepancy = routed distance - recorded distance

Positive means the router took a longer way round than the vehicle really did;
negative means the vehicle drove further than the route the model would pick.
Both are interesting, and they say different things:

  * a broad symmetric spread is route-choice noise -- drivers do not always take
    the path the router considers best,
  * a systematic offset means the network or the routing weights disagree with
    reality (missing roads force detours; over-permissive shortcuts do the
    opposite),
  * a long negative tail usually means real journeys that were not point-to-
    point at all -- circuitous trips, waiting, or a logger that kept recording.

Two measures are produced because they answer different questions. The absolute
one (km) shows how much distance is at stake in total; the relative one (% of
the recorded distance) shows whether a discrepancy is proportionally serious,
which a 200 m error on a 300 m trip is and on a 30 km trip is not.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")          # no display needed; write the file and exit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import load_scenario, add_scenario_argument, writable_path

DEFAULT_BINS = 60
MIN_DISTANCE_KM = 0.01        # below this a relative error is meaningless
# Fixed histogram ranges rather than percentile-trimmed ones. A fixed range is
# comparable between scenarios and between runs -- a percentile band silently
# rescales itself when the data changes, so two histograms that look alike can
# be showing quite different spreads. Anything outside is counted, never lost.
DEFAULT_ABS_RANGE_KM = 1.0    # plots -1 km to +1 km
DEFAULT_REL_RANGE_PCT = 100.0 # plots -100% to +100%; -100% is the natural floor,
                              # since a routed distance cannot be negative


def load_discrepancies(path, max_recorded_km=None):
    """Recorded vs routed distance per trip, with the unusable rows removed."""
    df = pd.read_csv(path)
    needed = {"routed_distance_m", "distance_km", "routed"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing column(s): {', '.join(sorted(missing))}\n"
                         "Is it the trip_routing_analysis.csv written by "
                         "Simulation/captured_trips_to_geojson.py?")

    dropped = {}
    before = len(df)
    # A straight-line fallback has no routed distance to speak of.
    df = df[df["routed"] == True]                                    # noqa: E712
    dropped["not routed (straight-line fallback)"] = before - len(df)

    before = len(df)
    df = df[pd.to_numeric(df["distance_km"], errors="coerce") > MIN_DISTANCE_KM]
    dropped[f"recorded distance under {MIN_DISTANCE_KM * 1000:.0f} m"] = before - len(df)

    if max_recorded_km:
        before = len(df)
        df = df[df["distance_km"] <= max_recorded_km]
        dropped[f"recorded distance over {max_recorded_km:g} km"] = before - len(df)

    recorded = df["distance_km"].to_numpy(dtype=float)
    routed = df["routed_distance_m"].to_numpy(dtype=float) / 1000.0
    return recorded, routed, dropped, df


def histogram_rows(values, measure, unit, bins, limit):
    """
    Bin `values` over a fixed range of -limit..+limit and return
    (rows, edges, clipped_low, clipped_high).

    The range is fixed rather than derived from the data, so histograms stay
    comparable across runs and scenarios. It affects the plotted RANGE only --
    everything below and above is counted and reported, so no trip disappears.
    """
    lo, hi = -float(limit), float(limit)
    inside = values[(values >= lo) & (values <= hi)]
    counts, edges = np.histogram(inside, bins=bins, range=(lo, hi))
    clipped_low = int((values < lo).sum())
    clipped_high = int((values > hi).sum())

    total = len(values)
    cumulative = clipped_low + np.cumsum(counts)
    rows = [{
        "measure": measure,
        "unit": unit,
        "bin_index": i,
        "bin_left": round(float(edges[i]), 4),
        "bin_right": round(float(edges[i + 1]), 4),
        "bin_centre": round(float((edges[i] + edges[i + 1]) / 2), 4),
        "count": int(c),
        "share_pct": round(100.0 * c / total, 4),
        "cumulative_pct": round(100.0 * cumulative[i] / total, 4),
    } for i, c in enumerate(counts)]

    # Everything outside the plotted range, kept as explicit rows so the CSV
    # accounts for all trips rather than only the ones that fitted on screen.
    if clipped_low:
        rows.insert(0, {"measure": measure, "unit": unit, "bin_index": -1,
                        "bin_left": float("-inf"), "bin_right": round(lo, 4),
                        "bin_centre": None, "count": clipped_low,
                        "share_pct": round(100.0 * clipped_low / total, 4),
                        "cumulative_pct": round(100.0 * clipped_low / total, 4)})
    if clipped_high:
        rows.append({"measure": measure, "unit": unit, "bin_index": bins,
                     "bin_left": round(hi, 4), "bin_right": float("inf"),
                     "bin_centre": None, "count": clipped_high,
                     "share_pct": round(100.0 * clipped_high / total, 4),
                     "cumulative_pct": 100.0})
    return rows, edges, clipped_low, clipped_high


def describe(values, unit):
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p1": float(np.percentile(values, 1)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "pct_routed_longer": float(100.0 * (values > 0).mean()),
        "pct_routed_shorter": float(100.0 * (values < 0).mean()),
        "unit": unit,
    }


def plot(abs_km, rel_pct, stats_abs, stats_rel, bins, abs_limit, rel_limit, log_y,
         title, out_path):
    """
    Histograms on top, cumulative distributions below.

    The distribution is extremely peaked -- on the Nairobi capture roughly half
    the trips agree to within 50 m -- so the histogram alone puts almost
    everything in one bar and says little about the tails. The cumulative panel
    is immune to that and answers the question actually being asked: what
    fraction of trips agree to within some tolerance. --log-y rescues the
    histogram too, at the cost of a y-axis that has to be read carefully.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    panels = ((abs_km, stats_abs, "Absolute discrepancy", "km", abs_limit),
              (rel_pct, stats_rel, "Relative discrepancy", "% of recorded", rel_limit))

    for col, (values, stats, label, unit, limit) in enumerate(panels):
        lo, hi = -float(limit), float(limit)
        outside = int(((values < lo) | (values > hi)).sum())

        ax = axes[0][col]
        ax.hist(values, bins=bins, range=(lo, hi), color="#3498db",
                edgecolor="white", linewidth=0.4)
        if log_y:
            ax.set_yscale("log")
        ax.axvline(0, color="#2c3e50", lw=1.6, label="no discrepancy")
        ax.axvline(stats["median"], color="#e67e22", lw=1.8, ls="--",
                   label=f"median {stats['median']:.2f}")
        ax.axvline(stats["mean"], color="#c0392b", lw=1.8, ls=":",
                   label=f"mean {stats['mean']:.2f}")
        ax.set_xlim(lo, hi)
        # Say how much sits beyond the fixed range, so a heavy tail cannot be
        # mistaken for an absent one.
        ax.set_title(f"{label}  ({unit})"
                     + (f"   -   {outside:,} trips outside +/-{limit:g}"
                        if outside else ""),
                     fontsize=13, fontweight="bold")
        ax.set_xlabel(f"routed - recorded  ({unit})")
        ax.set_ylabel("trips" + (" (log scale)" if log_y else ""))
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)

        # Cumulative share of trips within a given absolute tolerance.
        ax = axes[1][col]
        magnitude = np.abs(values)
        order = np.sort(magnitude)
        share = 100.0 * np.arange(1, len(order) + 1) / len(order)
        cap = np.percentile(magnitude, 99)
        ax.plot(order, share, color="#2c3e50", lw=2)
        ax.set_xlim(0, cap if cap > 0 else 1)
        ax.set_ylim(0, 100)
        for mark, colour in ((50, "#bdc3c7"), (80, "#bdc3c7"), (95, "#bdc3c7")):
            ax.axhline(mark, color=colour, lw=0.8, ls=":")
        for frac in (50, 80, 95):
            x = float(np.percentile(magnitude, frac))
            ax.plot([x], [frac], "o", color="#e67e22", ms=6)
            ax.annotate(f"{frac}% within {x:.2f}", (x, frac),
                        textcoords="offset points", xytext=(8, -12), fontsize=9)
        ax.set_title(f"Trips within a tolerance  ({unit})", fontsize=12)
        ax.set_xlabel(f"|routed - recorded|  ({unit})")
        ax.set_ylabel("% of trips within")
        ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.005,
             f"positive = router went further than the vehicle actually did   |   "
             f"histograms fixed at +/-{abs_limit:g} km and +/-{rel_limit:g}%   |   "
             f"cumulative panels cut at the 99th percentile",
             ha="center", fontsize=9, color="#555555")
    plt.tight_layout(rect=[0, 0.025, 1, 0.955])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scenario_argument(p)
    p.add_argument("--input", default=None,
                   help="trip_routing_analysis.csv (default: in the scenario's "
                        "output_trips_time_queued folder)")
    p.add_argument("--out-csv", default=None,
                   help="histogram data CSV (default: distance_discrepancy_"
                        "histogram.csv in the scenario's output folder)")
    p.add_argument("--out-png", default=None,
                   help="figure (default: distance_discrepancy_histogram.png "
                        "in the scenario's output folder)")
    p.add_argument("--bins", type=int, default=DEFAULT_BINS)
    p.add_argument("--abs-range-km", type=float, default=DEFAULT_ABS_RANGE_KM,
                   help="absolute histogram spans -X..+X km "
                        f"(default {DEFAULT_ABS_RANGE_KM:g}); trips outside are "
                        "still counted, in the CSV and in the panel title")
    p.add_argument("--rel-range-pct", type=float, default=DEFAULT_REL_RANGE_PCT,
                   help="relative histogram spans -X..+X percent "
                        f"(default {DEFAULT_REL_RANGE_PCT:g})")
    p.add_argument("--max-recorded-km", type=float, default=None,
                   help="ignore trips whose recorded distance exceeds this")
    p.add_argument("--log-y", action="store_true",
                   help="log-scale the histogram counts, which makes the tails "
                        "visible when the distribution is sharply peaked")
    return p.parse_args()


def main():
    args = parse_args()
    scenario = load_scenario(args.scenario)

    src = args.input or os.path.join(scenario.trips_time_dir,
                                     "trip_routing_analysis.csv")
    if not os.path.exists(src):
        raise SystemExit(
            f"No analysis file at {src}\nRun Simulation/captured_trips_to_geojson.py "
            "first -- it writes trip_routing_analysis.csv alongside the tracks.")

    out_csv = args.out_csv or os.path.join(scenario.output_dir,
                                           "distance_discrepancy_histogram.csv")
    out_png = args.out_png or os.path.join(scenario.output_dir,
                                           "distance_discrepancy_histogram.png")

    print(f"Scenario : {scenario.name}")
    print(f"Analysis : {src}")

    recorded, routed, dropped, _df = load_discrepancies(src, args.max_recorded_km)
    abs_km = routed - recorded
    rel_pct = 100.0 * abs_km / recorded

    print(f"\nTrips compared: {len(abs_km):,}")
    for reason, n in dropped.items():
        if n:
            print(f"  dropped {n:,}: {reason}")

    stats_abs = describe(abs_km, "km")
    stats_rel = describe(rel_pct, "% of recorded")

    rows_abs, _, lo_a, hi_a = histogram_rows(abs_km, "absolute", "km",
                                             args.bins, args.abs_range_km)
    rows_rel, _, lo_r, hi_r = histogram_rows(rel_pct, "relative", "pct",
                                             args.bins, args.rel_range_pct)

    out_csv = writable_path(out_csv)
    pd.DataFrame(rows_abs + rows_rel).to_csv(out_csv, index=False)

    out_png = writable_path(out_png)
    plot(abs_km, rel_pct, stats_abs, stats_rel, args.bins, args.abs_range_km,
         args.rel_range_pct, args.log_y,
         f"Recorded vs routed trip distance - {scenario.name}", out_png)

    print(f"\n{'':14s} {'absolute (km)':>16s} {'relative (%)':>16s}")
    for key, label in [("mean", "mean"), ("median", "median"), ("std", "std dev"),
                       ("p5", "5th pct"), ("p25", "25th pct"), ("p75", "75th pct"),
                       ("p95", "95th pct"), ("min", "min"), ("max", "max")]:
        print(f"  {label:12s} {stats_abs[key]:>16.2f} {stats_rel[key]:>16.1f}")
    print(f"  {'router further':12s} {stats_abs['pct_routed_longer']:>15.1f}% "
          f"{'':>15s}")
    print(f"  {'router shorter':12s} {stats_abs['pct_routed_shorter']:>15.1f}%")

    if lo_a or hi_a:
        print(f"\n  Outside the plotted range: {lo_a:,} below, {hi_a:,} above "
              f"({100.0 * (lo_a + hi_a) / len(abs_km):.1f}% of trips). "
              "Both are counted in the CSV.")

    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
