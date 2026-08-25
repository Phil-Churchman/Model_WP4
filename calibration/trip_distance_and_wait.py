"""
Trip distance and wait-time distributions for the captured fleet.

Reads the per-trip analysis written by captured_trips_to_geojson.py and produces
four histograms, plus the binned data as CSV:

  1. trip distance -- every trip
  2. wait time since the previous trip -- every trip except the first of an
     agent's day, for which the "wait" is overnight and meaningless
  3. wait time for the subset that waited IN PLACE: this trip started within
     --same-place-m of where the previous one ended
  4. how far the vehicle repositioned between trips, with the threshold marked,
     so the choice of 100 m can be judged against the data rather than assumed

    python calibration/trip_distance_and_wait.py --scenario scenario.json

Why the in-place subset is broken out rather than filtered
----------------------------------------------------------
A gap between two trips is only a wait AT A PLACE if the vehicle was still
there. When the next trip starts somewhere else, the gap covers a journey the
logger did not record -- a dead-heading leg, or a break in recording -- and
reading it as idle time at the drop-off point would be wrong.

Both populations are therefore shown: panel 2 is every gap, panel 3 only those
where the vehicle demonstrably stayed put. If the two distributions differ, the
"wait" in panel 2 is partly travel and should not be used as a dwell time.

A limitation worth knowing
--------------------------
gap_from_previous_s was computed against the true preceding trip, before rows
with missing coordinates were dropped. The END LOCATION of that trip is only
available if it survived into the analysis file. Where it did not, the distance
test cannot be applied, and those trips are reported separately rather than
being quietly counted as "moved" or "stayed". On the Nairobi capture the
predecessor is present for about 98% of trips.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import load_scenario, add_scenario_argument, writable_path

DEFAULT_SAME_PLACE_M = 100.0
DEFAULT_DISTANCE_BINS = 40
DEFAULT_MAX_DISTANCE_KM = 20.0
DEFAULT_WAIT_BINS = 40
DEFAULT_MAX_WAIT_MIN = 60.0
DEFAULT_MAX_REPOSITION_M = 2000.0
EARTH_RADIUS_M = 6371000.0


def detect_quantum(values, scale=1000):
    """
    The smallest step the data actually uses, or None if it is effectively
    continuous.

    Wait times come from timestamps recorded to the whole minute, so every gap
    is an exact number of minutes with nothing in between. That matters for
    binning: a bin width that is not a whole number of minutes captures two
    distinct values in some bins and one in others, producing an alternating
    comb of peaks and troughs that looks like structure and is only arithmetic.
    """
    v = np.unique(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return None
    scaled = np.round(v * scale)
    if not np.allclose(v * scale, scaled, atol=1e-6):
        return None                              # finer than `scale` can express
    nonzero = np.abs(scaled[scaled != 0]).astype(np.int64)
    if not nonzero.size:
        return None
    # GCD converges within a handful of values; no need to fold all 60,000.
    step = int(np.gcd.reduce(nonzero[:2000]))
    if step == 0:
        return None
    quantum = step / scale
    # A quantum far below the resolution of any sane histogram is continuous
    # for our purposes.
    span = float(v.max() - v.min())
    return quantum if span > 0 and quantum >= span / 5000 else None


def aligned_bins(cap, requested, quantum):
    """
    A bin count whose width is a whole number of quanta AND divides the range
    exactly, so every bin holds the same number of distinct values.

    Returns (bin_count, width, adjusted) -- adjusted is True when the requested
    count had to move. The count closest to what was asked for is chosen, so the
    resolution stays near the caller's intent.
    """
    if not quantum:
        return requested, cap / requested, False
    steps = cap / quantum
    if abs(steps - round(steps)) > 1e-6:
        return requested, cap / requested, False     # cap is not a whole number of quanta
    steps = int(round(steps))
    options = sorted({steps // k for k in range(1, steps + 1) if steps % k == 0})
    best = min(options, key=lambda n: (abs(n - requested), -n))
    return best, cap / best, best != requested


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle metres between two arrays of WGS84 points."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_trips(path):
    """
    Trip distances, waits, and the reposition distance where it can be measured.

    Returns (DataFrame, dict of note counts).
    """
    df = pd.read_csv(path)
    needed = {"user_id", "start_date", "start_time", "end_date", "end_time",
              "distance_km", "gap_from_previous_s",
              "start_lat", "start_lon", "end_lat", "end_lon"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(
            f"{path} is missing column(s): {', '.join(sorted(missing))}\n"
            "Regenerate it with the current calibration/captured_trips_to_geojson.py.")

    notes = {}
    df = df.sort_values(["user_id", "start_date", "start_time"]).reset_index(drop=True)
    df["start_dt"] = pd.to_datetime(df["start_date"].astype(str) + " "
                                    + df["start_time"].astype(str), errors="coerce")
    df["end_dt"] = pd.to_datetime(df["end_date"].astype(str) + " "
                                  + df["end_time"].astype(str), errors="coerce")

    # First trip of each agent-day: its gap is overnight, not a wait.
    by_day = df.groupby(["user_id", "start_date"], sort=False)
    df["first_of_day"] = by_day.cumcount() == 0
    notes["first trip of an agent-day (no wait)"] = int(df["first_of_day"].sum())

    # Previous retained trip, per agent.
    by_agent = df.groupby("user_id", sort=False)
    prev_end_lat = by_agent["end_lat"].shift(1)
    prev_end_lon = by_agent["end_lon"].shift(1)
    prev_end_dt = by_agent["end_dt"].shift(1)

    # The stored gap refers to the TRUE predecessor, which may have been dropped
    # for missing coordinates. Only when the retained row reproduces that gap is
    # its end location the right thing to measure against.
    recomputed = (df["start_dt"] - prev_end_dt).dt.total_seconds()
    df["predecessor_known"] = np.isclose(recomputed, df["gap_from_previous_s"],
                                         atol=1.0, equal_nan=False)

    df["reposition_m"] = np.where(
        df["predecessor_known"],
        haversine_m(prev_end_lat.to_numpy(), prev_end_lon.to_numpy(),
                    df["start_lat"].to_numpy(), df["start_lon"].to_numpy()),
        np.nan)

    df["wait_min"] = df["gap_from_previous_s"] / 60.0
    df["trip_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
    return df, notes


def histogram_rows(values, measure, unit, bins, cap):
    """Bin 0..cap, with everything above gathered into one overflow row."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    total = len(values)
    if not total:
        return [], 0
    counts, edges = np.histogram(values[values <= cap], bins=bins, range=(0.0, cap))
    overflow = int((values > cap).sum())

    rows, running = [], 0
    for i, c in enumerate(counts):
        running += int(c)
        rows.append({
            "measure": measure, "unit": unit, "bin_index": i,
            "bin_left": round(float(edges[i]), 4),
            "bin_right": round(float(edges[i + 1]), 4),
            "bin_centre": round(float((edges[i] + edges[i + 1]) / 2), 4),
            "count": int(c),
            "share_pct": round(100.0 * c / total, 4),
            "cumulative_pct": round(100.0 * running / total, 4),
        })
    if overflow:
        rows.append({
            "measure": measure, "unit": unit, "bin_index": bins,
            "bin_left": round(float(cap), 4), "bin_right": float("inf"),
            "bin_centre": None, "count": overflow,
            "share_pct": round(100.0 * overflow / total, 4),
            "cumulative_pct": 100.0,
        })
    return rows, overflow


def describe(values, unit):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "unit": unit}
    return {
        "n": int(len(values)), "unit": unit,
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def panel(ax, values, cap, bins, colour, title, xlabel, stats, log_y):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    over = int((values > cap).sum())
    ax.hist(values[values <= cap], bins=bins, range=(0.0, cap), color=colour,
            edgecolor="white", linewidth=0.4)
    if log_y:
        ax.set_yscale("log")
    if stats.get("n"):
        ax.axvline(stats["median"], color="#2c3e50", lw=1.8, ls="--",
                   label=f"median {stats['median']:.2f}")
        ax.axvline(stats["mean"], color="#c0392b", lw=1.6, ls=":",
                   label=f"mean {stats['mean']:.2f}")
        ax.legend(fontsize=9)
    ax.set_title(f"{title}\n{stats.get('n', 0):,} trips"
                 + (f"   ({over:,} above {cap:g})" if over else ""),
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("trips" + (" (log scale)" if log_y else ""))
    ax.set_xlim(0, cap)
    ax.grid(axis="y", alpha=0.25)


def cdf_panel(ax, values, colour, title, xlabel, log_x, reference=None,
              reference_label=None):
    """
    Cumulative share of trips at or below each value, reaching 100%.

    Plotted over the FULL range rather than a trimmed one, so the curve actually
    arrives at 100% -- which is the point of a cumulative view. These
    distributions have long tails, so --cdf-log-x is usually the readable
    choice; it uses a symmetric log scale so zero values (common here) still
    have a place on the axis.
    """
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    if not values.size:
        ax.set_title(f"{title}\nno data", fontsize=12, fontweight="bold")
        return

    share = 100.0 * np.arange(1, values.size + 1) / values.size

    if reference is not None:
        ref = np.sort(np.asarray(reference, dtype=float))
        ref = ref[np.isfinite(ref)]
        if ref.size:
            ax.plot(ref, 100.0 * np.arange(1, ref.size + 1) / ref.size,
                    color="#95a5a6", lw=1.6, ls="--", label=reference_label)

    ax.plot(values, share, color=colour, lw=2.2, label="this measure"
            if reference is not None else None)

    for frac, colour_mark in ((50, "#bdc3c7"), (80, "#bdc3c7"),
                              (90, "#bdc3c7"), (95, "#bdc3c7"), (99, "#bdc3c7")):
        x = float(np.percentile(values, frac))
        ax.axhline(frac, color=colour_mark, lw=0.7, ls=":")
        ax.plot([x], [frac], "o", color="#e67e22", ms=5)
        ax.annotate(f"{frac}%: {x:,.4g}", (x, frac), textcoords="offset points",
                    xytext=(7, -11), fontsize=8.5)

    if log_x:
        # symlog, not log: a zero wait and a zero reposition are real values.
        ax.set_xscale("symlog", linthresh=max(values[values > 0].min(), 1e-3)
                      if (values > 0).any() else 1.0)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 101)
    ax.set_title(f"{title}\n{values.size:,} trips, to 100%", fontsize=12,
                 fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("% of trips at or below")
    ax.grid(alpha=0.25)
    if reference is not None:
        ax.legend(fontsize=9, loc="lower right")


def plot_cdfs(data, args, title, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    cdf_panel(axes[0][0], data["trip_km"], "#2980b9",
              "Trip distance - all trips", "recorded trip distance (km)",
              args.cdf_log_x)

    cdf_panel(axes[0][1], data["wait_all"], "#27ae60",
              "Wait since previous trip - all\n(first of each agent-day excluded)",
              "wait (minutes)", args.cdf_log_x)

    cdf_panel(axes[1][0], data["wait_same"], "#e67e22",
              f"Wait since previous trip - stayed within {args.same_place_m:g} m",
              "wait (minutes)", args.cdf_log_x,
              reference=data["wait_all"], reference_label="all waits")

    cdf_panel(axes[1][1], data["reposition_m"], "#8e44ad",
              "How far the vehicle moved between trips",
              "distance from previous drop-off to this pick-up (m)",
              args.cdf_log_x)
    axes[1][1].axvline(args.same_place_m, color="#c0392b", lw=1.8,
                       label=f"{args.same_place_m:g} m threshold")
    axes[1][1].legend(fontsize=9, loc="lower right")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.005,
             "cumulative over the full range, so every curve reaches 100%   |   "
             + ("symmetric log x-axis" if args.cdf_log_x
                else "linear x-axis; --cdf-log-x is easier to read on these tails"),
             ha="center", fontsize=9, color="#555555")
    plt.tight_layout(rect=[0, 0.025, 1, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot(data, args, stats, plan, title, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    panel(axes[0][0], data["trip_km"], plan["trip_km"]["cap"],
          plan["trip_km"]["bins"],
          "#2980b9", "Trip distance - all trips", "recorded trip distance (km)",
          stats["trip_km"], args.log_y)

    panel(axes[0][1], data["wait_all"], plan["wait_all"]["cap"],
          plan["wait_all"]["bins"],
          "#27ae60", "Wait since previous trip - all\n(first of each agent-day excluded)",
          "wait (minutes)", stats["wait_all"], args.log_y)

    panel(axes[1][0], data["wait_same"], plan["wait_same"]["cap"],
          plan["wait_same"]["bins"],
          "#e67e22",
          f"Wait since previous trip - stayed within {args.same_place_m:g} m",
          "wait (minutes)", stats["wait_same"], args.log_y)

    ax = axes[1][1]
    panel(ax, data["reposition_m"], plan["reposition_m"]["cap"],
          plan["reposition_m"]["bins"],
          "#8e44ad", "How far the vehicle moved between trips",
          "distance from previous drop-off to this pick-up (m)",
          stats["reposition_m"], args.log_y)
    ax.axvline(args.same_place_m, color="#c0392b", lw=2,
               label=f"{args.same_place_m:g} m threshold")
    ax.legend(fontsize=9)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.005,
             f"in-place subset: this trip started within {args.same_place_m:g} m of "
             f"where the previous one ended   |   "
             f"bars beyond each cap are counted in the CSV, never dropped",
             ha="center", fontsize=9, color="#555555")
    plt.tight_layout(rect=[0, 0.025, 1, 0.95])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scenario_argument(p)
    p.add_argument("--input", default=None)
    p.add_argument("--out-csv", default=None,
                   help="binned data (default: trip_distance_and_wait.csv)")
    p.add_argument("--out-png", default=None,
                   help="histogram figure (default: trip_distance_and_wait.png)")
    p.add_argument("--out-cdf-png", default=None,
                   help="cumulative figure (default: "
                        "trip_distance_and_wait_cumulative.png)")
    p.add_argument("--same-place-m", type=float, default=DEFAULT_SAME_PLACE_M,
                   help="a trip counts as having waited in place when it starts "
                        "within this many metres of the previous trip's end "
                        f"(default {DEFAULT_SAME_PLACE_M:g})")
    p.add_argument("--distance-bins", type=int, default=DEFAULT_DISTANCE_BINS)
    p.add_argument("--max-distance-km", type=float, default=DEFAULT_MAX_DISTANCE_KM)
    p.add_argument("--wait-bins", type=int, default=DEFAULT_WAIT_BINS)
    p.add_argument("--max-wait-min", type=float, default=DEFAULT_MAX_WAIT_MIN)
    p.add_argument("--max-reposition-m", type=float, default=DEFAULT_MAX_REPOSITION_M)
    p.add_argument("--no-align", action="store_true",
                   help="do not snap bin counts to the data's step size; bins "
                        "may then span unequal numbers of distinct values, "
                        "which shows up as an alternating comb")
    p.add_argument("--cdf-log-x", action="store_true",
                   help="symmetric-log x-axis on the cumulative plots, which "
                        "these long tails usually need to be readable")
    p.add_argument("--log-y", action="store_true",
                   help="log-scale the counts, which helps when a distribution "
                        "is sharply peaked")
    return p.parse_args()


def main():
    args = parse_args()
    scenario = load_scenario(args.scenario)

    src = args.input or os.path.join(scenario.trips_time_dir,
                                     "trip_routing_analysis.csv")
    if not os.path.exists(src):
        raise SystemExit(
            f"No analysis file at {src}\nRun calibration/captured_trips_to_geojson.py "
            "first -- it writes trip_routing_analysis.csv alongside the tracks.")

    out_csv = args.out_csv or os.path.join(scenario.output_dir,
                                           "trip_distance_and_wait.csv")
    out_png = args.out_png or os.path.join(scenario.output_dir,
                                           "trip_distance_and_wait.png")
    out_cdf = args.out_cdf_png or os.path.join(
        scenario.output_dir, "trip_distance_and_wait_cumulative.png")

    print(f"Scenario : {scenario.name}")
    print(f"Analysis : {src}")

    df, notes = load_trips(src)

    waitable = df[~df["first_of_day"] & df["wait_min"].notna()
                  & (df["wait_min"] >= 0)]
    measurable = waitable[waitable["predecessor_known"]]
    same_place = measurable[measurable["reposition_m"] <= args.same_place_m]

    data = {
        "trip_km": df["trip_km"].to_numpy(),
        "wait_all": waitable["wait_min"].to_numpy(),
        "wait_same": same_place["wait_min"].to_numpy(),
        "reposition_m": measurable["reposition_m"].to_numpy(),
    }
    stats = {
        "trip_km": describe(data["trip_km"], "km"),
        "wait_all": describe(data["wait_all"], "min"),
        "wait_same": describe(data["wait_same"], "min"),
        "reposition_m": describe(data["reposition_m"], "m"),
    }

    print(f"\nTrips in the file: {len(df):,}")
    for reason, n in notes.items():
        print(f"  {n:,}: {reason}")
    print(f"  {len(waitable):,}: have a usable wait")
    unknown = len(waitable) - len(measurable)
    print(f"  {len(measurable):,}: previous trip's end location is known "
          f"({unknown:,} not, so untestable)")
    print(f"  {len(same_place):,}: started within {args.same_place_m:g} m of it "
          f"({100.0 * len(same_place) / len(measurable):.1f}% of testable)")

    # Snap each measure's bin count so every bin spans the same number of
    # distinct values. Without this the whole-minute wait times produce an
    # alternating comb that reads as structure but is only bin arithmetic.
    plan = {}
    print()
    for key, measure, unit, requested, cap in (
            ("trip_km", "trip_distance", "km",
             args.distance_bins, args.max_distance_km),
            ("wait_all", "wait_all", "min", args.wait_bins, args.max_wait_min),
            ("wait_same", f"wait_within_{args.same_place_m:g}m", "min",
             args.wait_bins, args.max_wait_min),
            ("reposition_m", "reposition", "m",
             args.wait_bins, args.max_reposition_m)):
        quantum = None if args.no_align else detect_quantum(data[key])
        bins, width, adjusted = aligned_bins(cap, requested, quantum)
        plan[key] = {"measure": measure, "unit": unit, "bins": bins,
                     "width": width, "cap": cap}
        note = ""
        if quantum:
            note = f"data steps in {quantum:g} {unit}"
            if adjusted:
                note += f"; bins {requested} -> {bins} so every bin spans " \
                        f"{width / quantum:.0f} step(s)"
        elif args.no_align:
            note = "alignment disabled"
        else:
            note = "continuous, no alignment needed"
        print(f"  {measure:22s} {bins:>3d} bins of {width:g} {unit:<4s} {note}")

    rows = []
    for key, spec in plan.items():
        band_rows, _over = histogram_rows(data[key], spec["measure"], spec["unit"],
                                          spec["bins"], spec["cap"])
        rows.extend(band_rows)

    out_csv = writable_path(out_csv)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_png = writable_path(out_png)
    plot(data, args, stats, plan,
         f"Trip distance and waiting time - {scenario.name}", out_png)
    out_cdf = writable_path(out_cdf)
    plot_cdfs(data, args,
              f"Cumulative distributions - {scenario.name}", out_cdf)

    print(f"\n{'':28s} {'n':>9s} {'median':>9s} {'mean':>9s} {'p90':>9s} {'max':>10s}")
    for key, label in (("trip_km", "trip distance, km"),
                       ("wait_all", "wait, all, min"),
                       ("wait_same", f"wait <= {args.same_place_m:g} m, min"),
                       ("reposition_m", "reposition, m")):
        s = stats[key]
        if not s.get("n"):
            print(f"  {label:26s} {'-':>9s}")
            continue
        print(f"  {label:26s} {s['n']:>9,} {s['median']:>9.2f} {s['mean']:>9.2f} "
              f"{s['p90']:>9.2f} {s['max']:>10.1f}")

    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_cdf}")


if __name__ == "__main__":
    main()
