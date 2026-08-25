"""
How trip length relates to the gap since the vehicle's previous trip.

Reads the per-trip analysis written by captured_trips_to_geojson.py and splits
trips by how long the vehicle had been stationary before setting off:

        >= 0  and < 1 minute
        >= 1  and < 10 minutes
        >= 10 and < 60 minutes
        >= 60 minutes

then plots the distribution of trip distance within each band, with the number
of trips in each. Writes the figure and the histogram data as CSV into the
scenario's output folder.

    python calibration/trip_gap_analysis.py --scenario scenario.json

Why the first trip of a day is excluded
---------------------------------------
The gap before a vehicle's first trip of the day is not a gap between two
journeys -- it is overnight, or however long since the logger was last on. It
carries no information about the pause between one job and the next, and being
much the largest number in the dataset it would swamp the >= 60 minute band and
make that band mean "first thing in the morning" rather than "after a long
wait". A day boundary is taken per vehicle, from start_date, so a driver whose
shift crosses midnight is treated as starting a new day at midnight -- the same
convention the source data uses.

What the comparison is for
--------------------------
The gap is a proxy for what the vehicle was doing. A near-zero gap usually means
a continuation -- the previous trip ended and another began immediately, often a
staged leg of one journey. A long gap means the vehicle was genuinely idle and
is now starting fresh work. If those populations have different trip length
distributions, then modelling all trips with a single distribution is losing
something, and the bands are where to look for it.
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

# (label, low minutes inclusive, high minutes exclusive)
GAP_BANDS = [
    (">=0 <1 min",    0.0,   1.0),
    (">=1 <10 min",   1.0,  10.0),
    (">=10 <60 min", 10.0,  60.0),
    (">=60 min",     60.0,  np.inf),
]
BAND_COLOURS = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad"]
DEFAULT_BINS = 40
DEFAULT_MAX_KM = 20.0     # x-axis cap; longer trips are counted in an overflow


def load_trips(path, distance_column):
    """
    Trips with a usable gap, excluding each vehicle's first trip of each day.

    Returns (DataFrame with gap_min and distance_km, dict of drop counts).
    """
    df = pd.read_csv(path)
    needed = {"user_id", "start_date", "start_time", "gap_from_previous_s"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(
            f"{path} is missing column(s): {', '.join(sorted(missing))}\n"
            "Regenerate it with the current calibration/captured_trips_to_geojson.py.")

    dropped = {}
    df = df.sort_values(["user_id", "start_date", "start_time"]).reset_index(drop=True)

    # The first trip of each vehicle-day, dropped by position rather than by
    # gap size: an unusually long gap mid-day is exactly the case worth keeping.
    before = len(df)
    first_of_day = df.groupby(["user_id", "start_date"], sort=False).head(1).index
    df = df.drop(index=first_of_day)
    dropped["first trip of a vehicle-day"] = before - len(df)

    before = len(df)
    df = df[df["gap_from_previous_s"].notna()]
    dropped["no previous trip"] = before - len(df)

    before = len(df)
    # Overlapping records: the next trip starts before the previous one ended.
    df = df[df["gap_from_previous_s"] >= 0]
    dropped["negative gap (overlapping trips)"] = before - len(df)

    if distance_column == "routed":
        if "routed_distance_m" not in df.columns:
            raise SystemExit("No routed_distance_m column for --distance routed")
        distance = df["routed_distance_m"] / 1000.0
    else:
        distance = pd.to_numeric(df["distance_km"], errors="coerce")

    out = pd.DataFrame({
        "user_id": df["user_id"].to_numpy(),
        "gap_min": df["gap_from_previous_s"].to_numpy() / 60.0,
        "distance_km": distance.to_numpy(),
    })
    before = len(out)
    out = out[out["distance_km"].notna() & (out["distance_km"] >= 0)]
    dropped["missing or negative distance"] = before - len(out)
    return out, dropped


def band_of(gap_min):
    for i, (_label, lo, hi) in enumerate(GAP_BANDS):
        if lo <= gap_min < hi:
            return i
    return None


def summarise(distances):
    if not len(distances):
        return {k: None for k in
                ("trips", "mean_km", "median_km", "p25_km", "p75_km", "p90_km", "max_km")}
    return {
        "trips": int(len(distances)),
        "mean_km": round(float(np.mean(distances)), 3),
        "median_km": round(float(np.median(distances)), 3),
        "p25_km": round(float(np.percentile(distances, 25)), 3),
        "p75_km": round(float(np.percentile(distances, 75)), 3),
        "p90_km": round(float(np.percentile(distances, 90)), 3),
        "max_km": round(float(np.max(distances)), 3),
    }


def histogram_rows(distances, label, bins, max_km):
    """
    Bin the distances, with everything beyond max_km gathered into one overflow
    row rather than dropped, so the CSV totals match the band's trip count.
    """
    inside = distances[distances <= max_km]
    counts, edges = np.histogram(inside, bins=bins, range=(0.0, max_km))
    overflow = int((distances > max_km).sum())
    total = len(distances)

    rows, running = [], 0
    for i, c in enumerate(counts):
        running += int(c)
        rows.append({
            "gap_band": label,
            "bin_index": i,
            "bin_left_km": round(float(edges[i]), 4),
            "bin_right_km": round(float(edges[i + 1]), 4),
            "bin_centre_km": round(float((edges[i] + edges[i + 1]) / 2), 4),
            "count": int(c),
            "share_pct": round(100.0 * c / total, 4) if total else 0.0,
            "cumulative_pct": round(100.0 * running / total, 4) if total else 0.0,
        })
    if overflow:
        rows.append({
            "gap_band": label, "bin_index": bins,
            "bin_left_km": round(float(max_km), 4), "bin_right_km": float("inf"),
            "bin_centre_km": None, "count": overflow,
            "share_pct": round(100.0 * overflow / total, 4),
            "cumulative_pct": 100.0,
        })
    return rows, edges, overflow


def plot(bands, stats, bins, max_km, distance_label, title, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ymax = 0
    for distances in bands.values():
        if len(distances):
            counts, _ = np.histogram(distances[distances <= max_km],
                                     bins=bins, range=(0.0, max_km))
            ymax = max(ymax, counts.max())

    for ax, (label, _lo, _hi), colour in zip(axes.flatten(), GAP_BANDS, BAND_COLOURS):
        distances = bands[label]
        s = stats[label]
        ax.hist(distances[distances <= max_km], bins=bins, range=(0.0, max_km),
                color=colour, edgecolor="white", linewidth=0.4)
        # Shared y-axis so the bands are visually comparable, which is the whole
        # point of splitting them.
        ax.set_ylim(0, ymax * 1.08 if ymax else 1)
        if s["trips"]:
            ax.axvline(s["median_km"], color="#2c3e50", lw=1.8, ls="--",
                       label=f"median {s['median_km']:.2f} km")
            ax.axvline(s["mean_km"], color="#c0392b", lw=1.6, ls=":",
                       label=f"mean {s['mean_km']:.2f} km")
            ax.legend(fontsize=9)
        ax.set_title(f"Gap {label}   -   {s['trips']:,} trips", fontsize=13,
                     fontweight="bold")
        ax.set_xlabel(f"{distance_label} (km)")
        ax.set_ylabel("trips")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.005,
             f"gap = time from the end of the previous trip to the start of this one   |   "
             f"first trip of each vehicle-day excluded   |   "
             f"x-axis capped at {max_km:g} km, longer trips counted in the CSV",
             ha="center", fontsize=9, color="#555555")
    plt.tight_layout(rect=[0, 0.025, 1, 0.95])
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
                   help="histogram data (default: trip_gap_analysis.csv in the "
                        "scenario's output folder)")
    p.add_argument("--out-png", default=None,
                   help="figure (default: trip_gap_analysis.png)")
    p.add_argument("--distance", choices=("recorded", "routed"), default="recorded",
                   help="which trip length to bin: the distance the logger "
                        "recorded (default) or the one the router produced")
    p.add_argument("--bins", type=int, default=DEFAULT_BINS)
    p.add_argument("--max-km", type=float, default=DEFAULT_MAX_KM,
                   help=f"x-axis cap in km (default {DEFAULT_MAX_KM:g}); longer "
                        "trips are counted in an overflow row, never dropped")
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

    out_csv = args.out_csv or os.path.join(scenario.output_dir, "trip_gap_analysis.csv")
    out_png = args.out_png or os.path.join(scenario.output_dir, "trip_gap_analysis.png")

    print(f"Scenario : {scenario.name}")
    print(f"Analysis : {src}")
    print(f"Distance : {args.distance}")

    trips, dropped = load_trips(src, args.distance)
    print(f"\nTrips analysed: {len(trips):,}")
    for reason, n in dropped.items():
        if n:
            print(f"  excluded {n:,}: {reason}")

    bands, stats, rows = {}, {}, []
    for label, lo, hi in GAP_BANDS:
        mask = (trips["gap_min"] >= lo) & (trips["gap_min"] < hi)
        distances = trips.loc[mask, "distance_km"].to_numpy()
        bands[label] = distances
        stats[label] = summarise(distances)
        band_rows, _edges, overflow = histogram_rows(distances, label,
                                                     args.bins, args.max_km)
        rows.extend(band_rows)
        stats[label]["over_max_km"] = overflow

    out_csv = writable_path(out_csv)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    distance_label = ("Recorded trip distance" if args.distance == "recorded"
                      else "Routed trip distance")
    out_png = writable_path(out_png)
    plot(bands, stats, args.bins, args.max_km, distance_label,
         f"Trip distance by gap since previous trip - {scenario.name}", out_png)

    total = sum(s["trips"] for s in stats.values())
    print(f"\n{'gap band':16s} {'trips':>9s} {'share':>8s} {'median':>9s} "
          f"{'mean':>9s} {'p90':>9s} {'>cap':>7s}")
    for label, _lo, _hi in GAP_BANDS:
        s = stats[label]
        share = 100.0 * s["trips"] / total if total else 0.0
        print(f"  {label:14s} {s['trips']:>9,} {share:>7.1f}% "
              f"{s['median_km']:>9.2f} {s['mean_km']:>9.2f} {s['p90_km']:>9.2f} "
              f"{s['over_max_km']:>7,}")
    print(f"  {'ALL':14s} {total:>9,} {100.0:>7.1f}%")

    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
