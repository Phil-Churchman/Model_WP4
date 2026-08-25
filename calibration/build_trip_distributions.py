"""
Derive trip_distributions.json from the chained capture.

Reads <folder_name>/captured_locations/chained_trip_data.csv -- one row per
observed journey, after over-segmented trips have been chained together by
chain_segments.py and merged by merge_chained_trips.py -- and writes the two
distributions that Simulation.py samples from when simulation_mode is
"distribution":

    distance_distribution   how far a trip goes, by road
    wait_distribution       how long a vehicle waits between fares

    python calibration/build_trip_distributions.py --scenario scenario_nairobi.json


================================================================================
WHY THE WAIT DISTRIBUTION IS NOT JUST THE OBSERVED GAPS
================================================================================

The simulation's cycle is

    fare wait -> pickup trip -> pickup wait -> passenger trip -> fare wait -> ...

so it has TWO kinds of wait, and they mean different things. The fare wait is
the vehicle looking for work. The pickup wait is boarding time once it has
arrived at the fare, and the simulation holds it fixed at pickup_wait_sec.

The capture cannot tell them apart. Each row is one leg, and consecutive legs
alternate empty-running-to-a-fare and carrying-that-fare, so the gaps between
them alternate fare wait, pickup wait, fare wait, pickup wait. Feeding all of
those gaps in as the fare-wait distribution would therefore be wrong in a
specific way: it would be a roughly 50/50 mixture of the thing being modelled
and a thing the model already represents separately, and the fare waits would
come out about half as long as they really are.

The separation used here is the simple one: assume every pickup wait is shorter
than every fare wait, so the shortest half of the observed gaps ARE the pickup
waits, and drop them. What is left is the fare-wait distribution.

    all gaps, sorted:  [.....pickup waits.....|.....fare waits.....]
                                              ^ cut at --pickup-share

The assumption is deliberately crude. The two distributions certainly overlap
in reality -- some fares are found immediately, some passengers are slow to
board -- so the cut misclassifies gaps near the boundary in both directions.
The consequence is one-sided and worth stating plainly: because every gap below
the cut is discarded, the fitted fare-wait distribution has NO mass below the
median of the observed gaps, and so is biased upward at the short end. Nothing
in the data identifies the two components separately, so a less crude split
would need an assumption about their shapes rather than a better measurement.

The discarded half is not thrown away silently. Its median is reported as an
empirical estimate of pickup_wait_sec, which is the scenario setting the same
assumption implies -- so the two halves of the split can be checked against each
other rather than only the half that is kept.

--pickup-share exists so the 50% can be moved. It is an assumption about how the
vehicles work, not a measurement, and a fleet that carries pre-booked fares
would have a different one.


================================================================================
BINNING
================================================================================
Bin edges are fixed rather than quantile-derived, so that re-running on more
data produces a file that can be diffed against the previous one. They are fine
where the mass is and widen into the tail, because equal-width bins over a
distribution this skewed either lose all resolution at the short end or leave
the long bins holding single-figure counts.

Timestamps in the capture are whole minutes, so gaps are quantised to 1 minute.
Wait bin edges are therefore kept on integer minutes: an edge at 2.5 min would
put two source values in one bin and one in the next, producing the alternating
peak-trough comb that equal-width bins over quantised data always produce.

Weights are raw counts. The loader normalises, so counts are as good as
probabilities and they let a reader see the sample size each bin rests on.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import (load_scenario, add_scenario_argument, writable_path,
                             TRIP_DISTRIBUTIONS_FILENAME, load_trip_distributions)

# Fine where the mass is, widening into the tail. In km.
DEFAULT_DISTANCE_EDGES = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
                          6.0, 8.0, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 40.0]
# Integer minutes throughout; see BINNING above.
DEFAULT_WAIT_EDGES = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240]

# A record whose average speed exceeds this contradicts itself -- the same cut
# captured_trips_to_geojson.py applies before routing.
DEFAULT_MAX_SPEED_KMH = 110.0
# Beyond this a gap is a break in the shift, not a wait for a fare.
DEFAULT_MAX_GAP_MIN = 240.0
# Fraction of observed gaps assumed to be pickup waits rather than fare waits.
DEFAULT_PICKUP_SHARE = 0.5


def load_journeys(path, max_speed_kmh):
    """The chained journeys, with the records that contradict themselves cut."""
    if not os.path.exists(path):
        raise SystemExit(
            f"No chained trip data at:\n  {path}\n"
            "Run calibration/chain_segments.py and calibration/"
            "merge_chained_trips.py first, or pass --source.")

    df = pd.read_csv(path)
    for col in ("user_id", "start_date", "start_time", "end_date", "end_time",
                "distance_km", "avg_speed_kmh"):
        if col not in df.columns:
            raise SystemExit(f"{path} has no {col!r} column. Columns: "
                             f"{list(df.columns)}")

    n_raw = len(df)
    kept = df[df["distance_km"].notna() & (df["distance_km"] > 0)]
    n_zero = len(df) - len(kept)
    before = len(kept)
    kept = kept[kept["avg_speed_kmh"] <= max_speed_kmh]
    n_fast = before - len(kept)

    print(f"Journeys: {n_raw:,} in the file")
    print(f"  dropped {n_zero:,} with no distance, "
          f"{n_fast:,} over {max_speed_kmh:g} km/h")
    print(f"  {len(kept):,} used")
    if kept.empty:
        raise SystemExit("Nothing left after filtering.")
    return kept.copy()


def observed_gaps(df, max_gap_min):
    """
    Minutes between one journey ending and the same vehicle's next starting.

    Keyed to the LATER journey -- it is the wait before that trip, not after it.
    Getting this backwards is a silent one-row misalignment, so the gap is
    computed against the previous row's end time directly rather than by
    shifting an already-computed column.

    Restricted to within a single day per vehicle: the first journey of a day
    has no measurable wait before it, only an overnight break.
    """
    df = df.assign(
        _start=pd.to_datetime(df["start_date"] + " " + df["start_time"]),
        _end=pd.to_datetime(df["end_date"] + " " + df["end_time"]),
    ).sort_values(["user_id", "_start"])

    prev_end = df.groupby(["user_id", "start_date"], sort=False)["_end"].shift(1)
    gaps = (df["_start"] - prev_end).dt.total_seconds() / 60.0

    n_first = int(gaps.isna().sum())
    finite = gaps[gaps.notna()]
    n_negative = int((finite < 0).sum())
    n_long = int((finite > max_gap_min).sum())
    kept = finite[(finite >= 0) & (finite <= max_gap_min)]

    print(f"Gaps: {len(finite):,} measurable "
          f"({n_first:,} first-of-day excluded)")
    print(f"  dropped {n_negative:,} negative (overlapping records), "
          f"{n_long:,} over {max_gap_min:g} min")
    print(f"  {len(kept):,} used")
    if kept.empty:
        raise SystemExit("No usable gaps.")
    return kept.to_numpy(dtype=float)


def split_pickup_waits(gaps, pickup_share):
    """
    Drop the shortest `pickup_share` of gaps as pickup waits.

    Cut by RANK, not by value. Gaps are quantised to whole minutes, so a great
    many are identical -- cutting at "<= the median" would drop every gap in the
    tie block straddling the median and so remove far more or far less than the
    share asked for. Splitting a tie block is arbitrary, but both parts land in
    the same histogram bin, so it costs nothing and the share comes out exact.
    """
    order = np.argsort(gaps, kind="stable")
    n_drop = int(round(len(gaps) * pickup_share))
    dropped, kept = gaps[order[:n_drop]], gaps[order[n_drop:]]
    if kept.size == 0:
        raise SystemExit(f"--pickup-share {pickup_share} discards every gap.")

    boundary = float(kept.min()) if kept.size else float("nan")
    tied = int((gaps == boundary).sum())
    tied_dropped = int((dropped == boundary).sum())

    print(f"\nPickup / fare split at the {pickup_share:.0%} point:")
    print(f"  assumed pickup waits : {len(dropped):,}  "
          f"median {np.median(dropped):.1f} min" if dropped.size else
          "  assumed pickup waits : 0")
    print(f"  fare waits kept      : {len(kept):,}  "
          f"median {np.median(kept):.1f} min")
    print(f"  cut lands at {boundary:g} min")
    if tied_dropped:
        print(f"    {tied:,} gaps share that value; {tied_dropped:,} of them fell "
              f"below the cut and were dropped")
    if dropped.size:
        print(f"  => pickup_wait_sec implied by the discarded half: "
              f"{np.median(dropped) * 60:.0f} s "
              f"({np.median(dropped):.1f} min)")
    return kept, dropped


def make_bins(values, edges, label):
    """Histogram `values` over `edges`, dropping empty bins."""
    edges = sorted(float(e) for e in edges)
    counts, _ = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)
    bins = [{"min": lo, "max": hi, "weight": int(c)}
            for lo, hi, c in zip(edges[:-1], edges[1:], counts) if c > 0]
    if not bins:
        raise SystemExit(f"Every {label} bin is empty.")

    over = int((np.asarray(values) > edges[-1]).sum())
    if over:
        print(f"  {label}: {over:,} value(s) above {edges[-1]:g} folded into the "
              f"top bin rather than dropped")
    empty = len(edges) - 1 - len(bins)
    if empty:
        print(f"  {label}: {empty} empty bin(s) omitted")
    return bins


def summarise(bins, scale, unit):
    """Weighted mean and median of the binned distribution, in display units."""
    lows = np.array([b["min"] for b in bins])
    highs = np.array([b["max"] for b in bins])
    w = np.array([b["weight"] for b in bins], dtype=float)
    mids = (lows + highs) / 2.0
    mean = float((w * mids).sum() / w.sum())
    cum = np.cumsum(w) / w.sum()
    median = float(mids[int(np.searchsorted(cum, 0.5))])
    return (f"{len(bins)} bins, {lows[0]:g}-{highs[-1]:g}{unit}, "
            f"mean {mean:.2f}{unit}, median ~{median:.2f}{unit}")


def parse_edges(raw, default):
    if raw is None:
        return default
    try:
        edges = [float(x) for x in raw.replace(",", " ").split()]
    except ValueError:
        raise SystemExit(f"Could not read bin edges from {raw!r}.")
    if len(edges) < 2:
        raise SystemExit("Need at least two bin edges.")
    return edges


def main():
    p = argparse.ArgumentParser(
        description="Derive trip_distributions.json from chained_trip_data.csv")
    add_scenario_argument(p)
    p.add_argument("--source", default=None,
                   help="chained trip CSV (default: the scenario's "
                        "captured_locations/chained_trip_data.csv)")
    p.add_argument("--out", default=None,
                   help=f"where to write (default: {TRIP_DISTRIBUTIONS_FILENAME} "
                        f"in the scenario folder)")
    p.add_argument("--pickup-share", type=float, default=DEFAULT_PICKUP_SHARE,
                   help="fraction of gaps assumed to be pickup rather than fare "
                        f"waits, dropped from the low end (default "
                        f"{DEFAULT_PICKUP_SHARE})")
    p.add_argument("--max-speed-kmh", type=float, default=DEFAULT_MAX_SPEED_KMH,
                   help="drop journeys averaging faster than this")
    p.add_argument("--max-gap-min", type=float, default=DEFAULT_MAX_GAP_MIN,
                   help="gaps longer than this are shift breaks, not fare waits")
    p.add_argument("--distance-edges", default=None,
                   help="bin edges in km, comma or space separated")
    p.add_argument("--wait-edges", default=None,
                   help="bin edges in minutes, comma or space separated")
    p.add_argument("--no-split", action="store_true",
                   help="use every gap as a fare wait, without removing pickup "
                        "waits. For comparison only -- see the module docstring")
    args = p.parse_args()

    if not 0.0 <= args.pickup_share < 1.0:
        raise SystemExit("--pickup-share must be at least 0 and below 1.")

    scenario = load_scenario(args.scenario).require_folder()
    source = args.source or os.path.join(scenario.captured_dir,
                                         "chained_trip_data.csv")
    out_path = args.out or os.path.join(scenario.folder,
                                        TRIP_DISTRIBUTIONS_FILENAME)

    print(f"Scenario : {scenario.path}")
    print(f"Source   : {source}\n")

    df = load_journeys(source, args.max_speed_kmh)
    distances_km = df["distance_km"].to_numpy(dtype=float)

    print()
    gaps = observed_gaps(df, args.max_gap_min)
    share = 0.0 if args.no_split else args.pickup_share
    if args.no_split:
        fare_waits, pickup_waits = gaps, np.empty(0)
        print("\n--no-split: every gap treated as a fare wait. The result mixes "
              "pickup waits\n  into the fare-wait distribution and will run "
              "short; see the docstring.")
    else:
        fare_waits, pickup_waits = split_pickup_waits(gaps, share)

    print("\nBinning:")
    distance_bins = make_bins(distances_km,
                              parse_edges(args.distance_edges, DEFAULT_DISTANCE_EDGES),
                              "distance")
    wait_bins = make_bins(fare_waits,
                          parse_edges(args.wait_edges, DEFAULT_WAIT_EDGES),
                          "wait")

    doc = {
        "_comment": [
            "Trip distributions for simulation_mode = 'distribution'.",
            "Generated by calibration/build_trip_distributions.py -- edit by hand",
            "only if you do not intend to regenerate.",
            "distance_distribution is the ROAD distance of an observed trip. The",
            "simulation divides a drawn band by deviation_factor (scenario.json)",
            "to get the straight-line radius it samples a target node inside.",
            "wait_distribution is the FARE wait only. The shortest",
            f"{share:.0%} of observed gaps are assumed to be pickup waits and are",
            "excluded, so there is no mass at the short end. See the script",
            "docstring for what that assumes and what it costs.",
            "'weight' is a raw count; the loader normalises.",
        ],
        "_provenance": {
            "generated_by": "calibration/build_trip_distributions.py",
            "source": os.path.relpath(source, scenario.folder).replace(os.sep, "/")
                      if source.startswith(scenario.folder) else source,
            "scenario": os.path.basename(scenario.path),
            "journeys_used": int(len(df)),
            "gaps_observed": int(len(gaps)),
            "gaps_used_as_fare_waits": int(len(fare_waits)),
            "pickup_share_removed": share,
            "pickup_wait_median_min": (round(float(np.median(pickup_waits)), 2)
                                       if pickup_waits.size else None),
            "filters": [
                f"avg_speed_kmh <= {args.max_speed_kmh:g}",
                "distance_km > 0",
                f"gap between 0 and {args.max_gap_min:g} min",
                "first journey of a vehicle-day excluded from gaps",
            ],
        },
        "distance_distribution": {"units": "km", "bins": distance_bins},
        "wait_distribution": {"units": "min", "bins": wait_bins},
    }

    out_path = writable_path(out_path)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(doc, indent=2) + "\n")

    print(f"\nWritten: {out_path}")
    print(f"  distance: {summarise(distance_bins, 1.0, ' km')}")
    print(f"  wait    : {summarise(wait_bins, 1.0, ' min')}")
    if pickup_waits.size:
        print(f"\nSet pickup_wait_sec in {os.path.basename(scenario.path)} to "
              f"about {np.median(pickup_waits) * 60:.0f} to match the half this "
              f"treated as boarding time.")

    # Read it back through the loader the simulation uses, so a file that would
    # fail at the start of a run fails here instead.
    bands, _ = load_trip_distributions(scenario, path=out_path)
    print(f"\nReloaded through load_trip_distributions: "
          f"distance {bands['distance'].describe(1000.0, ' km')}; "
          f"wait {bands['wait'].describe(60.0, ' min')}")


if __name__ == "__main__":
    main()
