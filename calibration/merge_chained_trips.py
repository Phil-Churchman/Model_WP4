"""
Merge chained segments into journeys, writing a new captured trip file.

Takes the per-pair verdicts from chain_segments.py and applies them to the
captured data, producing captured_locations/chained_trip_data.csv with the SAME
columns as source_trip_data.csv. Anything that reads the source file can read
this one instead:

    python calibration/chain_segments.py --scenario scenario.json
    python calibration/merge_chained_trips.py --scenario scenario.json
    python calibration/captured_trips_to_geojson.py \
        --source ../Model_data/nairobi/captured_locations/chained_trip_data.csv

source_trip_data.csv is never modified. It is the raw capture and stays that way.

Chaining is transitive
----------------------
If A chains to B and B chains to C, all three are one journey. So the merge is
not a pairwise operation: consecutive "chain" flags are followed to build runs,
and each run collapses to a single row. A journey of one segment passes through
unchanged, which is why the output is a drop-in replacement rather than a
different kind of file.

How the merged fields are derived
---------------------------------
    start_date/time, start_lat/lon   the FIRST segment's
    end_date/time,   end_lat/lon     the LAST segment's
    distance_km                      sum over segments -- the vehicle really
                                     covered all of it
    duration_min                     first start to last end. This is longer
                                     than the sum of segment durations, because
                                     the pauses between them are now inside the
                                     journey
    idle_time_min                    sum of segment idle PLUS those pauses,
                                     for the same reason
    avg_speed_kmh                    recomputed: distance / duration
    avg_moving_speed_kmh             recomputed: distance / (duration - idle)
    max_speed_kmh                    max over segments

Recomputing the speeds rather than averaging the originals matters: a mean of
means weighted by nothing is not the mean of the whole, and these segments have
very different lengths.

Segments with no verdict
------------------------
chain_segments.py only tests pairs within its --max-gap-min, and only where both
segments routed successfully. A pair with no verdict is left unchained, which is
the conservative choice: it keeps the data as captured rather than inventing a
join the geometry never supported.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from scenario_config import load_scenario, add_scenario_argument, writable_path

DEFAULT_CHAIN_RATIO = 0.95
# start_time alone is not unique: timestamps are whole minutes and a vehicle
# occasionally has two trips starting in the same one. end_time disambiguates,
# and a non-unique join key silently duplicates rows.
KEY = ["user_id", "start_date", "start_time", "end_time"]


def load_verdicts(path, chain_ratio):
    """
    Which (user, start_date, start_time) rows chain forward to the next segment.

    The threshold is applied here rather than trusting the `chain` column, so a
    different cut can be explored without re-running the routing.
    """
    pairs = pd.read_csv(path)
    # Checked against the names as they appear in the file, before renaming.
    needed = {"user_id", "a_start_date", "a_start_time", "a_end_time",
              "detour_ratio", "gap_min"}
    missing = needed - set(pairs.columns)
    if missing:
        raise SystemExit(
            f"{path} is missing column(s): {', '.join(sorted(missing))}\n"
            "Re-run calibration/chain_segments.py -- earlier versions did not "
            "record the identity of trip A.")

    pairs = pairs.rename(columns={"a_start_date": "start_date",
                                  "a_start_time": "start_time",
                                  "a_end_time": "end_time"})
    verdict = pairs[KEY + ["detour_ratio", "gap_min"]].copy()
    verdict["chain_next"] = verdict["detour_ratio"] >= chain_ratio
    return verdict.dropna(subset=["detour_ratio"])


def merge(source, verdict):
    """Collapse runs of chained segments into one row each."""
    df = source.copy()
    df["_row"] = np.arange(len(df))
    df = df.sort_values(["user_id", "start_date", "start_time",
                         "end_time"]).reset_index(drop=True)

    # A duplicated key would multiply rows through the merge below and inflate
    # every total, so it is checked rather than assumed.
    for label, frame in (("source", df), ("verdicts", verdict)):
        dupes = int(frame.duplicated(KEY).sum())
        if dupes:
            raise SystemExit(
                f"{dupes} {label} row(s) share a "
                f"(user_id, start_date, start_time, end_time) key. Merging on a "
                "non-unique key would duplicate trips; cannot continue.")

    # load_verdicts has already normalised the column names.
    df = df.merge(verdict[KEY + ["chain_next", "gap_min"]], on=KEY, how="left")
    df["chain_next"] = df["chain_next"].fillna(False).astype(bool)

    # A run breaks wherever the PREVIOUS row did not chain forward, or the
    # vehicle/day changes. cumsum over those breaks numbers the journeys.
    by_day = df.groupby(["user_id", "start_date"], sort=False)
    prev_chains = by_day["chain_next"].shift(1).fillna(False).astype(bool)
    new_day = by_day.cumcount() == 0
    df["journey"] = (new_day | ~prev_chains).cumsum()

    df["start_dt"] = pd.to_datetime(df["start_date"].astype(str) + " "
                                    + df["start_time"].astype(str), errors="coerce")
    df["end_dt"] = pd.to_datetime(df["end_date"].astype(str) + " "
                                  + df["end_time"].astype(str), errors="coerce")
    # The pause before this segment counts as internal idle only when the
    # segment was actually joined to the one before it.
    #
    # gap_min is keyed by trip A of each pair, so it is the gap FOLLOWING the
    # row it sits on. The pause preceding this row is therefore the PREVIOUS
    # row's gap_min, and it has to be shifted to line up -- using it unshifted
    # absorbed the wrong interval and left duration disagreeing with the
    # timestamp span on every multi-segment journey.
    prev_gap = by_day["gap_min"].shift(1)
    df["absorbed_gap_min"] = np.where(prev_chains, prev_gap.fillna(0.0), 0.0)

    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    g = df.groupby("journey", sort=True)

    out = pd.DataFrame({
        "user_id": g["user_id"].first(),
        "start_date": g["start_date"].first(),
        "start_time": g["start_time"].first(),
        "end_date": g["end_date"].last(),
        "end_time": g["end_time"].last(),
        "distance_km": g.apply(lambda x: pd.to_numeric(x["distance_km"],
                                                       errors="coerce").sum(),
                               include_groups=False),
        "start_lat": g["start_lat"].first(),
        "start_lon": g["start_lon"].first(),
        "end_lat": g["end_lat"].last(),
        "end_lon": g["end_lon"].last(),
        "max_speed_kmh": g.apply(lambda x: pd.to_numeric(x["max_speed_kmh"],
                                                         errors="coerce").max(),
                                 include_groups=False),
        "segments_merged": g.size(),
        "_first_start": g["start_dt"].min(),
        "_last_end": g["end_dt"].max(),
        "_segment_duration": g.apply(
            lambda x: pd.to_numeric(x["duration_min"], errors="coerce").fillna(0).sum(),
            include_groups=False),
        "_segment_idle": g.apply(lambda x: pd.to_numeric(x["idle_time_min"],
                                                        errors="coerce").fillna(0).sum(),
                                 include_groups=False),
        "_absorbed_gap": g["absorbed_gap_min"].sum(),
    }).reset_index(drop=True)

    # Duration from the segments' own durations plus the pauses now inside the
    # journey -- NOT recomputed from the timestamps.
    #
    # The two are identical by construction, since a gap is next_start minus
    # this_end. But duration_min carries sub-minute precision that whole-minute
    # timestamps cannot express: 169 source rows start and end in the same
    # minute with a real duration of 0.5-1.0 min. Deriving the span from the
    # timestamps rounds those to zero and produced 87 non-positive durations.
    out["duration_min"] = out["_segment_duration"] + out["_absorbed_gap"]
    out["idle_time_min"] = out["_segment_idle"] + out["_absorbed_gap"]
    # Kept for the cross-check below.
    out["_span_min"] = ((out["_last_end"] - out["_first_start"])
                        .dt.total_seconds() / 60.0)

    # Recomputed from the totals, never averaged from the segments.
    with np.errstate(divide="ignore", invalid="ignore"):
        out["avg_speed_kmh"] = np.where(out["duration_min"] > 0,
                                        out["distance_km"] / (out["duration_min"] / 60.0),
                                        np.nan)
        moving = out["duration_min"] - out["idle_time_min"]
        out["avg_moving_speed_kmh"] = np.where(moving > 0,
                                               out["distance_km"] / (moving / 60.0),
                                               np.nan)
    return out, df


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scenario_argument(p)
    p.add_argument("--source", default=None,
                   help="captured trip CSV (default: the scenario's "
                        "captured_locations/source_trip_data.csv)")
    p.add_argument("--pairs", default=None,
                   help="segment_chaining.csv from chain_segments.py "
                        "(default: in the scenario's output folder)")
    p.add_argument("--output", default=None,
                   help="merged CSV (default: the scenario's "
                        "captured_locations/chained_trip_data.csv)")
    p.add_argument("--chain-ratio", type=float, default=DEFAULT_CHAIN_RATIO,
                   help="detour ratio at or above which a pair is merged "
                        f"(default {DEFAULT_CHAIN_RATIO:g})")
    return p.parse_args()


def main():
    args = parse_args()
    scenario = load_scenario(args.scenario)

    src = args.source or os.path.join(scenario.captured_dir, "source_trip_data.csv")
    pairs_path = args.pairs or os.path.join(scenario.output_dir,
                                            "segment_chaining.csv")
    out_path = args.output or os.path.join(scenario.captured_dir,
                                          "chained_trip_data.csv")

    for label, path in (("source", src), ("pair verdicts", pairs_path)):
        if not os.path.exists(path):
            raise SystemExit(
                f"No {label} at {path}\n"
                + ("Run calibration/chain_segments.py first."
                   if label == "pair verdicts" else ""))
    if os.path.abspath(out_path) == os.path.abspath(src):
        raise SystemExit("Refusing to overwrite the source capture. "
                         "Choose a different --output.")

    print(f"Scenario : {scenario.name}")
    print(f"Source   : {src}")
    print(f"Verdicts : {pairs_path}")
    print(f"Threshold: detour ratio >= {args.chain_ratio:g}")

    source = pd.read_csv(src)
    columns = list(source.columns)
    verdict = load_verdicts(pairs_path, args.chain_ratio)
    merged, detail = merge(source, verdict)

    matched = int(detail["chain_next"].notna().sum())
    chained = int(detail["chain_next"].sum())
    print(f"\nSegments in    : {len(source):,}")
    print(f"  pairs with a verdict : {len(verdict):,}")
    print(f"  pairs merged         : {chained:,}")
    print(f"Journeys out   : {len(merged):,}  "
          f"({100.0 * len(merged) / len(source):.1f}% of the segment count)")

    counts = merged["segments_merged"].value_counts().sort_index()
    print(f"\n  journeys by number of segments merged:")
    for n, c in counts.head(8).items():
        print(f"    {n:>3d} segment(s): {c:>7,}")
    if len(counts) > 8:
        print(f"    more       : {int(counts.iloc[8:].sum()):>7,} "
              f"(largest {int(counts.index.max())})")

    print(f"\n{'':22s} {'source':>12s} {'chained':>12s}")
    for col, label in (("distance_km", "median distance km"),
                       ("duration_min", "median duration min"),
                       ("avg_speed_kmh", "median avg km/h"),
                       ("idle_time_min", "median idle min")):
        a = pd.to_numeric(source[col], errors="coerce").median()
        b = pd.to_numeric(merged[col], errors="coerce").median()
        print(f"  {label:20s} {a:>12.2f} {b:>12.2f}")
    # Duration must be positive, and should agree with the timestamp span.
    # Small disagreements are expected: duration_min has sub-minute precision
    # that whole-minute timestamps cannot hold. Large ones mean the source row
    # itself is inconsistent, which is worth naming rather than absorbing.
    nonpos = int((merged["duration_min"] <= 0).sum())
    drift = (merged["duration_min"] - merged["_span_min"]).abs()
    print(f"\n  durations <= 0             : {nonpos}"
          + ("" if nonpos == 0 else
             "   <- inherited from the source, not created here"))
    print(f"  under 1 min from the span  : "
          f"{int((drift < 1).sum()):,} of {len(drift):,}   (sub-minute rounding)")
    outliers = merged.loc[drift > 2, ["user_id", "start_date", "start_time",
                                      "end_date", "end_time", "duration_min"]]
    if len(outliers):
        print(f"  disagreeing by over 2 min  : {len(outliers)}   <- the source "
              "timestamps contradict its own duration_min here:")
        for r in outliers.head(5).itertuples():
            print(f"      user {r.user_id}  {r.start_date} {r.start_time} -> "
                  f"{r.end_date} {r.end_time}  but duration_min="
                  f"{r.duration_min:g}")
        print("    duration_min is used, being the only self-consistent value.")

    tot_a = pd.to_numeric(source["distance_km"], errors="coerce").sum()
    tot_b = pd.to_numeric(merged["distance_km"], errors="coerce").sum()
    print(f"  {'total distance km':20s} {tot_a:>12,.0f} {tot_b:>12,.0f}"
          f"   (must match: {abs(tot_a - tot_b) < 1})")

    # Same columns as the source, in the same order, so this is a drop-in
    # replacement; segments_merged is appended for provenance.
    merged = merged.drop(columns=[c for c in merged.columns if c.startswith("_")])
    ordered = [c for c in columns if c in merged.columns]
    extra = [c for c in merged.columns if c not in ordered]
    out_path = writable_path(out_path)
    merged[ordered + extra].to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(f"  columns: {len(ordered)} from the source, plus {extra}")
    print("\nUse it with:\n  venv\\Scripts\\python.exe "
          "calibration/captured_trips_to_geojson.py --source "
          f"\"{out_path}\"")


if __name__ == "__main__":
    main()
