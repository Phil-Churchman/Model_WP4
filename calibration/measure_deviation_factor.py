"""
Measure the deviation factor a simulation run actually exhibits.

The deviation factor is the ratio of the distance a vehicle drives to the
straight-line distance between where it started and where it ended:

    deviation factor  =  road distance / crow-flies distance

It is a property of the road network, not of the fleet, so it has to be measured
per city rather than carried over. Distribution mode uses it to convert a drawn
ROAD distance into the straight-line radius it samples a destination node
inside, so setting it wrong shifts every trip length in the run.

    python calibration/measure_deviation_factor.py --scenario scenario_accra.json

Reads the agent tracks from <folder_name>/output/output_trips_time_queued: each
trip is a LineString whose length_m property is the routed road distance, and
whose first and last coordinates give the straight line. Any run produces these,
so the measurement does not depend on which mode was simulated.


================================================================================
WHICH NUMBER TO USE
================================================================================
Three summaries are reported, and they do not agree. That is not noise, it is
what a ratio distribution with a small denominator does:

  median      The robust one, and the one to put in the scenario. Half the trips
              are more circuitous than this and half less.

  mean        Inflated, sometimes badly. A trip that doubles back or loops ends
              up near where it started, so its straight-line distance is small
              relative to the distance driven and its ratio is large. The
              arithmetic mean of the ratios is pulled by that tail rather than
              describing a typical trip: on Accra it reads 1.94 against a
              median of 1.50, with a 90th percentile of 2.95.

  aggregate   Total road distance divided by total straight-line distance. This
              is the factor that makes the FLEET's total mileage come out right,
              which is a different question from making a typical trip come out
              right. It weights by straight-line distance, so it follows
              whichever bands carry the most of it -- on Accra the middle
              distances, which are also the most circuitous, putting it above
              the median at 1.62. Do not assume it lands on either side.

--min-straight-m guards the mean: trips whose endpoints are closer together
than this are excluded from all three statistics, because their ratio is a
division by nearly zero. How much it is doing is always reported, and on a
distribution-mode run the answer is usually not much -- the sampler never aims
at the node it is already on, so only about 2% of trips fall under 100 m and the
median moves by 0.01. It earns its place on captured data and on modes that
produce loops, not here. Raising it to 500 m does move the mean (1.94 to 1.72),
which is worth knowing before quoting a mean from a filtered set.

The breakdown by distance band is worth reading before adopting a single value,
because the factor is not constant with trip length, and not in the direction
one might assume. Measured on Accra it is lowest at BOTH ends and peaks in the
middle: 1.33 under 500 m, rising to 1.63 at 2-5 km, falling back to 1.24 above
20 km. A very short trip is often a single straight segment, and a very long one
spends most of its length on trunk roads that run direct; it is the middle
distances that have to work around the network. The script reports which band is
highest rather than assuming a direction, since the shape belongs to each city.

================================================================================
ASSUMED VERSUS REALISED
================================================================================
This is the check the script exists for. Distribution mode ASSUMES a factor when
it picks a destination: it draws a road distance d, divides by deviation_factor
to get a straight-line radius d/f, and picks a node there. What the vehicle then
actually drives is whatever the network gives, so the REALISED ratio r is a
property of the roads, not of f. Comparing the two says whether the assumption
held.

The consequence of a gap is direct. A trip drawn at d ends up covering

    road_actual  =  (d / f) * r  =  d * (r / f)

so r/f is the factor by which every trip length is stretched or shrunk against
the distribution that was asked for. r/f above 1 means trips came out long and f
should be raised; below 1, the reverse. Setting f to the realised median is one
step of that iteration, and it converges quickly.

Exact agreement is not expected, and a small gap is not a defect:

  - Circuity varies with distance, so which bands the run happens to visit
    moves the overall figure. The per-band table shows the spread directly.
  - The sampler takes the first candidate in the annulus that routes and stays
    within the range budget, so where the annulus is thin or the budget tight
    the node it settles on is not a free draw from that radius.
  - Nothing constrains the ratio itself at any point, only the radius.

Only distribution mode assumes a factor. Run against a hail-rank or demand-model
run the script still measures circuity, but there is no assumption to compare
it against, and the numbers are not comparable across modes -- hail trips are
truncated part-way along a route, which lowers the ratio on its own.
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import (load_scenario, add_scenario_argument, writable_path,
                             simulation_mode, DISTRIBUTION)

EARTH_RADIUS_M = 6371008.8

# Below this the endpoints are close enough that the ratio is meaningless.
DEFAULT_MIN_STRAIGHT_M = 100.0
# Bands to break the result down by, in metres of ROAD distance.
DEFAULT_BANDS = [0, 500, 1000, 2000, 5000, 10000, 20000, float("inf")]


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres between two WGS84 points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def read_trips(run_dir, max_agents=None):
    """
    Every routed trip in the run, as (type, road_m, straight_m).

    A Point feature is a wait, not a trip, and a LineString with fewer than two
    distinct coordinates never went anywhere; both are skipped rather than
    counted as zero-length trips, which would otherwise pull every statistic
    down without appearing in the drop counts.
    """
    if not os.path.isdir(run_dir):
        raise SystemExit(f"No run output at:\n  {run_dir}\nRun the simulation first.")

    files = sorted(f for f in os.listdir(run_dir) if f.endswith(".geojson"))
    if max_agents:
        files = files[:max_agents]
    if not files:
        raise SystemExit(f"No agent files in {run_dir}.")

    trips, n_points, n_degenerate = [], 0, 0
    for name in files:
        with open(os.path.join(run_dir, name), "r", encoding="utf-8") as f:
            feats = json.load(f).get("features", [])
        for ft in feats:
            geom = ft.get("geometry") or {}
            if geom.get("type") != "LineString":
                n_points += 1
                continue
            coords = geom.get("coordinates") or []
            props = ft.get("properties", {})
            road = float(props.get("length_m", 0.0) or 0.0)
            if len(coords) < 2 or road <= 0:
                n_degenerate += 1
                continue
            (lon1, lat1), (lon2, lat2) = coords[0], coords[-1]
            trips.append((props.get("type", "?"), road,
                          haversine_m(lon1, lat1, lon2, lat2)))

    print(f"Run: {len(files)} agent file(s)")
    print(f"  {len(trips):,} routed trips, {n_points:,} waits skipped, "
          f"{n_degenerate:,} zero-length trips skipped")
    if not trips:
        raise SystemExit("No routed trips found.")
    return trips


def summarise(road, straight, label, n_excluded=None):
    """The three summaries, for one group of trips."""
    if road.size == 0:
        return None
    ratios = road / straight
    return {
        "group": label,
        "trips": int(road.size),
        "excluded": n_excluded,
        "median": float(np.median(ratios)),
        "mean": float(ratios.mean()),
        "aggregate": float(road.sum() / straight.sum()),
        "p10": float(np.percentile(ratios, 10)),
        "p90": float(np.percentile(ratios, 90)),
        "road_km": float(road.sum() / 1000.0),
    }


def band_label(lo, hi):
    if math.isinf(hi):
        return f"{lo / 1000:g}+ km"
    return f"{lo / 1000:g}-{hi / 1000:g} km"


def print_table(rows, title, first_col="group", width=22):
    print(f"\n{title}")
    print(f"  {first_col:<{width}} {'trips':>7} {'median':>8} {'mean':>8} "
          f"{'aggregate':>10} {'p10':>7} {'p90':>7}")
    print("  " + "-" * (width + 50))
    for r in rows:
        print(f"  {r['group']:<{width}} {r['trips']:>7,} {r['median']:>8.2f} "
              f"{r['mean']:>8.2f} {r['aggregate']:>10.2f} "
              f"{r['p10']:>7.2f} {r['p90']:>7.2f}")


def main():
    p = argparse.ArgumentParser(
        description="Report the road / straight-line distance ratio of a run")
    add_scenario_argument(p)
    p.add_argument("--run-dir", default=None,
                   help="agent tracks (default: the scenario's "
                        "output/output_trips_time_queued)")
    p.add_argument("--min-straight-m", type=float, default=DEFAULT_MIN_STRAIGHT_M,
                   help="exclude trips whose endpoints are closer than this; "
                        "their ratio is division by nearly zero "
                        f"(default {DEFAULT_MIN_STRAIGHT_M:g})")
    p.add_argument("--types", default=None,
                   help="comma-separated trip types to include "
                        "(default: all, e.g. 'pickup,passenger')")
    p.add_argument("--csv", default=None,
                   help="write the breakdown to this CSV")
    p.add_argument("--max-agents", type=int, default=None,
                   help="read only the first N agent files")
    args = p.parse_args()

    scenario = load_scenario(args.scenario).require_folder()
    run_dir = args.run_dir or scenario.trips_time_dir
    print(f"Scenario : {scenario.path}")
    print(f"Run      : {run_dir}\n")

    trips = read_trips(run_dir, args.max_agents)

    wanted = None
    if args.types:
        wanted = {t.strip() for t in args.types.split(",") if t.strip()}
        trips = [t for t in trips if t[0] in wanted]
        if not trips:
            raise SystemExit(f"No trips of type {sorted(wanted)}.")

    kinds = np.array([t[0] for t in trips])
    road_all = np.array([t[1] for t in trips], dtype=float)
    straight_all = np.array([t[2] for t in trips], dtype=float)

    keep = straight_all >= args.min_straight_m
    n_excluded = int((~keep).sum())
    if not keep.any():
        raise SystemExit(
            f"Every trip has endpoints closer than {args.min_straight_m:g} m. "
            "Lower --min-straight-m.")

    kinds, road, straight = kinds[keep], road_all[keep], straight_all[keep]
    print(f"  {n_excluded:,} trip(s) excluded: endpoints under "
          f"{args.min_straight_m:g} m apart "
          f"({100 * n_excluded / len(road_all):.1f}%)")

    overall = summarise(road, straight, "ALL", n_excluded)

    assumed = scenario.cfg.get("deviation_factor")
    mode = simulation_mode(scenario)

    print("\n" + "=" * 66)
    print(f"  REALISED deviation factor   median {overall['median']:.2f}   "
          f"mean {overall['mean']:.2f}")
    print("=" * 66)
    print(f"  aggregate (total road / total straight): {overall['aggregate']:.2f}")
    print(f"  10th-90th percentile: {overall['p10']:.2f} - {overall['p90']:.2f}")
    print(f"  over {overall['trips']:,} trips, {overall['road_km']:,.0f} km driven")

    # The whole point: did the factor the sampler assumed hold up once the
    # trips were actually routed? Only distribution mode assumes one.
    if mode == DISTRIBUTION and assumed:
        stretch = overall["median"] / float(assumed)
        print(f"\n  ASSUMED (scenario deviation_factor) : {float(assumed):.2f}")
        print(f"  REALISED (median of routed trips)   : {overall['median']:.2f}")
        print(f"  ratio realised / assumed            : {stretch:.3f}")
        pct = 100 * (stretch - 1)
        if abs(pct) < 5:
            print(f"  -> trips came out {abs(pct):.1f}% "
                  f"{'longer' if pct > 0 else 'shorter'} than the distribution asked for,")
            print("     which is inside the spread the distance bands alone "
                  "produce. No change indicated.")
        else:
            direction = "longer" if pct > 0 else "shorter"
            print(f"  -> every trip is coming out about {abs(pct):.0f}% "
                  f"{direction} than the distribution asked for.")
            print(f"     Set deviation_factor to {overall['median']:.2f} and re-run; "
                  f"the correction converges in a step or two.")
    elif mode == DISTRIBUTION:
        print(f"\n  No deviation_factor set, so the sampler assumed 1.00 and made no "
              f"allowance\n  for circuity. Set it to {overall['median']:.2f}.")
    else:
        print(f"\n  {os.path.basename(scenario.path)} is in {mode!r} mode, which does not "
              f"use deviation_factor\n  to place destinations, so there is no "
              f"assumption to check this against.")

    by_type = []
    for kind in sorted(set(kinds.tolist())):
        m = kinds == kind
        row = summarise(road[m], straight[m], kind)
        if row:
            by_type.append(row)
    if len(by_type) > 1:
        print_table(by_type, "By trip type:", "type")

    by_band = []
    for lo, hi in zip(DEFAULT_BANDS[:-1], DEFAULT_BANDS[1:]):
        m = (road >= lo) & (road < hi)
        row = summarise(road[m], straight[m], band_label(lo, hi))
        if row:
            by_band.append(row)
    if len(by_band) > 1:
        print_table(by_band, "By road distance (the factor is not constant):",
                    "road distance")

    print(f"\nRecommended deviation_factor for "
          f"{os.path.basename(scenario.path)}: {overall['median']:.2f}")
    if len(by_band) > 1:
        hi = max(by_band, key=lambda r: r["median"])
        lo = min(by_band, key=lambda r: r["median"])
        spread = hi["median"] - lo["median"]
        if spread > 0.2:
            # Reported rather than asserted: which band is most circuitous is a
            # property of the city, and on Accra it is the middle distances, not
            # the short ones as one might assume.
            print(f"  Note: the median ranges {spread:.2f} across distance "
                  f"bands -- highest at {hi['group']} ({hi['median']:.2f}), "
                  f"lowest at {lo['group']} ({lo['median']:.2f}).")
            print(f"  One factor cannot serve them all; the bands furthest from "
                  f"{overall['median']:.2f} are the ones "
                  f"compare_trip_distributions.py will show as mis-served.")

    rows = [overall] + by_type + by_band
    out_csv = args.csv or os.path.join(scenario.output_dir,
                                       "deviation_factor.csv")
    out_csv = writable_path(out_csv)
    fields = ["group", "trips", "median", "mean", "aggregate", "p10", "p90",
              "road_km", "excluded"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items() if k in fields})
    print(f"\nWritten: {out_csv}")


if __name__ == "__main__":
    main()
