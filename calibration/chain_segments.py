"""
Should two consecutive segments be chained into one journey?

The test is geometric, not just temporal. For each consecutive pair of trips by
the same vehicle, ask the router for the direct path from where the FIRST trip
started to where the SECOND trip ended, and compare it with what the vehicle
actually covered:

        detour_ratio = routed(A.start -> B.end)
                       -------------------------------------------------
                       routed(A) + routed(A.end -> B.start) + routed(B)

    ratio near 1   the intermediate point lies on the natural path anyway, so
                   stopping there cost nothing. The pause was incidental --
                   traffic, a light, a segmentation cut -- and the two segments
                   are one journey.

    ratio well
    below 1        reaching the intermediate point required going out of the
                   way. The driver had a reason to be there: it was a real
                   destination, most likely a drop-off. Do not chain.

    python calibration/chain_segments.py --scenario scenario.json

Which quantity the ratio is built from
--------------------------------------
The ratio is computed on TIME, not distance, because time is what the router
minimises. A path constrained to pass through an intermediate point can never
beat the unconstrained one on the optimised quantity, so the time ratio is
bounded by 1 and 1 - ratio is exactly the share of the journey that visiting the
intermediate point added.

Distance has no such guarantee under speed-based routing: a longer but faster
road can win, so the direct route may be further than the stitched one and the
distance ratio can exceed 1. It is reported as detour_ratio_distance for
interpretation, but the decision uses time.

Every leg -- A, B, the link between them, and the direct path -- is routed here,
with the graph and the road_speed_km-h in the scenario file NOW. None of the
distances in trip_routing_analysis.csv are reused: they were produced with
whatever speeds were in force when that file was written, and mixing weightings
silently breaks the comparison.

Why this is better than a time gap alone
----------------------------------------
A short gap is necessary but not sufficient. A taxi that drops a passenger and
immediately drives off to find another leaves a near-zero gap and yet the two
segments are genuinely different journeys with a real destination between them.
The detour test separates those cases: the drop-off shows as a detour, the
traffic light does not.

edge_overlap is a second, stricter measure: the fraction of the direct route's
edges that also appear in the stitched route. Two routes can agree on cost
without sharing a road, and the overlap catches that.
"""

import argparse
import importlib.util
import multiprocessing
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.dirname(HERE)
sys.path.insert(0, MODEL)
from scenario_config import load_scenario, add_scenario_argument, writable_path

DEFAULT_MAX_GAP_MIN = 10.0
DEFAULT_CHAIN_RATIO = 0.95     # at or above this, treat the pause as incidental
DEFAULT_MAX_SNAP_M = 2000.0

# Worker globals
CTG = None


def load_routing_module():
    """
    Import captured_trips_to_geojson for its graph builder and router.

    It parses argv at import time (its constants are read by its own pool
    workers), so argv is neutralised first -- otherwise this script's options
    would be handed to its parser.
    """
    saved = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        spec = importlib.util.spec_from_file_location(
            "ctg", os.path.join(HERE, "captured_trips_to_geojson.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = saved


def init_worker(route_graph, wgs_coords):
    global CTG
    CTG = load_routing_module()
    CTG.init_worker(route_graph, wgs_coords,
                    {"scale_times": False, "trip_type": "x", "out_dir": ".",
                     "source_columns": [], "segment_detail": False})


def measure(task):
    """
    One pair, every leg routed here under the SAME graph and weighting.

    A, B and the link are re-routed rather than read from
    trip_routing_analysis.csv. The CSV's distances were produced with whatever
    road_speed_km-h was in force when it was written, and route choice depends
    on those speeds -- mixing them with freshly routed distances made the ratio
    exceed 1, which is geometrically impossible under one weighting.

    Returns (index, dict of measurements) or (index, None) if any leg fails.
    """
    idx, a_start, a_end, b_start, b_end = task
    try:
        path_a, a_m, sec_a, ok_a, *_ = CTG.route(a_start, a_end)
        path_b, b_m, sec_b, ok_b, *_ = CTG.route(b_start, b_end)
        path_d, d_m, sec_d, ok_d, *_ = CTG.route(a_start, b_end)
        if a_end == b_start:
            link_m, link_s, ok_l = 0.0, np.zeros(0), True
        else:
            _p, link_m, link_s, ok_l, *_ = CTG.route(a_end, b_start)
    except Exception:
        return idx, None
    if not (ok_a and ok_b and ok_d and ok_l):
        return idx, None

    total = lambda arr: float(np.sum(arr)) if np.size(arr) else 0.0

    stitched_edges = set(zip(path_a[:-1], path_a[1:])) | \
                     set(zip(path_b[:-1], path_b[1:]))
    direct_edges = set(zip(path_d[:-1], path_d[1:]))
    overlap = (len(direct_edges & stitched_edges) / len(direct_edges)
               if direct_edges else None)

    return idx, {
        "a_routed_m": float(a_m), "b_routed_m": float(b_m),
        "link_routed_m": float(link_m), "direct_routed_m": float(d_m),
        "a_routed_s": total(sec_a), "b_routed_s": total(sec_b),
        "link_routed_s": total(link_s), "direct_routed_s": total(sec_d),
        "edge_overlap": overlap,
    }


def build_pairs(path, max_gap_min):
    """Consecutive same-vehicle, same-day pairs with a short gap."""
    df = pd.read_csv(path)
    needed = {"user_id", "start_date", "start_time", "end_date", "end_time",
              "start_lat", "start_lon", "end_lat", "end_lon",
              "routed_distance_m", "routed", "gap_from_previous_s"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing column(s): {', '.join(sorted(missing))}")

    df = df.sort_values(["user_id", "start_date", "start_time"]).reset_index(drop=True)
    df["start_dt"] = pd.to_datetime(df["start_date"].astype(str) + " "
                                    + df["start_time"].astype(str), errors="coerce")
    df["end_dt"] = pd.to_datetime(df["end_date"].astype(str) + " "
                                  + df["end_time"].astype(str), errors="coerce")

    key = ["user_id", "start_date"]
    by_day = df.groupby(key, sort=False)
    # B is the trip AFTER A, so A is the row whose successor exists in the day.
    nxt = {c: by_day[c].shift(-1) for c in
           ("start_lat", "start_lon", "end_lat", "end_lon",
            "routed_distance_m", "routed", "start_dt")}

    gap_min = (nxt["start_dt"] - df["end_dt"]).dt.total_seconds() / 60.0
    ok = (nxt["start_lat"].notna() & df["routed"].astype(bool)
          & nxt["routed"].fillna(False).astype(bool)
          & gap_min.notna() & (gap_min >= 0) & (gap_min <= max_gap_min))

    pairs = pd.DataFrame({
        "a_index": df.index[ok],
        # Semantic identity for trip A, so a consumer can join on something
        # meaningful rather than on a row position in a particular sort order.
        "user_id": df.loc[ok, "user_id"].to_numpy(),
        "a_start_date": df.loc[ok, "start_date"].to_numpy(),
        "a_start_time": df.loc[ok, "start_time"].to_numpy(),
        # end_time too: timestamps are whole minutes, so a vehicle occasionally
        # has two trips starting in the same minute and start_time alone does
        # not identify a row. Joining on a non-unique key duplicates rows.
        "a_end_time": df.loc[ok, "end_time"].to_numpy(),
        "b_start_date": by_day["start_date"].shift(-1)[ok].to_numpy(),
        "b_start_time": by_day["start_time"].shift(-1)[ok].to_numpy(),
        "gap_min": gap_min[ok].to_numpy(),
        "a_start_lat": df.loc[ok, "start_lat"].to_numpy(),
        "a_start_lon": df.loc[ok, "start_lon"].to_numpy(),
        "a_end_lat": df.loc[ok, "end_lat"].to_numpy(),
        "a_end_lon": df.loc[ok, "end_lon"].to_numpy(),
        "b_start_lat": nxt["start_lat"][ok].to_numpy(),
        "b_start_lon": nxt["start_lon"][ok].to_numpy(),
        "b_end_lat": nxt["end_lat"][ok].to_numpy(),
        "b_end_lon": nxt["end_lon"][ok].to_numpy(),
        "a_routed_m": df.loc[ok, "routed_distance_m"].to_numpy(),
        "b_routed_m": nxt["routed_distance_m"][ok].to_numpy(),
    }).reset_index(drop=True)
    return pairs, len(df)


def plot(pairs, threshold, title, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    ratio = pairs["detour_ratio"].dropna()
    axes[0].hist(ratio, bins=50, range=(0, 1), color="#2980b9",
                 edgecolor="white", linewidth=0.4)
    axes[0].axvline(threshold, color="#c0392b", lw=2,
                    label=f"chain at >= {threshold:g}")
    axes[0].set_title("Detour ratio\ndirect / (A + B)", fontsize=12,
                      fontweight="bold")
    axes[0].set_xlabel("ratio  (1 = stopping cost nothing)")
    axes[0].set_ylabel("pairs")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)

    order = np.sort(ratio)
    axes[1].plot(order, 100.0 * np.arange(1, order.size + 1) / order.size,
                 color="#2c3e50", lw=2)
    axes[1].axvline(threshold, color="#c0392b", lw=2)
    axes[1].set_title("Cumulative", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("detour ratio")
    axes[1].set_ylabel("% of pairs at or below")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.25)

    # Does the time gap tell you what the geometry tells you? If short gaps and
    # near-1 ratios were the same thing, this would be a tight diagonal.
    sample = pairs.dropna(subset=["detour_ratio"]).sample(
        min(6000, int(pairs["detour_ratio"].notna().sum())), random_state=0)
    axes[2].scatter(sample["gap_min"], sample["detour_ratio"], s=4, alpha=0.25,
                    color="#8e44ad", edgecolors="none")
    axes[2].axhline(threshold, color="#c0392b", lw=1.6)
    axes[2].set_title("Gap vs geometry", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("gap since previous trip (minutes)")
    axes[2].set_ylabel("detour ratio")
    axes[2].grid(alpha=0.25)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scenario_argument(p)
    p.add_argument("--input", default=None)
    p.add_argument("--out-csv", default=None,
                   help="per-pair results (default: segment_chaining.csv)")
    p.add_argument("--out-png", default=None,
                   help="figure (default: segment_chaining.png)")
    p.add_argument("--max-gap-min", type=float, default=DEFAULT_MAX_GAP_MIN,
                   help="only test pairs at most this far apart in time "
                        f"(default {DEFAULT_MAX_GAP_MIN:g})")
    p.add_argument("--chain-ratio", type=float, default=DEFAULT_CHAIN_RATIO,
                   help="detour ratio at or above which a pair is called one "
                        f"journey (default {DEFAULT_CHAIN_RATIO:g})")
    p.add_argument("--max-snap-m", type=float, default=DEFAULT_MAX_SNAP_M)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="test only the first N pairs, for a quick look")
    return p.parse_args()


def main():
    args = parse_args()
    scenario = load_scenario(args.scenario)
    src = args.input or os.path.join(scenario.trips_time_dir,
                                     "trip_routing_analysis.csv")
    if not os.path.exists(src):
        raise SystemExit(f"No analysis file at {src}")

    out_csv = args.out_csv or os.path.join(scenario.output_dir,
                                           "segment_chaining.csv")
    out_png = args.out_png or os.path.join(scenario.output_dir,
                                           "segment_chaining.png")

    print(f"Scenario : {scenario.name}")
    print(f"Analysis : {src}")

    pairs, n_trips = build_pairs(src, args.max_gap_min)
    if args.limit:
        pairs = pairs.head(args.limit)
    print(f"\nTrips: {n_trips:,}   candidate pairs (gap <= "
          f"{args.max_gap_min:g} min): {len(pairs):,}")
    if pairs.empty:
        raise SystemExit("No candidate pairs.")

    print(f"Loading road network ...")
    ctg = load_routing_module()
    route_graph, wgs_coords = ctg.build_route_graph(ctg.ROAD_NETWORK_FILE)
    print(f"  {route_graph.number_of_nodes():,} nodes, "
          f"{route_graph.number_of_edges():,} edges")
    # Stated explicitly because it is the thing that must not drift: every leg
    # below is routed with these speeds, read now, rather than with whatever was
    # in force when trip_routing_analysis.csv was written. Reported from the
    # routing module itself so this line cannot disagree with what it used.
    speeds = ctg.ROAD_SPEEDS
    print(f"  weight: {ctg.ROUTING_ATTR}   speeds: "
          f"{os.path.basename(ctg.ROAD_SPEEDS_SOURCE)} "
          f"({len(speeds)} types, e.g. secondary={speeds.get('secondary')}, "
          f"residential={speeds.get('residential')})")

    snap = ctg.build_snapper(wgs_coords)
    a_start, d1 = snap(pairs["a_start_lon"], pairs["a_start_lat"])
    a_end, d2 = snap(pairs["a_end_lon"], pairs["a_end_lat"])
    b_start, d3 = snap(pairs["b_start_lon"], pairs["b_start_lat"])
    b_end, d4 = snap(pairs["b_end_lon"], pairs["b_end_lat"])
    worst_snap = np.maximum.reduce([d1, d2, d3, d4])

    keep = worst_snap <= args.max_snap_m
    dropped = int((~keep).sum())
    pairs = pairs[keep].reset_index(drop=True)
    for name, arr in (("a_start", a_start), ("a_end", a_end),
                      ("b_start", b_start), ("b_end", b_end)):
        pairs[name] = np.asarray(arr)[keep]
    if dropped:
        print(f"  dropped {dropped:,} pairs with an endpoint over "
              f"{args.max_snap_m:.0f} m from the network")

    tasks = list(zip(pairs.index, pairs["a_start"], pairs["a_end"],
                     pairs["b_start"], pairs["b_end"]))
    print(f"Routing {len(tasks):,} pairs x 4 legs, all under this graph ...")

    workers = max(1, min(args.workers, len(tasks)))
    init_args = (route_graph, wgs_coords)
    if workers > 1:
        with multiprocessing.Pool(workers, initializer=init_worker,
                                  initargs=init_args) as pool:
            results = list(tqdm(pool.imap_unordered(measure, tasks, chunksize=200),
                                total=len(tasks), desc="Pairs"))
    else:
        init_worker(*init_args)
        results = [measure(t) for t in tqdm(tasks, desc="Pairs")]

    # Overwrite the CSV's distances: everything below must come from one graph.
    cols = ("a_routed_m", "b_routed_m", "link_routed_m", "direct_routed_m",
            "a_routed_s", "b_routed_s", "link_routed_s", "direct_routed_s",
            "edge_overlap")
    for c in cols:
        pairs[c] = np.nan
    for idx, got in results:
        if got:
            for c in cols:
                pairs.at[idx, c] = got[c]

    # A + (A.end -> B.start) + B: everything the vehicle covered.
    pairs["stitched_routed_m"] = (pairs["a_routed_m"] + pairs["link_routed_m"]
                                  + pairs["b_routed_m"])
    pairs["stitched_routed_s"] = (pairs["a_routed_s"] + pairs["link_routed_s"]
                                  + pairs["b_routed_s"])
    # Two ratios. The TIME one is the sound test: it is the quantity the router
    # minimises, so a path through an intermediate point can never beat the
    # direct one and the ratio is bounded by 1. The DISTANCE one is the intuitive
    # reading, but under speed-based routing distance is not subadditive -- a
    # longer, faster road can win -- so it can exceed 1 legitimately.
    pairs["detour_ratio"] = (pairs["direct_routed_s"]
                             / pairs["stitched_routed_s"].replace(0, np.nan))
    pairs["detour_ratio_distance"] = (pairs["direct_routed_m"]
                                      / pairs["stitched_routed_m"].replace(0, np.nan))
    pairs["extra_m_from_stopping"] = (pairs["stitched_routed_m"]
                                      - pairs["direct_routed_m"])
    pairs["extra_s_from_stopping"] = (pairs["stitched_routed_s"]
                                      - pairs["direct_routed_s"])
    pairs["chain"] = pairs["detour_ratio"] >= args.chain_ratio

    usable = pairs["detour_ratio"].notna()
    print(f"\nUsable pairs: {int(usable.sum()):,}")
    r = pairs.loc[usable, "detour_ratio"]
    print(f"\n{'detour ratio':18s} {'value':>8s}")
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        print(f"  p{q:<15d} {r.quantile(q / 100):>8.3f}")
    print(f"\n  pairs at ratio >= {args.chain_ratio:g} (chain): "
          f"{int(pairs['chain'].sum()):,} "
          f"({100.0 * pairs['chain'].sum() / usable.sum():.1f}% of usable)")
    print(f"  median extra distance from stopping: "
          f"{pairs.loc[usable, 'extra_m_from_stopping'].median():,.0f} m")

    print(f"\n  does a short gap imply one journey?")
    for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 10)):
        band = pairs[usable & (pairs["gap_min"] >= lo) & (pairs["gap_min"] < hi)]
        if len(band):
            print(f"    gap {lo}-{hi} min: {len(band):>7,} pairs, "
                  f"{100.0 * band['chain'].mean():>5.1f}% look like one journey "
                  f"(median ratio {band['detour_ratio'].median():.3f})")
    if pairs["edge_overlap"].notna().any():
        print(f"\n  median edge overlap of the direct route with the stitched one")
        print(f"    chained pairs : "
              f"{pairs.loc[pairs['chain'], 'edge_overlap'].median():.3f}")
        print(f"    others        : "
              f"{pairs.loc[~pairs['chain'], 'edge_overlap'].median():.3f}")

    # One weighting throughout, so a ratio above 1 is now impossible.
    above = int((pairs["detour_ratio"] > 1.0001).sum())
    print(f"\n  sanity: pairs with detour ratio > 1 (must be 0): {above}")

    out_csv = writable_path(out_csv)
    pairs.to_csv(out_csv, index=False)
    out_png = writable_path(out_png)
    plot(pairs, args.chain_ratio,
         f"Should consecutive segments be chained? - {scenario.name}", out_png)
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
