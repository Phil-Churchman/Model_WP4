"""
Turn captured GPS trip records into per-vehicle GeoJSON tracks.

Reads <folder_name>/captured_locations/source_trip_data.csv -- one row per
observed trip, with a start/end timestamp and a start/end lat-lon -- snaps each
endpoint to the road network and routes between them with the same weights the
simulation uses, then writes one file per user_id in the format the simulation
produces (<folder_name>/output/output_trips_time_queued/agent_XXXX_time.geojson).

The folder name comes from scenario.json and the per-highway-type speeds from
road_speeds.json, exactly as they do for the simulation. Unlike
the simulation, the timeline here is the recorded one: start_time/end_time come
straight from the CSV (the data spans whole weeks, not scenario.json's single
day), and the routed per-hop times are rescaled so they add up to the duration
that was actually observed. Gaps between consecutive trips become "idle" points,
via the same fill_gaps helper the simulation uses.

Usage:
    python "captured_trips_to_geojson.py"                 # all users
    python "captured_trips_to_geojson.py" --limit-users 5 --out-dir <path>
"""

import argparse
import json
import multiprocessing
import os
import sys

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import (load_scenario, add_scenario_argument,
                             load_road_speeds, road_speeds_ms)

from fill_gaps import insert_idle_points

# ============================================================
# CONFIGURATION
# ============================================================
# Resolved at module level, not in main(), because the routing constants below
# are read inside the pool workers, which re-import this module under spawn.
# export=True publishes the resolved path so those re-imports agree with us.
_parser = argparse.ArgumentParser(add_help=False)
add_scenario_argument(_parser)
_args, _ = _parser.parse_known_args()
SCENARIO = load_scenario(_args.scenario, export=True)
scenario_cfg = SCENARIO.cfg

FOLDER_NAME = SCENARIO.folder_name
# From road_speeds.json, not the scenario file -- and via the same helper the
# simulation uses, so the two cannot drift apart. Routing weights that differ
# between the two produce routes that look reasonable and are not comparable.
ROAD_SPEEDS, ROAD_SPEEDS_SOURCE = load_road_speeds(SCENARIO, export=True)
ROAD_SPEEDS_MS = road_speeds_ms(ROAD_SPEEDS)
SPEED_BASED_ROUTING = scenario_cfg.get("speed_based_routing", False)
# Same reasoning as the simulation: the raw OSM graph carries no "travel_time",
# so both weights are materialised explicitly rather than left to default to 1.
ROUTING_ATTR = "travel_time_s" if SPEED_BASED_ROUTING else "length_m"
MIN_WEIGHT = 1e-6

# DEFAULT_SOURCE_CSV = os.path.join(SCENARIO.captured_dir, "source_trip_data.csv")
DEFAULT_SOURCE_CSV = os.path.join(SCENARIO.captured_dir, "chained_trip_data.csv")
ROAD_NETWORK_FILE = os.path.join(SCENARIO.input_dir, "roads.graphml")
DEFAULT_OUTPUT_DIR = SCENARIO.trips_time_dir

# An endpoint further than this from any node is off the extracted network, so
# the trip is dropped rather than dragged to a road it never used.
DEFAULT_MAX_SNAP_M = 2000.0
# Above this AVERAGE speed the record contradicts itself; see load_trips.
DEFAULT_MAX_SPEED_KMH = 110.0

POOL_WORKERS = max(1, min(int(scenario_cfg.get("worker_processes", 4)),
                          multiprocessing.cpu_count()))

# Worker globals
ROUTE_GRAPH = None
WGS_COORDS = None
OPTS = None


# ============================================================
# ROAD NETWORK
# ============================================================

def build_route_graph(graphml_path):
    """
    Collapse the OSM MultiDiGraph into one metric DiGraph carrying a single
    length/travel-time pair per (u, v) -- the fastest of any parallel edges.

    The graph is left unprojected: "length" is already metres, and only the
    endpoint snapping below needs a metric coordinate system.
    """
    G = ox.load_graphml(graphml_path)

    edge_lookup = {}
    for u, v, _k, data in G.edges(keys=True, data=True):
        hw = data.get("highway")
        # OSM sometimes tags several classes on one way. The fastest is what
        # governs the routing, so it is also the class the edge is attributed
        # to in the per-highway breakdown -- one rule, used for both.
        if isinstance(hw, list):
            highway = max(hw, key=lambda h: ROAD_SPEEDS_MS.get(h, 0.1))
        else:
            highway = hw if isinstance(hw, str) else "unknown"
        speed = ROAD_SPEEDS_MS.get(highway, 0.1)
        # Weights must stay strictly positive: a road type configured at 0 km/h
        # would otherwise divide by zero.
        speed = max(float(speed), 0.1)
        length = max(float(data.get("length", 1.0)), MIN_WEIGHT)
        travel_time = max(length / speed, MIN_WEIGHT)
        prev = edge_lookup.get((u, v))
        if prev is None or travel_time < prev[1]:
            edge_lookup[(u, v)] = (length, travel_time, highway, speed)

    route_graph = nx.DiGraph()
    route_graph.add_nodes_from(G.nodes())
    route_graph.add_edges_from(
        (u, v, {"length_m": length, "travel_time_s": travel_time,
                "highway": highway, "speed_m_s": speed})
        for (u, v), (length, travel_time, highway, speed) in edge_lookup.items()
    )

    wgs_coords = {n: (float(d["x"]), float(d["y"])) for n, d in G.nodes(data=True)}
    return route_graph, wgs_coords


def build_snapper(wgs_coords):
    """KDTree over the nodes in a local UTM, so snap distances come out in metres."""
    node_ids = list(wgs_coords)
    lons = np.array([wgs_coords[n][0] for n in node_ids])
    lats = np.array([wgs_coords[n][1] for n in node_ids])

    zone = int((float(lons.mean()) + 180.0) // 6.0) + 1
    epsg = (32600 if lats.mean() >= 0 else 32700) + zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    xs, ys = transformer.transform(lons, lats)
    tree = cKDTree(np.column_stack([xs, ys]))
    node_ids = np.array(node_ids)

    def snap(lon_values, lat_values):
        """(nearest node id, distance in metres) for each lon/lat pair."""
        px, py = transformer.transform(np.asarray(lon_values), np.asarray(lat_values))
        dist, idx = tree.query(np.column_stack([px, py]))
        return node_ids[idx], dist

    return snap


# ============================================================
# TRIP DATA
# ============================================================

def load_trips(csv_path, max_speed_kmh=None):
    """
    Read the captured trips and drop the rows that cannot be routed.

    Returns (DataFrame, dict of drop counts by reason, source column names).
    The column names are captured before any derived columns are added, so the
    analysis file can reproduce the source row in its original order.
    """
    df = pd.read_csv(csv_path)
    source_columns = list(df.columns)
    dropped = {}
    before = len(df)

    df["start_dt"] = pd.to_datetime(df["start_date"].astype(str) + " "
                                    + df["start_time"].astype(str), errors="coerce")
    df["end_dt"] = pd.to_datetime(df["end_date"].astype(str) + " "
                                  + df["end_time"].astype(str), errors="coerce")
    df = df.dropna(subset=["start_dt", "end_dt"])
    dropped["unparseable timestamp"] = before - len(df)

    df = df.sort_values(["user_id", "start_dt"]).reset_index(drop=True)

    # Idle time between consecutive trips of the same vehicle: previous trip's
    # end to this trip's start, NaN for a vehicle's first trip.
    #
    # Computed here, before any coordinate-based dropping, so the gap spans the
    # trip that actually preceded this one. Measuring it after the drops would
    # silently absorb a discarded trip into the gap and report a vehicle as idle
    # while it was in fact moving.
    df["gap_from_previous_s"] = (
        df["start_dt"] - df.groupby("user_id")["end_dt"].shift(1)
    ).dt.total_seconds()

    # Blank endpoints are a session-boundary artefact of the capture and always
    # come in pairs: a session's last trip has no recorded destination and the
    # next session's first trip has no recorded origin. There is therefore never
    # a neighbouring position to borrow -- both ends of the pair are blank -- so
    # these rows are simply dropped (~4% of the Nairobi capture).
    before = len(df)
    df = df.dropna(subset=["start_lat", "start_lon", "end_lat", "end_lon"]).copy()
    dropped["missing coordinates"] = before - len(df)

    # Prefer the timestamps; fall back to the reported duration when they are
    # equal or reversed (a handful of rows round to the same minute).
    duration = (df["end_dt"] - df["start_dt"]).dt.total_seconds()
    fallback = pd.to_numeric(df["duration_min"], errors="coerce") * 60.0
    df["duration_s"] = duration.where(duration > 0, fallback)

    before = len(df)
    df = df[df["duration_s"] > 0]
    dropped["non-positive duration"] = before - len(df)

    # Physically impossible average speeds mean the record is wrong somewhere --
    # a mis-stamped duration, a GPS jump, a distance that belongs to another
    # trip. There is no way to tell which field is at fault, so the trip is
    # dropped rather than half-trusted: it would otherwise pull the speed
    # calibration and inflate the distance totals.
    #
    # This is the trip's AVERAGE speed, not max_speed_kmh. A 120 km/h peak is a
    # plausible moment; a 120 km/h average over a Nairobi trip is not.
    if max_speed_kmh:
        speed_kmh = (pd.to_numeric(df["distance_km"], errors="coerce")
                     / (df["duration_s"] / 3600.0))
        before = len(df)
        df = df[~(speed_kmh > max_speed_kmh)]      # keeps NaN speeds
        dropped[f"average speed over {max_speed_kmh:g} km/h"] = before - len(df)

    return df.reset_index(drop=True), dropped, source_columns


# ============================================================
# ROUTING & FEATURE BUILDING
# ============================================================

def init_worker(route_graph, wgs_coords, opts):
    global ROUTE_GRAPH, WGS_COORDS, OPTS
    ROUTE_GRAPH, WGS_COORDS, OPTS = route_graph, wgs_coords, opts


def route(u, v):
    """
    Node path u -> v with its per-hop metres and seconds, and how those metres
    are split across highway types.

    Returns (path, length_m, hop_secs, routed, by_highway, hop_m, hop_highway).
    `routed` is False when the two endpoints are in different components, in
    which case the caller gets a straight-line stand-in rather than losing the
    trip.

    by_highway maps highway type -> {distance_m, speed_kmh, time_s}. The speed is
    the one from scenario.json that actually governed the routing, so the times
    sum back to the trip's routed time -- the breakdown is a decomposition of
    the result, not a second estimate of it.

    hop_m and hop_highway are the per-segment metres and highway type, aligned
    with hop_secs and with the coordinate pairs of the LineString.
    """
    if u == v:
        return [u], 0.0, np.empty(0), True, {}, np.empty(0), []
    try:
        _, path = nx.bidirectional_dijkstra(ROUTE_GRAPH, u, v, weight=ROUTING_ATTR)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [u, v], 0.0, np.empty(0), False, {}, np.empty(0), []

    hop_m = np.empty(len(path) - 1)
    hop_s = np.empty(len(path) - 1)
    hop_highway = []
    by_highway = {}
    for i in range(len(path) - 1):
        data = ROUTE_GRAPH[path[i]][path[i + 1]]
        hop_m[i] = data["length_m"]
        hop_s[i] = data["travel_time_s"]
        hop_highway.append(data["highway"])
        entry = by_highway.setdefault(
            data["highway"], {"distance_m": 0.0, "speed_kmh": data["speed_m_s"] * 3.6})
        entry["distance_m"] += data["length_m"]

    for entry in by_highway.values():
        # Time from the full-precision distance and speed, then round -- deriving
        # it from the already-rounded values let a tenth of a metre per type
        # accumulate into most of a second on long trips.
        entry["time_s"] = round(entry["distance_m"] / (entry["speed_kmh"] / 3.6), 1)
        entry["distance_m"] = round(entry["distance_m"], 1)
        entry["speed_kmh"] = round(entry["speed_kmh"], 2)

    return path, float(hop_m.sum()), hop_s, True, by_highway, hop_m, hop_highway


def build_feature(agent_id, trip, odometer):
    """
    One CSV row -> one GeoJSON feature, in the simulation's output format, plus
    the routing metrics for the per-trip analysis file.
    """
    (path, length_m, hop_secs, routed, by_highway,
     hop_m, hop_highway) = route(trip["start_node"], trip["end_node"])
    coords = [WGS_COORDS[n] for n in path]
    duration_s = float(trip["duration_s"])

    # Captured before hop_secs is rescaled below: this is what the routing
    # algorithm says the trip takes at the scenario's speeds, which is the
    # quantity the analysis file is asking for. The observed duration is
    # already in the source columns alongside it.
    routed_time_s = float(hop_secs.sum()) if hop_secs.size else 0.0

    # Every segment keeps the speed ratio scenario.json gives its highway type;
    # the whole profile is then multiplied by one factor so the journey lasts
    # exactly as long as it really did. A segment's effective speed is therefore
    # its configured speed times `speed_scale`, identical for every segment of
    # the trip -- the shape of the speed profile is the scenario's, only its
    # level is set by the recording.
    speed_scale = 1.0
    if OPTS["scale_times"] and hop_secs.size:
        # The recorded duration is ground truth; the modelled free-flow times only
        # say how that duration was spread along the route.
        total = float(hop_secs.sum())
        if total > 0:
            hop_secs = hop_secs * (duration_s / total)
            speed_scale = total / duration_s
        else:
            hop_secs = np.full(hop_secs.size, duration_s / hop_secs.size)
    elif not routed:
        hop_secs = np.array([duration_s])

    # Effective metres per second on each segment, from the scaled times, so the
    # speeds and the times in the feature are always consistent with each other
    # and with the geometry.
    with np.errstate(divide="ignore", invalid="ignore"):
        hop_speed_kmh = np.where(hop_secs > 0, hop_m / hop_secs * 3.6, 0.0) \
            if hop_m.size == hop_secs.size else np.zeros(hop_secs.size)

    if len(coords) > 1 and coords[0] != coords[-1]:
        geom = {"type": "LineString", "coordinates": coords}
    else:
        geom = {"type": "Point", "coordinates": coords[0]}

    # Deliberately lean. Every field here is read by something: animation.html
    # and geologger_upload_agent (geometry, times, segment_times), fill_gaps
    # (agent, start/end_time), analyse_productivity (type, duration_s),
    # check_output (start/end_time), extract_station_visits (total_distance_m).
    #
    # segment_times is what makes the animation vary speed by road type: each
    # segment's share of the journey time is set by its highway's configured
    # speed, so a vehicle already slows on residential streets and speeds up on
    # trunk roads without any of that having to be stated again.
    #
    # Per-segment speeds, lengths and highway types are therefore NOT stored:
    # lengths are recoverable from the geometry, speeds are length over
    # segment_times, and the highway split is in trip_routing_analysis.csv,
    # which is where analysis belongs. Carrying them here roughly doubled the
    # file size for data nothing read. --segment-detail puts them back.
    props = {
        "agent": agent_id,
        "type": OPTS["trip_type"],
        "length_m": round(length_m, 1),
        "total_distance_m": odometer,
        "start_time": trip["start_dt"].isoformat(),
        "end_time": trip["end_dt"].isoformat(),
        "duration_s": duration_s,
        "segment_times": [round(float(t), 2) for t in hop_secs],
        # One int, and it is what tells you whose track this file is without
        # having to open agent_user_map.csv alongside it.
        "user_id": trip["user_id"],
    }
    if OPTS.get("segment_detail"):
        props["segment_speeds_kmh"] = [round(float(v), 2) for v in hop_speed_kmh]
        props["segment_highways"] = list(hop_highway)
        props["segment_lengths_m"] = [round(float(m), 1) for m in hop_m]
        # Multiply any road_speed_km-h entry by this to get the speed actually
        # used on this trip. >1 means the vehicle beat the configured speeds.
        props["speed_scale"] = round(float(speed_scale), 4)
    if geom["type"] == "Point":
        # Nothing to interpolate along, and these would all be empty lists.
        for key in ("segment_times", "segment_speeds_kmh",
                    "segment_highways", "segment_lengths_m", "speed_scale"):
            props.pop(key, None)
    if not routed:
        props["routed"] = False

    metrics = {
        "routed": routed,
        "routed_distance_m": round(length_m, 1),
        "routed_time_s": round(routed_time_s, 1),
        # Straight from the routing: distance over the time the speed table
        # implies, not the observed speed, which the source columns already give.
        "routed_avg_speed_kmh": (round(length_m / routed_time_s * 3.6, 2)
                                 if routed_time_s > 0 else None),
        "highway_breakdown": by_highway,
    }
    return props, geom, length_m, metrics


def process_user(task):
    """Route every trip of one user, write their GeoJSON, and return one
    analysis row per trip."""
    agent_id, user_id, trips = task
    features, odometer, unrouted = [], 0.0, 0
    rows = []

    for trip in trips:
        props, geom, length_m, metrics = build_feature(agent_id, trip, odometer)
        if props.get("routed") is False:
            unrouted += 1
        features.append({"type": "Feature", "geometry": geom, "properties": props})

        # Every source column, unchanged, then what the routing produced. Keeping
        # the source verbatim is what makes the file joinable back to the CSV.
        row = {k: trip[k] for k in OPTS["source_columns"] if k in trip}
        row["agent_id"] = agent_id
        row["routed"] = metrics["routed"]
        row["routed_distance_m"] = metrics["routed_distance_m"]
        row["routed_time_s"] = metrics["routed_time_s"]
        row["routed_avg_speed_kmh"] = metrics["routed_avg_speed_kmh"]
        row["snap_distance_m"] = round(float(trip["snap_m"]), 1)
        # None, not 0, for a vehicle's first trip: there is no previous trip to
        # measure from, which is a different statement from "no waiting".
        gap = trip.get("gap_from_previous_s")
        row["gap_from_previous_s"] = (None if gap is None or pd.isna(gap)
                                      else round(float(gap), 1))
        # One JSON object per row: highway type -> distance, speed, time.
        row["highway_breakdown"] = json.dumps(metrics["highway_breakdown"],
                                              default=to_serializable)
        rows.append(row)

        odometer += length_m

    collection = insert_idle_points({"type": "FeatureCollection", "features": features})
    out_path = os.path.join(OPTS["out_dir"], f"agent_{agent_id:04d}_time.geojson")
    with open(out_path, "w") as f:
        json.dump(collection, f, default=to_serializable)

    return {"agent_id": agent_id, "user_id": user_id, "trips": len(trips),
            "features": len(collection["features"]), "unrouted": unrouted,
            "distance_m": odometer, "file": os.path.basename(out_path),
            "rows": rows}


def to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scenario_argument(p)
    p.add_argument("--source", default=DEFAULT_SOURCE_CSV,
                   help="captured trip CSV (default: from scenario.json's folder_name)")
    p.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR,
                   help="directory for the agent_XXXX_time.geojson files")
    p.add_argument("--trips-csv", default=None,
                   help="per-trip routing analysis CSV "
                        "(default: trip_routing_analysis.csv in --out-dir)")
    p.add_argument("--graphml", default=ROAD_NETWORK_FILE, help="road network")
    p.add_argument("--workers", type=int, default=POOL_WORKERS)
    p.add_argument("--max-snap-m", type=float, default=DEFAULT_MAX_SNAP_M,
                   help="drop trips with an endpoint further than this from the network")
    p.add_argument("--max-speed-kmh", type=float, default=DEFAULT_MAX_SPEED_KMH,
                   help="drop trips whose AVERAGE speed (distance / duration) "
                        f"exceeds this (default {DEFAULT_MAX_SPEED_KMH:g}); such "
                        "a record contradicts itself and cannot be repaired. "
                        "0 disables the check")
    p.add_argument("--users", help="comma-separated user_ids to process")
    p.add_argument("--limit-users", type=int, help="process only the first N user_ids")
    p.add_argument("--trip-type", default="passenger",
                   help="value written to each moving feature's \"type\" property")
    p.add_argument("--segment-detail", action="store_true",
                   help="also store per-segment speeds, lengths and highway "
                        "types in the GeoJSON. Roughly doubles the file size, "
                        "and nothing currently reads them -- the animation gets "
                        "its road-type speed variation from segment_times, and "
                        "the highway split is in trip_routing_analysis.csv")
    p.add_argument("--no-time-scaling", action="store_true",
                   help="keep modelled free-flow segment times instead of "
                        "rescaling them to the observed trip duration")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Scenario folder: {FOLDER_NAME}")
    print(f"Routing mode:    {'TIME' if SPEED_BASED_ROUTING else 'DISTANCE'} "
          f"(weight: {ROUTING_ATTR})")

    df, dropped, SOURCE_CSV_COLUMNS = load_trips(args.source, args.max_speed_kmh)
    # Counted over the whole CSV, before any --users/--limit-users selection.
    load_drops = dict(dropped)

    if args.users:
        wanted = {int(u.strip()) for u in args.users.split(",") if u.strip()}
        df = df[df["user_id"].isin(wanted)]
    if args.limit_users:
        keep = sorted(df["user_id"].unique())[:args.limit_users]
        df = df[df["user_id"].isin(keep)]
    if df.empty:
        raise SystemExit("No trips left to process.")

    print(f"Loading road network from {args.graphml} ...")
    route_graph, wgs_coords = build_route_graph(args.graphml)
    print(f"  {route_graph.number_of_nodes()} nodes, {route_graph.number_of_edges()} edges")

    snap = build_snapper(wgs_coords)
    start_nodes, start_dist = snap(df["start_lon"].values, df["start_lat"].values)
    end_nodes, end_dist = snap(df["end_lon"].values, df["end_lat"].values)
    df = df.assign(start_node=start_nodes, end_node=end_nodes,
                   snap_m=np.maximum(start_dist, end_dist))

    before = len(df)
    df = df[df["snap_m"] <= args.max_snap_m]
    snap_drops = before - len(df)
    if df.empty:
        raise SystemExit("No trips left after snapping to the road network.")

    # The analysis file reproduces the source row verbatim, so the whole frame
    # travels to the workers rather than the handful of columns routing needs.
    source_columns = [c for c in SOURCE_CSV_COLUMNS if c in df.columns]
    cols = list(dict.fromkeys(source_columns + [
        "user_id", "start_dt", "end_dt", "duration_s", "distance_km",
        "start_node", "end_node", "snap_m", "gap_from_previous_s"]))
    tasks = [(agent_id, user_id, group[cols].to_dict("records"))
             for agent_id, (user_id, group) in enumerate(df.groupby("user_id", sort=True))]

    opts = {"out_dir": args.out_dir, "trip_type": args.trip_type,
            "scale_times": not args.no_time_scaling,
            "segment_detail": args.segment_detail,
            "source_columns": source_columns}
    worker_args = (route_graph, wgs_coords, opts)

    print(f"Routing {len(df)} trips for {len(tasks)} users "
          f"into {args.out_dir} ...")
    workers = max(1, min(args.workers, len(tasks)))
    if workers > 1:
        with multiprocessing.Pool(workers, initializer=init_worker,
                                  initargs=worker_args) as pool:
            results = list(tqdm(pool.imap_unordered(process_user, tasks),
                                total=len(tasks), desc="Users"))
    else:
        init_worker(*worker_args)
        results = [process_user(t) for t in tqdm(tasks, desc="Users")]

    results.sort(key=lambda r: r["agent_id"])
    index_path = os.path.join(args.out_dir, "agent_user_map.csv")
    pd.DataFrame([{k: v for k, v in r.items() if k != "rows"}
                  for r in results]).to_csv(index_path, index=False)

    # One row per trip: the source record, what the routing made of it, and how
    # the distance splits across highway types.
    trip_rows = [row for r in results for row in r["rows"]]
    trips_path = args.trips_csv or os.path.join(args.out_dir, "trip_routing_analysis.csv")
    os.makedirs(os.path.dirname(os.path.abspath(trips_path)), exist_ok=True)
    pd.DataFrame(trip_rows).to_csv(trips_path, index=False)

    total_unrouted = sum(r["unrouted"] for r in results)
    print(f"\nWrote {len(results)} files to {args.out_dir}")
    print(f"Index: {index_path}")
    print(f"Per-trip analysis: {trips_path}  ({len(trip_rows)} rows)")
    print(f"Trips routed: {len(df)}  "
          f"({total_unrouted} fell back to a straight line, no path on the network)")
    print(f"Total routed distance: {sum(r['distance_m'] for r in results) / 1000:,.0f} km")
    for reason, count in load_drops.items():
        if count:
            print(f"Dropped {count} CSV rows on load: {reason}")
    if snap_drops:
        print(f"Dropped {snap_drops} selected trips: an endpoint was over "
              f"{args.max_snap_m:.0f} m from the network")


if __name__ == "__main__":
    main()
