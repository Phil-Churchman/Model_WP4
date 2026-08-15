"""
Turn captured GPS trip records into per-vehicle GeoJSON tracks.

Reads <folder_name>/captured_locations/source_trip_data.csv -- one row per
observed trip, with a start/end timestamp and a start/end lat-lon -- snaps each
endpoint to the road network and routes between them with the same weights the
simulation uses, then writes one file per user_id in the format the simulation
produces (<folder_name>/output/output_trips_time_queued/agent_XXXX_time.geojson).

The folder name comes from scenario.json, as it does for the simulation. Unlike
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
from scenario_config import load_scenario, add_scenario_argument

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
ROAD_SPEEDS_MS = {k: v / 3.6 for k, v in scenario_cfg["road_speed_km-h"].items()}
SPEED_BASED_ROUTING = scenario_cfg.get("speed_based_routing", False)
# Same reasoning as the simulation: the raw OSM graph carries no "travel_time",
# so both weights are materialised explicitly rather than left to default to 1.
ROUTING_ATTR = "travel_time_s" if SPEED_BASED_ROUTING else "length_m"
MIN_WEIGHT = 1e-6

DEFAULT_SOURCE_CSV = os.path.join(SCENARIO.captured_dir, "source_trip_data.csv")
ROAD_NETWORK_FILE = os.path.join(SCENARIO.input_dir, "roads.graphml")
DEFAULT_OUTPUT_DIR = SCENARIO.trips_time_dir

# An endpoint further than this from any node is off the extracted network, so
# the trip is dropped rather than dragged to a road it never used.
DEFAULT_MAX_SNAP_M = 2000.0

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
        speed = (max(ROAD_SPEEDS_MS.get(h, 0.1) for h in hw) if isinstance(hw, list)
                 else ROAD_SPEEDS_MS.get(hw, 0.1))
        # Weights must stay strictly positive: a road type configured at 0 km/h
        # would otherwise divide by zero.
        speed = max(float(speed), 0.1)
        length = max(float(data.get("length", 1.0)), MIN_WEIGHT)
        travel_time = max(length / speed, MIN_WEIGHT)
        prev = edge_lookup.get((u, v))
        if prev is None or travel_time < prev[1]:
            edge_lookup[(u, v)] = (length, travel_time)

    route_graph = nx.DiGraph()
    route_graph.add_nodes_from(G.nodes())
    route_graph.add_edges_from(
        (u, v, {"length_m": length, "travel_time_s": travel_time})
        for (u, v), (length, travel_time) in edge_lookup.items()
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

def load_trips(csv_path):
    """
    Read the captured trips and drop the rows that cannot be routed.

    Returns (DataFrame, dict of drop counts by reason).
    """
    df = pd.read_csv(csv_path)
    dropped = {}
    before = len(df)

    df["start_dt"] = pd.to_datetime(df["start_date"].astype(str) + " "
                                    + df["start_time"].astype(str), errors="coerce")
    df["end_dt"] = pd.to_datetime(df["end_date"].astype(str) + " "
                                  + df["end_time"].astype(str), errors="coerce")
    df = df.dropna(subset=["start_dt", "end_dt"])
    dropped["unparseable timestamp"] = before - len(df)

    df = df.sort_values(["user_id", "start_dt"]).reset_index(drop=True)

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

    return df.reset_index(drop=True), dropped


# ============================================================
# ROUTING & FEATURE BUILDING
# ============================================================

def init_worker(route_graph, wgs_coords, opts):
    global ROUTE_GRAPH, WGS_COORDS, OPTS
    ROUTE_GRAPH, WGS_COORDS, OPTS = route_graph, wgs_coords, opts


def route(u, v):
    """
    Node path u -> v with its per-hop metres and seconds.

    Returns (path, length_m, hop_secs, routed). `routed` is False when the two
    endpoints are in different components, in which case the caller gets a
    straight-line stand-in rather than losing the trip.
    """
    if u == v:
        return [u], 0.0, np.empty(0), True
    try:
        _, path = nx.bidirectional_dijkstra(ROUTE_GRAPH, u, v, weight=ROUTING_ATTR)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [u, v], 0.0, np.empty(0), False

    hop_m = np.empty(len(path) - 1)
    hop_s = np.empty(len(path) - 1)
    for i in range(len(path) - 1):
        data = ROUTE_GRAPH[path[i]][path[i + 1]]
        hop_m[i] = data["length_m"]
        hop_s[i] = data["travel_time_s"]
    return path, float(hop_m.sum()), hop_s, True


def build_feature(agent_id, trip, odometer):
    """One CSV row -> one GeoJSON feature, in the simulation's output format."""
    path, length_m, hop_secs, routed = route(trip["start_node"], trip["end_node"])
    coords = [WGS_COORDS[n] for n in path]
    duration_s = float(trip["duration_s"])

    if OPTS["scale_times"] and hop_secs.size:
        # The recorded duration is ground truth; the modelled free-flow times only
        # say how that duration was spread along the route.
        total = float(hop_secs.sum())
        hop_secs = (hop_secs * (duration_s / total) if total > 0
                    else np.full(hop_secs.size, duration_s / hop_secs.size))
    elif not routed:
        hop_secs = np.array([duration_s])

    if len(coords) > 1 and coords[0] != coords[-1]:
        geom = {"type": "LineString", "coordinates": coords}
    else:
        geom = {"type": "Point", "coordinates": coords[0]}

    props = {
        "agent": agent_id,
        "type": OPTS["trip_type"],
        "length_m": length_m,
        "total_distance_m": odometer,
        "start_time": trip["start_dt"].isoformat(),
        "end_time": trip["end_dt"].isoformat(),
        "duration_s": duration_s,
        "segment_times": [round(float(t), 2) for t in hop_secs],
        # Provenance, so a track can be traced back to the rows it came from.
        "user_id": trip["user_id"],
        "observed_distance_m": (None if pd.isna(trip["distance_km"])
                                else float(trip["distance_km"]) * 1000.0),
        "snap_distance_m": round(float(trip["snap_m"]), 1),
    }
    if geom["type"] == "Point":
        # Nothing to interpolate along, and the field would be an empty list.
        props.pop("segment_times")
    if not routed:
        props["routed"] = False

    return props, geom, length_m


def process_user(task):
    """Route every trip of one user and write that user's GeoJSON file."""
    agent_id, user_id, trips = task
    features, odometer, unrouted = [], 0.0, 0

    for trip in trips:
        props, geom, length_m = build_feature(agent_id, trip, odometer)
        if props.get("routed") is False:
            unrouted += 1
        features.append({"type": "Feature", "geometry": geom, "properties": props})
        odometer += length_m

    collection = insert_idle_points({"type": "FeatureCollection", "features": features})
    out_path = os.path.join(OPTS["out_dir"], f"agent_{agent_id:04d}_time.geojson")
    with open(out_path, "w") as f:
        json.dump(collection, f, default=to_serializable)

    return {"agent_id": agent_id, "user_id": user_id, "trips": len(trips),
            "features": len(collection["features"]), "unrouted": unrouted,
            "distance_m": odometer, "file": os.path.basename(out_path)}


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
    p.add_argument("--graphml", default=ROAD_NETWORK_FILE, help="road network")
    p.add_argument("--workers", type=int, default=POOL_WORKERS)
    p.add_argument("--max-snap-m", type=float, default=DEFAULT_MAX_SNAP_M,
                   help="drop trips with an endpoint further than this from the network")
    p.add_argument("--users", help="comma-separated user_ids to process")
    p.add_argument("--limit-users", type=int, help="process only the first N user_ids")
    p.add_argument("--trip-type", default="passenger",
                   help="value written to each moving feature's \"type\" property")
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

    df, dropped = load_trips(args.source)
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

    cols = ["user_id", "start_dt", "end_dt", "duration_s", "distance_km",
            "start_node", "end_node", "snap_m"]
    tasks = [(agent_id, user_id, group[cols].to_dict("records"))
             for agent_id, (user_id, group) in enumerate(df.groupby("user_id", sort=True))]

    opts = {"out_dir": args.out_dir, "trip_type": args.trip_type,
            "scale_times": not args.no_time_scaling}
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
    pd.DataFrame(results).to_csv(index_path, index=False)

    total_unrouted = sum(r["unrouted"] for r in results)
    print(f"\nWrote {len(results)} files to {args.out_dir}")
    print(f"Index: {index_path}")
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
