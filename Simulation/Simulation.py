import os
import sys
import json
import math
import argparse
import random
import stat
import time
import multiprocessing
import heapq
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import namedtuple
from datetime import datetime, timedelta  # NB: datetime.time would shadow `time`
import osmnx as ox
import networkx as nx
import pandas as pd
from tqdm import tqdm
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra
from shapely.geometry import Point
from pyproj import Transformer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import load_scenario, add_scenario_argument

from trip_demand_generator import generate_trips
from fill_gaps import insert_idle_points

# ============================================================
# CONFIGURATION & GLOBALS
# ============================================================
# Parsed at module level, not in main(), because the constants below are read
# inside calculate_next_activity -- which runs in the pool workers, and those
# re-import this module under spawn. export=True publishes the resolved path so
# those re-imports land on the same scenario rather than the default.
_parser = argparse.ArgumentParser(description="Run the taxi fleet simulation")
add_scenario_argument(_parser)
_args, _ = _parser.parse_known_args()
SCENARIO = load_scenario(_args.scenario, export=True).require_folder()
scenario_cfg = SCENARIO.cfg

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEMAND_MODEL = scenario_cfg.get("demand_model", False)
FOLDER_NAME = SCENARIO.folder_name
ROAD_SPEEDS = scenario_cfg["road_speed_km-h"]
ROAD_SPEEDS_MS = {k: v / 3.6 for k, v in ROAD_SPEEDS.items()}
NUM_AGENTS_CFG = scenario_cfg["agents"]
INPUT_DIR = SCENARIO.input_dir
ROAD_NETWORK_FILE = os.path.join(INPUT_DIR, "roads.graphml")
SWAP_STATIONS_FILE = os.path.join(INPUT_DIR, "swap_stations.geojson")
TAXI_RANKS_FILE = os.path.join(INPUT_DIR, "taxi_ranks.geojson")

OUTPUT_DIR = SCENARIO.output_dir
OUTPUT_PER_AGENT_DIR = os.path.join(OUTPUT_DIR, "output_trips")
OUTPUT_PER_AGENT_TIME_DIR = os.path.join(OUTPUT_DIR, "output_trips_time_queued")
HISTOGRAM_PLOT = os.path.join(OUTPUT_DIR, "station_queues_analysis.png")
SWAP_EXCEL_OUTPUT = os.path.join(OUTPUT_DIR, "swap_station_timesteps.xlsx")

# --- SIMULATION PARAMETERS ---
SIMULATION_INTERVAL_SEC = scenario_cfg.get("simulation_step_sec", 60) 
SWAP_WAIT_SEC = scenario_cfg.get("swap_wait_sec", 600)
d_start, d_end = scenario_cfg["start_time"], scenario_cfg["end_time"]
DAY_START = datetime(d_start[0], d_start[1], d_start[2], d_start[3], d_start[4], d_start[5])
DAY_END = datetime(d_end[0], d_end[1], d_end[2], d_end[3], d_end[4], d_end[5])
MAX_TOTAL_DISTANCE_M = scenario_cfg.get("max_total_distance_m", 80000)
BUFFER_DISTANCE = scenario_cfg.get("buffer_distance", 12000)
PASSENGER_MAX_DIST = scenario_cfg.get("passenger_max_dist", 6000)
DEVIATION_FACTOR = scenario_cfg.get("deviation_factor", 1)
PROBABILITY_OF_HAILING_TAXI = scenario_cfg.get("probability_hail", 0.75)

# --- WORKER POOL ---
# Every worker unpickles its own copy of the routing state at pool construction
# (~530 MB each), whether or not it is ever used, and each pool.map round trip
# pickles the agents both ways. Measured at 400 agents x 3h on Accra:
#   12 workers (the cpu_count default)  110.0 s   7101 MB
#    4 workers                           64.1 s   3151 MB
#   pool never used (serial routing)     93.5 s   7097 MB
# Past a handful of workers the IPC costs more than the routing it parallelises.
POOL_WORKERS = max(1, min(int(scenario_cfg.get("worker_processes", 4)),
                          multiprocessing.cpu_count()))
# The route cache is per worker, so its size is multiplied by POOL_WORKERS.
# Reuse is very short-range (a route is looked up once to pick a target and
# again to build the feature), so a small cache captures nearly all the hits.
ROUTE_CACHE_SIZE = int(scenario_cfg.get("route_cache_size", 5000))

# --- ROUTING LOGIC ---
SPEED_BASED_ROUTING = scenario_cfg.get("speed_based_routing", False)
# Attribute name on ROUTE_GRAPH. The raw OSM graph carries no "travel_time",
# so routing against it silently fell back to networkx's default weight of 1
# (i.e. hop count). Both weights are now materialised explicitly.
ROUTING_ATTR = "travel_time_s" if SPEED_BASED_ROUTING else "length_m"
MIN_WEIGHT = 1e-6  # scipy.csgraph reads a stored zero as "no edge"

# Workers Globals
ROUTE_GRAPH = None      # metric nx.DiGraph, one weight per (u, v)
NODES_GLOBAL = None
NODE_TREE = None        # KDTree over all nodes, for passenger sampling only
SWAP_NODES_GLOBAL = None
TAXI_NODES_GLOBAL = None
UTM_COORD_LOOKUP = None # --- MODIFIED ---
WGS_COORD_LOOKUP = None # --- ADDED ---
EDGE_LOOKUP = None
FIELD = None            # TargetField over the fixed destinations
SWAP_ROWS = None        # FIELD rows belonging to swap stations
TAXI_ROWS = None        # FIELD rows belonging to taxi ranks

unallocated_demand = []
allocated_demand = []
met_demand = []


# ============================================================
# CLASSES
# ============================================================

class SwapStation:
    def __init__(self, station_id, num_servers=2):
        self.station_id = station_id
        self.num_servers = num_servers
        self.servers_free_at = [0] * num_servers
        heapq.heapify(self.servers_free_at)
        self.active_taxis = [] # Stores (finish_sec, arrival_sec, agent_id, service_start_sec)
        self.queue_history = []
        self.timestep_records = [] # <-- ADD THIS

    def process_arrival(self, arrival_sec, duration_sec, agent_id): # <-- ADD agent_id parameter
        earliest_free = self.servers_free_at[0]
        start_sec = max(arrival_sec, earliest_free)
        finish_sec = start_sec + duration_sec
        heapq.heapreplace(self.servers_free_at, finish_sec)
        heapq.heappush(self.active_taxis, (finish_sec, arrival_sec, agent_id, start_sec)) # <-- UPDATE TUPLE
        return finish_sec

    def record_queue(self, current_sec, timestamp_str=""): # <-- ADD timestamp_str parameter
        while self.active_taxis and self.active_taxis[0][0] <= current_sec:
            heapq.heappop(self.active_taxis)
        
        self.queue_history.append(len(self.active_taxis))


        swapping_agent_ids = []
        queueing_agent_ids = []

        for finish_sec, arrival_sec, agent_id, start_sec in self.active_taxis:
            if current_sec >= start_sec:
                swapping_agent_ids.append(agent_id)
            else:
                queueing_agent_ids.append(agent_id)

        self.timestep_records.append((
            self.station_id,
            timestamp_str,
            current_sec,
            len(swapping_agent_ids),
            swapping_agent_ids,
            len(queueing_agent_ids),
            queueing_agent_ids
        ))

# What crosses the pool boundary. Sending the TaxiAgent itself would pickle its
# whole time_features list -- which grows all run -- in both directions on every
# timestep, so IPC cost scaled with simulation length. These carry only what the
# decision needs, and only what it produced.
AgentRequest = namedtuple("AgentRequest",
                          "id pos running_total trip_count assigned_trip "
                          "allocated_trip current_sec")
AgentUpdate = namedtuple("AgentUpdate",
                         "trip_type feature pos running_total trip_count "
                         "busy_until pending_wait_sec state arrival_distance "
                         "assigned_trip met_record")


class TaxiAgent:
    def __init__(self, agent_id, start_node, spawn_time):
        self.id = agent_id
        self.pos = start_node
        self.spawn_time = spawn_time
        self.running_total = random.uniform(5000, 60000)
        self.arrival_distance = self.running_total
        self.trip_count = 0
        self.busy_until = 0
        self.state = "IDLE" 
        self.current_trip_type = None # Track the type of the active trip
        self.target_node = None
        self.pending_wait_sec = 0
        self.agent_features = []
        self.time_features = []
        self.marked_for_removal = False
        self.allocated_trip = None
        self.assigned_trip = False

# ============================================================
# WORKER INITIALIZATION & CACHING
# ============================================================

class TargetField:
    """
    Cost-to-target, and path reconstruction, for a FIXED set of destination
    nodes -- precomputed once with one reverse Dijkstra per target.

    Memory is bounded at len(targets) x n_nodes regardless of how many agents
    or timesteps the simulation runs, which is what makes this safe: a
    many-sources csgraph call allocates n_agents x n_nodes on EVERY call
    (~850 MB per timestep per process at 1000 agents on the Accra graph).
    """
    __slots__ = ("targets", "row_of", "dist", "pred", "index_of", "node_at")

    def __init__(self, targets, dist, pred, index_of, node_at):
        self.targets = targets
        self.row_of = {t: i for i, t in enumerate(targets)}
        self.dist = dist    # float32 [n_targets x n_nodes], cost node -> target
        self.pred = pred    # int32,   next node on the path node -> target
        self.index_of = index_of
        self.node_at = node_at

    def best(self, from_node, rows):
        """Cheapest reachable target among `rows`, or None if all unreachable."""
        if rows is None or len(rows) == 0:
            return None
        col = self.dist[rows, self.index_of[from_node]]
        k = int(np.argmin(col))
        return self.targets[rows[k]] if np.isfinite(col[k]) else None

    def cost(self, from_node, target):
        """Routing cost from `from_node` to `target`, or None if unreachable."""
        c = float(self.dist[self.row_of[target], self.index_of[from_node]])
        return c if math.isfinite(c) else None

    def path(self, from_node, target):
        """Node list from `from_node` to `target`, or None if unreachable."""
        i, stop = self.index_of[from_node], self.index_of[target]
        pred_row = self.pred[self.row_of[target]]
        out = [from_node]
        while i != stop:
            i = int(pred_row[i])
            if i < 0:
                return None
            out.append(self.node_at[i])
        return out


def build_edge_lookup(G):
    """One record per (u, v): the fastest of any parallel edges."""
    edge_lookup = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        hw = data.get("highway")
        speed = max([ROAD_SPEEDS_MS.get(h, 0.1) for h in hw]) if isinstance(hw, list) else ROAD_SPEEDS_MS.get(hw, 0.1)
        # Weights must stay strictly positive: a road type configured at 0 km/h
        # would divide by zero, and a zero stored in the CSR reads as "no edge".
        speed = max(float(speed), 0.1)
        length = max(float(data.get("length", 1.0)), MIN_WEIGHT)
        travel_time = max(length / speed, MIN_WEIGHT)
        if (u, v) not in edge_lookup or travel_time < edge_lookup[(u, v)]["travel_time_s"]:
            edge_lookup[(u, v)] = {"u": int(u), "v": int(v), "highway": hw, "speed_m_s": float(speed), "length_m": float(length), "travel_time_s": float(travel_time)}
    return edge_lookup


def build_routing_structures(edge_lookup, nodes):
    """
    Collapse the OSM MultiDiGraph into one metric DiGraph (a single float
    weight per (u, v) instead of a min() over parallel edge dicts on every
    relaxation) plus the reverse CSR that the target fields are built from.
    """
    route_graph = nx.DiGraph()
    route_graph.add_nodes_from(nodes)
    route_graph.add_edges_from(
        (u, v, {"length_m": e["length_m"], "travel_time_s": e["travel_time_s"]})
        for (u, v), e in edge_lookup.items()
    )

    index_of = {n: i for i, n in enumerate(nodes)}
    m = len(edge_lookup)
    rows, cols = np.empty(m, dtype=np.int32), np.empty(m, dtype=np.int32)
    vals = np.empty(m, dtype=np.float64)
    for i, ((u, v), e) in enumerate(edge_lookup.items()):
        rows[i], cols[i], vals[i] = index_of[u], index_of[v], e[ROUTING_ATTR]

    forward = csr_matrix((vals, (rows, cols)), shape=(len(nodes), len(nodes)))
    return route_graph, index_of, forward.T.tocsr()


def build_target_field(target_nodes, reverse_csr, index_of, nodes):
    """
    One reverse Dijkstra per distinct destination. Searching the transposed
    graph from a target yields the cost to that target from every node, and
    its predecessors are the forward successors on the way there.
    """
    targets = sorted(set(target_nodes), key=lambda t: index_of[t])
    if not targets:
        return None
    dist, pred = csgraph_dijkstra(
        reverse_csr,
        indices=[index_of[t] for t in targets],
        return_predecessors=True,
    )
    return TargetField(targets, dist.astype(np.float32), pred.astype(np.int32),
                       index_of, nodes)


def init_worker(route_graph, nodes, node_tree, swap_nodes, taxi_nodes, utm_coords,
                wgs_coords, edge_lookup, field, swap_rows, taxi_rows):
    global ROUTE_GRAPH, NODES_GLOBAL, NODE_TREE, SWAP_NODES_GLOBAL, TAXI_NODES_GLOBAL
    global UTM_COORD_LOOKUP, WGS_COORD_LOOKUP, EDGE_LOOKUP, FIELD, SWAP_ROWS, TAXI_ROWS
    ROUTE_GRAPH, NODES_GLOBAL, NODE_TREE = route_graph, nodes, node_tree
    SWAP_NODES_GLOBAL, TAXI_NODES_GLOBAL = swap_nodes, taxi_nodes
    UTM_COORD_LOOKUP, WGS_COORD_LOOKUP, EDGE_LOOKUP = utm_coords, wgs_coords, edge_lookup
    FIELD, SWAP_ROWS, TAXI_ROWS = field, swap_rows, taxi_rows


def measure_path(path):
    """
    Cumulative metres and seconds at each node of `path`. Cumulative (rather
    than total-only) so a truncated route can be measured by slicing instead of
    by routing again, and so per-hop times come back as a diff -- which is why
    the per-hop edge records need not be cached at all.

    float32 halves the cached footprint; the sums are accumulated in float64
    first, so the only loss is in the stored result (~0.01 m at 100 km).
    """
    n = len(path)
    d_m, d_s = np.empty(max(n - 1, 0)), np.empty(max(n - 1, 0))
    for i in range(n - 1):
        edge = EDGE_LOOKUP.get((path[i], path[i + 1]))
        if edge is not None:
            d_m[i], d_s[i] = edge["length_m"], edge["travel_time_s"]
        else:
            data = ROUTE_GRAPH.get_edge_data(path[i], path[i + 1]) or {}
            d_m[i] = data.get("length_m", 0.0)
            d_s[i] = data.get("travel_time_s", d_m[i] / 10.0)  # Fallback speed
    cum_m = np.concatenate(([0.0], np.cumsum(d_m))).astype(np.float32)
    cum_s = np.concatenate(([0.0], np.cumsum(d_s))).astype(np.float32)
    return cum_m, cum_s


@functools.lru_cache(maxsize=ROUTE_CACHE_SIZE)
def get_cached_route_info(u, v):
    """Path plus its cumulative metres/seconds at each node."""
    path = FIELD.path(u, v) if (FIELD is not None and v in FIELD.row_of) else None
    if path is None:
        _, path = nx.bidirectional_dijkstra(ROUTE_GRAPH, u, v, weight=ROUTING_ATTR)
    cum_m, cum_s = measure_path(path)
    return path, cum_m, cum_s


def route(u, v, upto=None):
    """
    Route u -> v as (path, metres, seconds, per-hop seconds), optionally
    truncated to the first `upto` nodes. A prefix of a shortest path is itself a
    shortest path, so truncation is a slice of the cached result -- never a new
    search.
    """
    path, cum_m, cum_s = get_cached_route_info(u, v)
    end = upto - 1 if (upto is not None and upto < len(path)) else len(path) - 1
    if end != len(path) - 1:
        path = path[:end + 1]
    return path, float(cum_m[end]), float(cum_s[end]), np.diff(cum_s[:end + 1])


def get_cached_cost(u, v):
    """Routing cost only -- an array lookup when `v` is a fixed destination."""
    if FIELD is not None and v in FIELD.row_of:
        c = FIELD.cost(u, v)
        if c is None:
            raise nx.NetworkXNoPath(f"no path from {u} to {v}")
        return c
    _, cum_m, cum_s = get_cached_route_info(u, v)
    return float(cum_s[-1] if SPEED_BASED_ROUTING else cum_m[-1])

# ============================================================
# HELPERS
# ============================================================

def to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    return obj

def get_target_node(agent_pos, trip_type):
    # Swap stations and taxi ranks are fixed, so their cost from here is a
    # precomputed array lookup -- no search, and over ALL of them rather than
    # the handful the KD-Tree used to shortlist.
    if trip_type == "to_swap":
        best = FIELD.best(agent_pos, SWAP_ROWS)
        return best if best is not None else random.choice(SWAP_NODES_GLOBAL)

    elif trip_type in ["taxi", "hail"]:
        best = FIELD.best(agent_pos, TAXI_ROWS)
        return best if best is not None else random.choice(TAXI_NODES_GLOBAL)

    elif trip_type == "passenger":
        # Destinations vary, so this still samples the KD-Tree ball that
        # DEVIATION_FACTOR sizes, and accepts the first candidate inside the
        # distance budget. The route is cached, so the caller reuses it rather
        # than routing to the accepted target a second time.
        candidates = NODE_TREE.query_ball_point(UTM_COORD_LOOKUP[agent_pos],
                                                PASSENGER_MAX_DIST / DEVIATION_FACTOR)
        # sample() draws the 5 we actually look at; shuffle() used to permute
        # the whole ball (~6800 nodes on Accra) just to discard all but 5.
        nearest = None
        for idx in random.sample(candidates, min(5, len(candidates))):
            target = NODES_GLOBAL[idx]
            if target == agent_pos:
                continue
            try:
                _, metres, _, _ = route(agent_pos, target)
            except Exception:
                continue
            if metres <= PASSENGER_MAX_DIST:
                return target
            if nearest is None or metres < nearest[1]:
                nearest = (target, metres)
        return nearest[0] if nearest else random.choice(NODES_GLOBAL)

def calculate_next_activity(req):
    """Decide and route one agent's next trip. Returns None if it stays idle."""
    current_sec = req.current_sec
    assigned_trip, allocated_trip = req.assigned_trip, req.allocated_trip
    is_swap = (req.running_total + BUFFER_DISTANCE > MAX_TOTAL_DISTANCE_M and req.trip_count % 2 == 0)

    if not DEMAND_MODEL:
        if is_swap:
            trip_type = "to_swap"
        elif req.trip_count % 2 == 0:
            trip_type = "hail" if random.random() < PROBABILITY_OF_HAILING_TAXI else "taxi"
        else:
            trip_type = "passenger"
    else:
        if is_swap:
            trip_type = "to_swap"
            assigned_trip = False
        elif assigned_trip and allocated_trip:
            if req.trip_count % 2 == 0:
                trip_type = "pickup"
            else:
                trip_type = "passenger"
                assigned_trip = False
        else:
            return None

    wait_sec_raw = SWAP_WAIT_SEC if trip_type == "to_swap" else (60 if trip_type == "hail" else (random.randint(1, 20) * 60 if trip_type == "taxi" else 60))
    wait_sec = math.ceil(wait_sec_raw / SIMULATION_INTERVAL_SEC) * SIMULATION_INTERVAL_SEC

    # Target & Route Determination
    if trip_type == "passenger" and DEMAND_MODEL and allocated_trip:
        target_node = allocated_trip["dest_node"]
    elif trip_type == "pickup":
        target_node = allocated_trip["source_node"]
    else:
        target_node = get_target_node(req.pos, trip_type)

    # Retrieve all pre-computed route details in one cached call
    try:
        path, total_len, total_time, hop_secs = route(req.pos, target_node)
        if trip_type == "hail" and len(path) > 2:
            # Truncation re-slices the cached route rather than re-routing.
            path, total_len, total_time, hop_secs = route(
                req.pos, target_node, upto=random.randint(2, len(path))
            )
            target_node = path[-1]
    except Exception:
        path = [req.pos, target_node]
        total_len, total_time, hop_secs = 0.0, 0.0, np.empty(0)

    coords = [WGS_COORD_LOOKUP[n] for n in path]
    travel_sec = max(SIMULATION_INTERVAL_SEC, math.ceil(total_time / SIMULATION_INTERVAL_SEC) * SIMULATION_INTERVAL_SEC)
    geom = {"type": "LineString", "coordinates": coords} if (len(coords) > 1 and coords[0] != coords[-1]) else {"type": "Point", "coordinates": coords[0]}

    start_dt = DAY_START + timedelta(seconds=current_sec)
    end_dt = start_dt + timedelta(seconds=travel_sec)

    feat = {
        "type": "Feature", "geometry": geom,
        "properties": {
            "agent": req.id, "type": trip_type, "length_m": total_len,
            "total_distance_m": req.running_total, "start_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat(), "duration_s": travel_sec,
            # One float per hop, aligned with the coordinate pairs, replacing a
            # six-field edge record per hop -- the animation only ever read the
            # travel time out of those.
            "segment_times": [round(float(t), 2) for t in hop_secs]
        }
    }

    if trip_type == "to_swap":
        running_total, trip_count = 0, 0
    else:
        running_total, trip_count = req.running_total + total_len, req.trip_count + 1

    met_record = None
    if DEMAND_MODEL and trip_type == "passenger":
        met_record = {
            "idx": allocated_trip["id"],
            "request_time": allocated_trip["departure_time"],
            "pickup_time": (DAY_START + timedelta(seconds=current_sec)).isoformat(),
            "source_facility": allocated_trip["source_facility"],
            "dest_facility": allocated_trip["dest_facility"],
            "source_category": allocated_trip["source_category"],
            "dest_category": allocated_trip["dest_category"]
        }

    return AgentUpdate(
        trip_type=trip_type, feature=feat, pos=target_node,
        running_total=running_total, trip_count=trip_count,
        busy_until=current_sec + travel_sec, pending_wait_sec=wait_sec,
        state="DRIVING_TO_SWAP" if trip_type == "to_swap" else "DRIVING",
        arrival_distance=req.running_total + total_len,
        assigned_trip=assigned_trip, met_record=met_record)


def request_for(agent, current_sec):
    return AgentRequest(agent.id, agent.pos, agent.running_total, agent.trip_count,
                        agent.assigned_trip, agent.allocated_trip, current_sec)


def apply_update(agent, up):
    """Fold a worker's decision back into the agent the parent owns."""
    if up is None:
        return
    agent.current_trip_type = up.trip_type
    agent.time_features.append(up.feature)
    agent.state = up.state
    agent.busy_until = up.busy_until
    agent.pending_wait_sec = up.pending_wait_sec
    agent.pos = agent.target_node = up.pos
    agent.arrival_distance = up.arrival_distance
    agent.running_total = up.running_total
    agent.trip_count = up.trip_count
    agent.assigned_trip = up.assigned_trip
    # Workers cannot append to the parent's met_demand, so it travels back here.
    if up.met_record is not None:
        met_demand.append(up.met_record)

# ============================================================
# MAIN
# ============================================================

def clear_output_dir():
    """
    Delete the files under OUTPUT_DIR so a smaller run cannot leave stale
    results behind from a larger one.

    Files only -- never the directories themselves. shutil.rmtree also has to
    rmdir, and on Windows a directory handle held by OneDrive, Explorer or an
    indexer makes that fail with PermissionError even once the directory is
    empty. That aborts the run *after* the files are already gone, which is the
    worst of both outcomes. Leaving the empty directories in place avoids the
    failure mode entirely; they get reused as they are.
    """
    if os.path.basename(OUTPUT_DIR) != "output" or not str(FOLDER_NAME).strip():
        raise ValueError(f"Refusing to clear unexpected output path: {OUTPUT_DIR}")
    if not os.path.isdir(OUTPUT_DIR):
        return

    stubborn = []
    for root, _dirs, files in os.walk(OUTPUT_DIR):
        for name in files:
            path = os.path.join(root, name)
            for attempt in range(4):
                try:
                    os.remove(path)
                    break
                except FileNotFoundError:
                    break
                except OSError:
                    if attempt == 3:
                        stubborn.append(path)
                        break
                    # Clear a read-only bit, then give a transient lock (sync
                    # client, indexer, antivirus) a moment to let go.
                    try:
                        os.chmod(path, stat.S_IWRITE)
                    except OSError:
                        pass
                    time.sleep(0.25)

    if stubborn:
        shown = "\n  ".join(stubborn[:10])
        more = f"\n  ... and {len(stubborn) - 10} more" if len(stubborn) > 10 else ""
        raise RuntimeError(
            "Could not clear the previous run from the output folder. Close "
            "whatever is holding these open (Excel and OneDrive are the usual "
            f"culprits) and re-run:\n  {shown}{more}")


def main():
    # Wipe previous results before anything writes, so analysis can never mix
    # this run's output with an earlier one's.
    clear_output_dir()

    # Created up front: plot_station_queues, the Excel export and the demand
    # files all write here long before the per-agent save loop, so creating
    # these late meant a fresh scenario folder crashed after the whole
    # simulation had already run.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # os.makedirs(OUTPUT_PER_AGENT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PER_AGENT_TIME_DIR, exist_ok=True)

    G_wgs84 = ox.load_graphml(ROAD_NETWORK_FILE)
    
    # Save original lat/long for GeoJSON mapping outputs
    wgs_coords_dict = {n: (float(G_wgs84.nodes[n]['x']), float(G_wgs84.nodes[n]['y'])) for n in G_wgs84.nodes()}
    
    # Project the graph to UTM (Meters)
    G = ox.project_graph(G_wgs84)
    print(f"Routing Mode: {'TIME' if SPEED_BASED_ROUTING else 'DISTANCE'} (Weight: {ROUTING_ATTR})")
    
    # Create transformer for GeoJSON inputs (WGS84 -> UTM)
    target_crs = G.graph['crs']
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    
    edge_lookup = build_edge_lookup(G)
    nodes = list(G.nodes())
    route_graph, index_of, reverse_csr = build_routing_structures(edge_lookup, nodes)

    # Projected metric coordinates, for passenger-destination sampling
    utm_coords_dict = {n: (float(G.nodes[n]['x']), float(G.nodes[n]['y'])) for n in nodes}
    node_tree = KDTree([utm_coords_dict[n] for n in nodes])

    def load_pts(path):
        with open(path) as f: data = json.load(f)
        xs, ys = [], []
        for feat in data["features"]:
            lon, lat = feat["geometry"]["coordinates"]
            # Transform WGS84 GeoJSON coordinate to UTM Graph Coordinate
            x, y = transformer.transform(lon, lat)
            xs.append(x)
            ys.append(y)
        return list(ox.nearest_nodes(G, xs, ys))

    if DEMAND_MODEL:
        print("Generating trip demand using the demand model...")
        trips=[]
        features = generate_trips(SCENARIO)["features"]
        idx = 0
        for feat in features:
            src_lon, src_lat = feat["geometry"]["coordinates"][0]
            dst_lon, dst_lat = feat["geometry"]["coordinates"][-1]
            
            # Transform demand geometries to UTM
            src_x, src_y = transformer.transform(src_lon, src_lat)
            dst_x, dst_y = transformer.transform(dst_lon, dst_lat)

            trips.append({
                "id": idx,
                "departure_time": feat["properties"]["departure_time"],
                "source_facility": feat["properties"]["source_facility"],
                "dest_facility": feat["properties"]["dest_facility"],
                "source_category": feat["properties"]["source_category"],
                "dest_category": feat["properties"]["dest_category"],
                "source_node": ox.nearest_nodes(G, src_x, src_y),
                "dest_node": ox.nearest_nodes(G, dst_x, dst_y)
          })
            idx += 1
        trips.sort(key=lambda x: x["departure_time"])

    swap_nodes = load_pts(SWAP_STATIONS_FILE)
    taxi_nodes = [] if DEMAND_MODEL else load_pts(TAXI_RANKS_FILE)

    # Swap stations and taxi ranks never move, so the cost to reach each one
    # from anywhere is precomputed once here instead of being re-searched per
    # agent per trip. Bounded at n_targets x n_nodes (~68 MB for Accra).
    fixed_targets = list(swap_nodes) + list(taxi_nodes)
    print(f"Precomputing cost-to-target fields for {len(set(fixed_targets))} fixed destinations...")
    field = build_target_field(fixed_targets, reverse_csr, index_of, nodes)
    swap_rows = np.array(sorted({field.row_of[n] for n in swap_nodes}), dtype=np.intp)
    taxi_rows = np.array(sorted({field.row_of[n] for n in taxi_nodes}), dtype=np.intp)

    with open(SWAP_STATIONS_FILE) as f: station_data = json.load(f)
    stations = {swap_nodes[i]: SwapStation(feat["properties"].get("facility_id", i), int(feat["properties"].get("posts", 2))) for i, feat in enumerate(station_data["features"])}

    total_seconds = int((DAY_END - DAY_START).total_seconds())
    agent_counts = NUM_AGENTS_CFG if isinstance(NUM_AGENTS_CFG, list) else [NUM_AGENTS_CFG]
    num_periods, period_duration = len(agent_counts), total_seconds / len(agent_counts)

    agents, retired_agents, next_agent_id, current_period_idx = [], [], 0, -1
    
    # Parent and workers share the same routing state. Initialising the parent
    # up front means the serial path below never runs against unset globals.
    worker_args = (route_graph, nodes, node_tree, swap_nodes, taxi_nodes,
                   utm_coords_dict, wgs_coords_dict, edge_lookup,
                   field, swap_rows, taxi_rows)
    init_worker(*worker_args)
    print(f"Worker pool: {POOL_WORKERS} process(es) of {multiprocessing.cpu_count()} CPUs")
    pool = multiprocessing.Pool(processes=POOL_WORKERS, initializer=init_worker,
                                initargs=worker_args)

    for s in tqdm(range(0, total_seconds, SIMULATION_INTERVAL_SEC), desc="Simulating"):
        new_period_idx = min(int(s // period_duration), num_periods - 1)

        if DEMAND_MODEL:
            while True:
                if trips and trips[0]["departure_time"] <= (DAY_START + timedelta(seconds=s)).isoformat():
                    unallocated_demand.append(trips.pop(0))
                else:
                    break

            ## Assign closest idle agents to unallocated demand ##    
            idle_idxs = [i for i, a in enumerate(agents) if a.state == "IDLE" and not a.assigned_trip]
            while unallocated_demand and idle_idxs:
                source_node = unallocated_demand[0]["source_node"]
                src_x = G.nodes[source_node]['x']
                src_y = G.nodes[source_node]['y']

                # Filter top 4 by squared Euclidean distance using G.nodes attributes
                if len(idle_idxs) > 4:
                    def squared_dist(idx):
                        pos = agents[idx].pos
                        dx = G.nodes[pos]['x'] - src_x
                        dy = G.nodes[pos]['y'] - src_y
                        return dx * dx + dy * dy

                    candidate_idxs = sorted(idle_idxs, key=squared_dist)[:4]
                else:
                    candidate_idxs = idle_idxs

                best_agent_idx = None
                min_cost = float('inf')

                # Find the candidate among the top 4 with the shortest route time
                for idx in candidate_idxs:
                    agent = agents[idx]
                    try:
                        cost = get_cached_cost(agent.pos, source_node)
                        if cost < min_cost:
                            min_cost = cost
                            best_agent_idx = idx
                    except Exception:
                        continue

                if best_agent_idx is not None:
                    agents[best_agent_idx].allocated_trip = unallocated_demand.pop(0)
                    agents[best_agent_idx].assigned_trip = True
                    idle_idxs.remove(best_agent_idx)
                else:
                    break

        if new_period_idx > current_period_idx:
            target = agent_counts[new_period_idx]
            active = [a for a in agents if not a.marked_for_removal]
            if target > len(active):
                for _ in range(target - len(active)):
                    if not DEMAND_MODEL:
                        initial_node = random.choice(nodes)
                    else:
                        initial_node = random.choice(swap_nodes)
                    agents.append(TaxiAgent(next_agent_id, initial_node, spawn_time=s)); next_agent_id += 1

            elif target < len(active):
                active.sort(key=lambda x: x.spawn_time)
                for a in active[:(len(active)-target)]: a.marked_for_removal = True
            current_period_idx = new_period_idx

        still_active = []
        for a in agents:
            if s >= a.busy_until:
                if a.state in ["DRIVING", "DRIVING_TO_SWAP"]:
                    if a.state == "DRIVING_TO_SWAP":
                        station = stations[a.target_node]
                        finish_s = station.process_arrival(s, a.pending_wait_sec, a.id)
                        wait_type = "to_swap"
                        extra_props = {"facility_id": station.station_id}
                    else:
                        finish_s = s + a.pending_wait_sec
                        wait_type = a.current_trip_type # FIX: Use stored trip type
                        extra_props = {}

                    feat = {
                        "type": "Feature", "geometry": {"type": "Point", "coordinates": wgs_coords_dict[a.pos]},
                        "properties": {
                            "agent": a.id, "type": wait_type, "duration_s": (finish_s - s),
                            "total_distance_m": a.arrival_distance, # <--- ADD THIS LINE
                            "start_time": (DAY_START + timedelta(seconds=s)).isoformat(),
                            "end_time": (DAY_START + timedelta(seconds=finish_s)).isoformat(),
                            **extra_props
                        }
                    }
                    a.agent_features.append(feat); a.time_features.append(feat)
                    a.busy_until, a.state = finish_s, "WAITING_COMPLETE"
                elif a.marked_for_removal: retired_agents.append(a); continue
                else: a.state = "IDLE"
            still_active.append(a)
        agents = still_active
        timestamp_str = (DAY_START + timedelta(seconds=s)).isoformat()
        for stat in stations.values(): stat.record_queue(s, timestamp_str)

        idle_idxs = [i for i, a in enumerate(agents) if a.state == "IDLE"]
        if len(idle_idxs) > 20:
            requests = [request_for(agents[i], s) for i in idle_idxs]
            for i, up in zip(idle_idxs, pool.map(calculate_next_activity, requests)):
                apply_update(agents[i], up)
        elif idle_idxs:
            for i in idle_idxs:
                apply_update(agents[i], calculate_next_activity(request_for(agents[i], s)))

    pool.close(); retired_agents.extend(agents)
    plot_station_queues(stations, HISTOGRAM_PLOT)

    # --- ADD THIS BLOCK FOR EXCEL EXPORT ---
    all_swap_records = []
    for stat in stations.values():
        all_swap_records.extend(stat.timestep_records)
        
    cols = [
        "station_id", "timestamp", "sim_step_sec", 
        "swapping_count", "swapping_agent_ids", 
        "queueing_count", "queueing_agent_ids"
    ]
    
    formatted_swap_records = []
    for record in all_swap_records:
        station_id, timestamp_str, current_sec, swap_count, swap_ids, queue_count, queue_ids = record
        formatted_swap_records.append((
            station_id,
            timestamp_str,
            current_sec,
            swap_count,
            ", ".join(map(str, swap_ids)),
            queue_count,
            ", ".join(map(str, queue_ids))
        ))
        
    df_swap = pd.DataFrame(formatted_swap_records, columns=cols)
    df_swap.sort_values(by=["sim_step_sec", "station_id"], inplace=True)
    df_swap.to_excel(SWAP_EXCEL_OUTPUT, index=False)
    # ----------------------------------------

    with open(os.path.join(OUTPUT_DIR, "met_demand.json"), "w") as f: json.dump(met_demand, f, default=to_serializable)

    unmet_demand = [{"idx": d["id"], "request_time": d["departure_time"], "source_facility": d["source_facility"], "dest_facility": d["dest_facility"], "source_category": d["source_category"], "dest_category": d["dest_category"]} for d in unallocated_demand]
    with open(os.path.join(OUTPUT_DIR, "unmet_demand.json"), "w") as f: json.dump(unmet_demand, f, default=to_serializable)

    for a in tqdm(retired_agents, desc="Saving"):
        time_geojson = insert_idle_points({"type":"FeatureCollection","features":a.time_features})
        # with open(f"{OUTPUT_PER_AGENT_DIR}/agent_{a.id:04d}.geojson","w") as f: json.dump({"type":"FeatureCollection","features":a.agent_features}, f, default=to_serializable)
        with open(f"{OUTPUT_PER_AGENT_TIME_DIR}/agent_{a.id:04d}_time.geojson","w") as f: json.dump(time_geojson, f, default=to_serializable)

def plot_station_queues(stations, output_path):
    active_stations = sorted([s for s in stations.values() if s.queue_history], key=lambda x: x.station_id)
    json.dump({s.station_id: s.queue_history for s in stations.values() if s.queue_history}, open(os.path.join(OUTPUT_DIR, "queue_history.json"), "w"), default=to_serializable)
    if not active_stations: return
    n_stations = len(active_stations)
    cols = 1 if n_stations == 1 else 2
    rows = (n_stations + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12 if cols == 1 else 24, rows * 5), sharex=True, sharey=True, squeeze=False)
    flat_axes = axes.flatten()
    
    # Calculate max pressure for consistent y-axis scaling
    global_y_max = max(max(s.queue_history) / s.num_servers for s in active_stations)
    
    for i, s in enumerate(active_stations):
        ax = flat_axes[i]
        times = [DAY_START + timedelta(seconds=sec) for sec in range(0, len(s.queue_history)*SIMULATION_INTERVAL_SEC, SIMULATION_INTERVAL_SEC)]
        pressure = np.array(s.queue_history) / s.num_servers
        
        ax.step(times, pressure, where='post', color='#2c3e50', lw=2.5)
        ax.fill_between(times, pressure, step='post', alpha=0.3, color='#3498db')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.8, lw=2.5)
        
        # Formatting
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: mdates.num2date(x).strftime('%I%p').lower().lstrip('0')))
        ax.set_ylim(0, global_y_max * 1.1)
        
        # --- MODIFIED LINE BELOW ---
        ax.set_title(f"Station {s.station_id} ({s.num_servers} Posts)", fontsize=18, fontweight='bold')
        # ---------------------------

    # Hide unused subplots if any
    for j in range(i + 1, len(flat_axes)):
        flat_axes[j].axis('off')

    plt.tight_layout(pad=4.0)
    plt.savefig(output_path, dpi=300)

if __name__ == "__main__":
    main()