#!/usr/bin/env python3
"""
Tabu Search P-Median for Swap Stations on Road Networks
- Optimized: incremental distance updates for speed
- All road network nodes are candidate locations
- Taxi ranks are demand points
- Precomputes shortest-path distances
- Normal mode outputs GeoJSON, analysis mode outputs CSV
- Runs each optimization 10 times and keeps the best solution
"""

import os
import sys
import json
import networkx as nx
import osmnx as ox
import numpy as np
from shapely.geometry import Point, LineString, mapping, shape
import argparse
import csv
import random
from tqdm import tqdm
from joblib import Parallel, delayed
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import load_scenario, add_scenario_argument

# ---------------------------
# ARGUMENTS
# ---------------------------
parser = argparse.ArgumentParser(description="Tabu Search P-Median for swap stations")
parser.add_argument("num_locations", nargs="?", type=int, help="Number of swap stations")
parser.add_argument("-analysis", action="store_true", help="Run analysis for 1–20 facilities")
add_scenario_argument(parser)
args = parser.parse_args()

# ---------------------------
# SETTINGS
# ---------------------------
SCENARIO = load_scenario(args.scenario)
FOLDER_NAME = SCENARIO.folder_name
INPUT_DIR = SCENARIO.input_dir
OUTPUT_DIR = SCENARIO.output_dir
NUM_LOCATIONS = args.num_locations if args.num_locations else 5
ROAD_NETWORK_FILE = os.path.join(INPUT_DIR, "road_network.graphml")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "facility_analysis_pmedian_tabu.csv")
INPUT_GEOJSON = os.path.join(INPUT_DIR, "taxi_ranks.geojson")
OUTPUT_GEOJSON = os.path.join(OUTPUT_DIR, "swap_stations_tabu.geojson")
BOUNDARY_FILE = os.path.join(INPUT_DIR, "area.geojson")

# ---------------------------
# LOAD ACCRA BOUNDARY
# ---------------------------
with open(BOUNDARY_FILE, "r") as f:
    data = json.load(f)
accra_polygon = shape(data["features"][0]["geometry"])
print("Polygon loaded.")

# ---------------------------
# LOAD TAXI RANKS (DEMAND POINTS)
# ---------------------------
with open(INPUT_GEOJSON, "r") as f:
    data = json.load(f)

points = []
for feature in data["features"]:
    x, y = feature["geometry"]["coordinates"]
    pt = Point(x, y)
    if accra_polygon.contains(pt):
        points.append([x, y])
points = np.array(points)
print(f"Total taxi ranks inside Accra boundary: {len(points)}")

# ---------------------------
# LOAD ROAD NETWORK
# ---------------------------
if os.path.exists(ROAD_NETWORK_FILE):
    G = ox.load_graphml(ROAD_NETWORK_FILE)
else:
    G = ox.graph_from_place("Accra, Ghana", network_type="drive", simplify=True)
    G = ox.distance.add_edge_lengths(G)
    largest_cc_nodes = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_cc_nodes).copy()
    ox.save_graphml(G, ROAD_NETWORK_FILE)

G = ox.distance.add_edge_lengths(G)
largest_cc = max(nx.strongly_connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()
print(f"Road network loaded: {len(G.nodes)} nodes, {len(G.edges)} edges.")

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def nearest_node(lon, lat):
    return ox.nearest_nodes(G, X=lon, Y=lat)

taxi_nodes = [nearest_node(x, y) for x, y in points]

# ---------------------------
# CANDIDATE NODES = ALL ROAD NODES
# ---------------------------
candidate_nodes = list(G.nodes)
print(f"Using all {len(candidate_nodes)} road network nodes as candidate medoids.")

candidate_index = {node: idx for idx, node in enumerate(candidate_nodes)}

# ---------------------------
# PRECOMPUTE SHORTEST-PATH DISTANCES
# ---------------------------
def compute_single_shortest_paths(tn):
    return tn, nx.single_source_dijkstra_path_length(G, tn, weight="length")

print("Precomputing taxi → node shortest paths (parallel)...")
results = Parallel(n_jobs=-1)(delayed(compute_single_shortest_paths)(tn) for tn in tqdm(taxi_nodes))
all_shortest_paths = {tn: paths for tn, paths in results}
print("Precomputation complete.")

def distance(tn, fn):
    return all_shortest_paths.get(tn, {}).get(fn, np.inf)

# ---------------------------
# TABU SEARCH P-MEDIAN (Incremental Distance Updates)
# ---------------------------
def tabu_search_p_median(taxi_nodes, candidate_nodes, k=None, max_iter=100, tabu_tenure=10, neighborhood_size=50):
    """
    Tabu Search with incremental distance updates
    """
    n_taxi = len(taxi_nodes)
    k = k if k else 5
    n_candidates = len(candidate_nodes)

    # Random initialization
    medoid_indices = random.sample(range(n_candidates), k)
    medoids = [candidate_nodes[i] for i in medoid_indices]

    # Compute initial distance matrix: n_taxi x k
    dist_matrix = np.zeros((n_taxi, k))
    for i, tn in enumerate(taxi_nodes):
        for j, fn in enumerate(medoids):
            dist_matrix[i, j] = distance(tn, fn)
    assigned = np.argmin(dist_matrix, axis=1)
    total_cost = dist_matrix[np.arange(n_taxi), assigned].sum()
    best_cost = total_cost
    best_solution = medoid_indices.copy()

    # Tabu list
    tabu_list = {}

    for iter_count in range(1, max_iter + 1):
        neighborhood = []

        # Generate random neighborhood swaps
        for _ in range(neighborhood_size):
            m_idx = random.randint(0, k - 1)
            c_idx = random.randint(0, n_candidates - 1)
            if c_idx in medoid_indices:
                continue
            neighborhood.append((m_idx, c_idx))

        best_move = None
        best_move_cost = np.inf
        best_move_dist_matrix = None
        best_move_assigned = None

        # Evaluate neighborhood with incremental update
        for m_idx, c_idx in neighborhood:
            if tabu_list.get((m_idx, c_idx), 0) > iter_count:
                continue  # tabu

            # Only update the column corresponding to the swapped medoid
            new_dist_matrix = dist_matrix.copy()
            for i, tn in enumerate(taxi_nodes):
                new_dist_matrix[i, m_idx] = distance(tn, candidate_nodes[c_idx])

            new_assigned = np.argmin(new_dist_matrix, axis=1)
            new_cost = new_dist_matrix[np.arange(n_taxi), new_assigned].sum()

            # Aspiration: allow tabu if improves global best
            if new_cost < best_cost or new_cost < best_move_cost:
                best_move = (m_idx, c_idx)
                best_move_cost = new_cost
                best_move_dist_matrix = new_dist_matrix
                best_move_assigned = new_assigned

        if best_move is None:
            print(f"Iteration {iter_count}: no non-tabu move found, stopping early.")
            break

        # Apply best move
        m_idx, c_idx = best_move
        medoid_indices[m_idx] = c_idx
        medoids = [candidate_nodes[i] for i in medoid_indices]
        dist_matrix = best_move_dist_matrix
        assigned = best_move_assigned
        total_cost = best_move_cost

        # Update tabu list
        tabu_list[best_move] = iter_count + tabu_tenure

        # Update global best
        if total_cost < best_cost:
            best_cost = total_cost
            best_solution = medoid_indices.copy()

        print(f"Iteration {iter_count}: total cost = {total_cost:.2f}, best cost = {best_cost:.2f}")

    # Return best solution
    best_medoids = [candidate_nodes[i] for i in best_solution]
    final_dist_matrix = np.zeros((n_taxi, k))
    for i, tn in enumerate(taxi_nodes):
        for j, fn in enumerate(best_medoids):
            final_dist_matrix[i, j] = distance(tn, fn)
    final_assigned = np.argmin(final_dist_matrix, axis=1)

    return best_medoids, final_assigned, final_dist_matrix

# ---------------------------
# RUN MULTIPLE TIMES AND KEEP BEST
# ---------------------------
def best_of_multiple_runs(search_fn, taxi_nodes, candidate_nodes, k, runs=10, **kwargs):
    best_total_cost = np.inf
    best_medoids, best_assigned, best_dist_matrix = None, None, None

    for run in range(1, runs + 1):
        medoids, assigned, dist_matrix = search_fn(
            taxi_nodes, candidate_nodes, k=k, **kwargs
        )
        total_cost = dist_matrix[np.arange(len(taxi_nodes)), assigned].sum()
        print(f"Run {run}/{runs} total cost = {total_cost:.2f}")
        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_medoids, best_assigned, best_dist_matrix = medoids, assigned, dist_matrix

    print(f"Best total cost after {runs} runs = {best_total_cost:.2f}")
    return best_medoids, best_assigned, best_dist_matrix

# ---------------------------
# ANALYSIS MODE
# ---------------------------
if args.analysis:
    print("Running analysis mode for 1–20 swap stations...")
    start_time = time.time()  # Start timer
    results = []

    for k in range(1, 21):
        print(f"\nComputing P-Median for {k} facilities...")
        medoids, labels, dist_matrix = best_of_multiple_runs(
            tabu_search_p_median,
            taxi_nodes,
            candidate_nodes,
            k=k,
            runs=10,
            max_iter=100,
            tabu_tenure=10,
            neighborhood_size=50
        )
        min_distances = dist_matrix[np.arange(len(taxi_nodes)), labels]
        avg_distance = min_distances.mean()
        max_distance = min_distances.max()
        results.append((k, avg_distance, max_distance))
        print(f"{k} facilities: avg = {avg_distance:.2f} m, max = {max_distance:.2f} m")

    end_time = time.time()  # End timer
    total_time = end_time - start_time
    print(f"\nTotal analysis runtime: {total_time:.2f} seconds")

    # Save CSV with total time as final row
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_facilities", "average_route_distance_m", "max_route_distance_m"])
        writer.writerows(results)
        writer.writerow(["total_time_seconds", total_time, ""])  # Add runtime row

    print(f"\nAnalysis CSV saved to {OUTPUT_CSV}")
    exit()

# ---------------------------
# NORMAL MODE: OUTPUT GEOJSON
# ---------------------------
medoids, labels, dist_matrix = best_of_multiple_runs(
    tabu_search_p_median,
    taxi_nodes,
    candidate_nodes,
    k=NUM_LOCATIONS,
    runs=10,
    max_iter=100,
    tabu_tenure=10,
    neighborhood_size=50
)

medoid_coords = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in medoids])
features = []

# Facility points
for i, c in enumerate(medoid_coords):
    features.append({
        "type": "Feature",
        "properties": {"facility_id": i, "type": "facility"},
        "geometry": mapping(Point(c))
    })

# Lines linking taxi ranks to nearest facility
for idx, pt in enumerate(points):
    fid = labels[idx]
    line = LineString([pt, medoid_coords[fid]])
    features.append({
        "type": "Feature",
        "properties": {"facility_id": int(fid), "type": "link_to_facility"},
        "geometry": mapping(line)
    })

fc = {"type": "FeatureCollection", "features": features}
with open(OUTPUT_GEOJSON, "w") as f:
    json.dump(fc, f)

print(f"Saved {NUM_LOCATIONS} swap stations & links to {OUTPUT_GEOJSON}")
