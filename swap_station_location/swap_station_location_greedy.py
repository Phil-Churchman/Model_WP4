#!/usr/bin/env python3
"""
Greedy P-Median for Swap Stations on Road Networks
- All road network nodes are candidate locations
- Taxi ranks are demand points
- Precomputes shortest-path distances for speed
- Normal mode outputs GeoJSON, analysis mode outputs CSV
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
parser = argparse.ArgumentParser(description="Greedy P-Median for swap stations")
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
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "facility_analysis_pmedian_greedy.csv")
INPUT_GEOJSON = os.path.join(INPUT_DIR, "taxi_ranks.geojson")
OUTPUT_GEOJSON = os.path.join(OUTPUT_DIR, "swap_stations_greedy.geojson")
BOUNDARY_FILE = os.path.join(INPUT_DIR, "area.geojson")

# ---------------------------
# LOAD ACCRA BOUNDARY
# ---------------------------
with open(BOUNDARY_FILE, "r") as f:
    data = json.load(f)
accra_polygon = shape(data["features"][0]["geometry"])
print("Accra polygon loaded.")

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
# GREEDY P-MEDIAN (original logic, all swaps, random init)
# ---------------------------
def greedy_p_median(taxi_nodes, candidate_nodes, max_iter=10, k=None):
    n_taxi = len(taxi_nodes)
    k = k if k else 5
    n_candidates = len(candidate_nodes)

    # Random initialization from candidate nodes
    medoid_indices = random.sample(range(n_candidates), k)
    medoids = [candidate_nodes[i] for i in medoid_indices]

    # Compute initial distance matrix
    dist_matrix = np.zeros((n_taxi, k))
    for i, tn in enumerate(taxi_nodes):
        for j, fn in enumerate(medoids):
            dist_matrix[i, j] = distance(tn, fn)

    assigned = np.argmin(dist_matrix, axis=1)
    total_cost = dist_matrix[np.arange(n_taxi), assigned].sum()

    iter_count = 0
    improved = True
    while improved and iter_count < max_iter:
        iter_count += 1
        improved = False
        for m_idx in range(k):
            best_cost = total_cost
            best_candidate = medoid_indices[m_idx]
            for c_idx in range(n_candidates):
                if c_idx in medoid_indices:
                    continue
                new_medoid_indices = medoid_indices.copy()
                new_medoid_indices[m_idx] = c_idx
                new_medoids = [candidate_nodes[i] for i in new_medoid_indices]

                # Compute new distance matrix for these medoids
                new_dist_matrix = np.zeros((n_taxi, k))
                for i, tn in enumerate(taxi_nodes):
                    for j, fn in enumerate(new_medoids):
                        new_dist_matrix[i, j] = distance(tn, fn)

                new_assigned = np.argmin(new_dist_matrix, axis=1)
                new_cost = new_dist_matrix[np.arange(n_taxi), new_assigned].sum()

                if new_cost < best_cost:
                    best_cost = new_cost
                    best_candidate = c_idx
                    best_dist_matrix = new_dist_matrix
                    best_assigned = new_assigned

            # Apply best swap if it improves cost
            if best_candidate != medoid_indices[m_idx]:
                medoid_indices[m_idx] = best_candidate
                medoids = [candidate_nodes[i] for i in medoid_indices]
                dist_matrix = best_dist_matrix
                assigned = best_assigned
                total_cost = best_cost
                improved = True

        print(f"Iteration {iter_count}: total cost = {total_cost:.2f}")

    return medoids, assigned, dist_matrix

# ---------------------------
# ANALYSIS MODE
# ---------------------------
if args.analysis:
    print("Running analysis mode for 1–20 swap stations...")
    start_time = time.time()  # Start timer
    results = []
    for k in range(1, 21):
        print(f"\nComputing P-Median for {k} facilities...")
        medoids, labels, dist_matrix = greedy_p_median(taxi_nodes, candidate_nodes, max_iter=10, k=k)
        min_distances = dist_matrix[np.arange(len(taxi_nodes)), labels]
        avg_distance = min_distances.mean()
        max_distance = min_distances.max()
        results.append((k, avg_distance, max_distance))
        print(f"{k} facilities: avg = {avg_distance:.2f} m, max = {max_distance:.2f} m")

    end_time = time.time()  # End timer
    total_time = end_time - start_time
    print(f"\nTotal analysis runtime: {total_time:.2f} seconds")
    # Save CSV
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
medoids, labels, dist_matrix = greedy_p_median(taxi_nodes, candidate_nodes, max_iter=10, k=NUM_LOCATIONS)
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
