#!/usr/bin/env python3
"""
Simulated-Annealing P-Median for Swap Stations with Road-Network Distances
Supports:
- Normal mode: generate N swap stations + GeoJSON links
- Analysis mode: compute average/max route distances for 1–20 facilities
- Candidate selection: full network nodes or sampled subset (--sample N)
"""

import json
import geopandas as gpd
from shapely.geometry import Point, LineString, mapping, shape
import osmnx as ox
import networkx as nx
import numpy as np
import argparse
from tqdm import tqdm
import random
from joblib import Parallel, delayed
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import load_scenario, add_scenario_argument

# ---------------------------
# ARGUMENTS
# ---------------------------
parser = argparse.ArgumentParser(description="Simulated-Annealing P-Median for Swap Stations")
parser.add_argument("num_locations", nargs="?", type=int, help="Number of swap stations")
parser.add_argument("-analysis", action="store_true",
                    help="Run road-network analysis from 1 to 20 facilities")
parser.add_argument("--sample", type=int, default=None,
                    help="Use sampled candidate nodes (optional, default: all nodes)")
add_scenario_argument(parser)
args = parser.parse_args()

SCENARIO = load_scenario(args.scenario)
FOLDER_NAME = SCENARIO.folder_name
INPUT_DIR = SCENARIO.input_dir
OUTPUT_DIR = SCENARIO.output_dir
NUM_LOCATIONS = args.num_locations if args.num_locations else 5
ROAD_NETWORK_FILE = os.path.join(INPUT_DIR, "road_network.graphml")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "facility_analysis_pmedian_sa.csv")
INPUT_GEOJSON = os.path.join(INPUT_DIR, "taxi_ranks.geojson")
OUTPUT_GEOJSON = os.path.join(OUTPUT_DIR, "swap_stations_sa.geojson")
BOUNDARY_FILE = os.path.join(INPUT_DIR, "area.geojson")

# ---------------------------
# LOAD ACCRA BOUNDARY
# ---------------------------
with open(BOUNDARY_FILE, "r") as f:
    data = json.load(f)
accra_polygon = shape(data["features"][0]["geometry"])
print("Accra polygon loaded.")

# ---------------------------
# LOAD TAXI RANKS
# ---------------------------
with open(INPUT_GEOJSON, "r") as f:
    data = json.load(f)
points = []
node_features = []
for feature in data["features"]:
    x, y = feature["geometry"]["coordinates"]
    pt = Point(x, y)
    if accra_polygon.contains(pt):
        points.append([x, y])
        node_features.append(feature)
points = np.array(points)
print(f"Total taxi ranks inside Accra boundary: {len(points)}")

# ---------------------------
# LOAD ROAD NETWORK
# ---------------------------
print("Downloading Accra road network...")
# G = ox.graph_from_place("Accra, Ghana", network_type="drive", simplify=True)

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
# PARALLEL SHORTEST-PATH PRECOMPUTATION
# ---------------------------
def compute_single_shortest_paths(tn):
    return tn, nx.single_source_dijkstra_path_length(G, tn, weight="length")

print("Precomputing taxi → node shortest paths (parallel)...")
results = Parallel(n_jobs=-1)(delayed(compute_single_shortest_paths)(tn) for tn in tqdm(taxi_nodes))
all_shortest_paths = {tn: paths for tn, paths in results}
print("Precomputation complete.")

def dist(tn, fn):
    return all_shortest_paths.get(tn, {}).get(fn, np.inf)

# ---------------------------
# BUILD CANDIDATE NODE SET (works for both modes)
# ---------------------------
if args.sample:
    candidate_nodes = random.sample(list(G.nodes), args.sample)
    print(f"Using {len(candidate_nodes)} sampled candidate nodes for SA.")
else:
    candidate_nodes = list(G.nodes)
    print(f"Using all {len(candidate_nodes)} network nodes for SA (unsampled).")

# ---------------------------
# SIMULATED ANNEALING P-MEDIAN
# ---------------------------
def simulated_annealing(taxi_nodes, candidate_nodes, k,
                        T0=1.0, Tmin=0.0001, alpha=0.97,
                        iter_per_temp=1500, restarts=3):
    n_taxi = len(taxi_nodes)
    best_solution = None
    best_cost = np.inf

    for r in range(restarts):
        print(f"\nRestart {r+1}/{restarts}...")
        medoids = random.sample(candidate_nodes, k)
        dist_matrix = np.zeros((n_taxi, k))
        for i, tn in enumerate(taxi_nodes):
            for j, fn in enumerate(medoids):
                dist_matrix[i, j] = dist(tn, fn)
        assigned = np.argmin(dist_matrix, axis=1)
        cost = dist_matrix[np.arange(n_taxi), assigned].sum()

        T = T0
        while T > Tmin:
            for _ in range(iter_per_temp):
                m_idx = random.randint(0, k-1)
                new_fn = random.choice(candidate_nodes)
                if new_fn in medoids:
                    continue
                new_medoids = medoids.copy()
                new_medoids[m_idx] = new_fn

                new_dist_matrix = np.zeros((n_taxi, k))
                for i, tn in enumerate(taxi_nodes):
                    for j, fn in enumerate(new_medoids):
                        new_dist_matrix[i, j] = dist(tn, fn)
                new_assigned = np.argmin(new_dist_matrix, axis=1)
                new_cost = new_dist_matrix[np.arange(n_taxi), new_assigned].sum()

                delta = new_cost - cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    medoids = new_medoids
                    dist_matrix = new_dist_matrix
                    assigned = new_assigned
                    cost = new_cost

            T *= alpha  # Geometric cooling

        if cost < best_cost:
            best_solution = (medoids.copy(), assigned.copy(), dist_matrix.copy())
            best_cost = cost

    return best_solution

# ---------------------------
# ANALYSIS MODE
# ---------------------------
if args.analysis:
    print("Running analysis mode for 1–20 facilities using SA...")
    start_time = time.time()  # Start timer
    results = []
    for k in range(1, 21):
        print(f"\nComputing {k} facilities...")
        medoids, labels, dist_matrix = simulated_annealing(taxi_nodes, candidate_nodes, k)
        min_distances = dist_matrix[np.arange(len(taxi_nodes)), labels]
        avg_distance = min_distances.mean()
        max_distance = min_distances.max()
        results.append((k, avg_distance, max_distance))
        print(f"{k} facilities: avg={avg_distance:.2f} m, max={max_distance:.2f} m")

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
medoids, labels, dist_matrix = simulated_annealing(taxi_nodes, candidate_nodes, NUM_LOCATIONS)
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
