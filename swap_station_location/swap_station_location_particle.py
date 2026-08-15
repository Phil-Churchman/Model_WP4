#!/usr/bin/env python3
"""
Particle Swarm Optimization (PSO) P-Median for Swap Stations on Road Networks
- All road network nodes are candidate locations
- Taxi ranks are demand points
- Precomputes shortest-path distances
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
parser = argparse.ArgumentParser(description="PSO P-Median for swap stations")
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
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "facility_analysis_pmedian_particle.csv")
INPUT_GEOJSON = os.path.join(INPUT_DIR, "taxi_ranks.geojson")
OUTPUT_GEOJSON = os.path.join(OUTPUT_DIR, "swap_stations_particle.geojson")
BOUNDARY_FILE = os.path.join(INPUT_DIR, "area.geojson")

# PSO parameters
NUM_PARTICLES = 30
MAX_ITERATIONS = 100
W = 0.7      # inertia weight
C1 = 1.5     # cognitive coefficient
C2 = 1.5     # social coefficient

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

candidate_nodes = list(G.nodes)
candidate_index = {node: idx for idx, node in enumerate(candidate_nodes)}
print(f"Using all {len(candidate_nodes)} road network nodes as candidate medoids.")

# Precompute shortest-path distances
def compute_single_shortest_paths(tn):
    return tn, nx.single_source_dijkstra_path_length(G, tn, weight="length")

print("Precomputing taxi → node shortest paths (parallel)...")
results = Parallel(n_jobs=-1)(delayed(compute_single_shortest_paths)(tn) for tn in tqdm(taxi_nodes))
all_shortest_paths = {tn: paths for tn, paths in results}
print("Precomputation complete.")

def distance(tn, fn):
    return all_shortest_paths.get(tn, {}).get(fn, np.inf)

# ---------------------------
# PARTICLE SWARM OPTIMIZATION P-MEDIAN
# ---------------------------
def compute_cost(medoids_indices):
    medoids = [candidate_nodes[i] for i in medoids_indices]
    dist_matrix = np.zeros((len(taxi_nodes), len(medoids)))
    for i, tn in enumerate(taxi_nodes):
        for j, fn in enumerate(medoids):
            dist_matrix[i, j] = distance(tn, fn)
    assigned = np.argmin(dist_matrix, axis=1)
    total_cost = dist_matrix[np.arange(len(taxi_nodes)), assigned].sum()
    return total_cost, assigned, dist_matrix

def initialize_particles(k):
    particles = []
    velocities = []
    for _ in range(NUM_PARTICLES):
        p = random.sample(range(len(candidate_nodes)), k)
        particles.append(p)
        v = np.zeros(k)  # velocities in index-space
        velocities.append(v)
    return particles, velocities

def particle_swarm_p_median(k):
    particles, velocities = initialize_particles(k)
    personal_best = particles.copy()
    personal_best_cost = [compute_cost(p)[0] for p in particles]

    global_best_idx = np.argmin(personal_best_cost)
    global_best = personal_best[global_best_idx].copy()
    global_best_cost = personal_best_cost[global_best_idx]
    global_best_assigned, global_best_dist_matrix = compute_cost(global_best)[1:]

    for iteration in range(1, MAX_ITERATIONS + 1):
        for i, particle in enumerate(particles):
            # Update velocity (continuous index-space)
            vel = velocities[i]
            for d in range(k):
                r1, r2 = random.random(), random.random()
                cognitive = C1 * r1 * (personal_best[i][d] - particle[d])
                social = C2 * r2 * (global_best[d] - particle[d])
                vel[d] = W * vel[d] + cognitive + social
            # Update position: round to nearest integer index and ensure unique medoids
            new_particle = [int(round(particle[d] + vel[d])) for d in range(k)]
            new_particle = [min(max(0, idx), len(candidate_nodes) - 1) for idx in new_particle]
            # Make unique
            new_particle = list(dict.fromkeys(new_particle))
            while len(new_particle) < k:
                idx = random.randint(0, len(candidate_nodes) - 1)
                if idx not in new_particle:
                    new_particle.append(idx)
            particles[i] = new_particle
            velocities[i] = vel

            # Evaluate cost
            cost, assigned, dist_matrix = compute_cost(new_particle)
            if cost < personal_best_cost[i]:
                personal_best[i] = new_particle.copy()
                personal_best_cost[i] = cost
                # Update global best
                if cost < global_best_cost:
                    global_best = new_particle.copy()
                    global_best_cost = cost
                    global_best_assigned = assigned
                    global_best_dist_matrix = dist_matrix

        print(f"Iteration {iteration}: global best cost = {global_best_cost:.2f}")

    best_medoids = [candidate_nodes[i] for i in global_best]
    return best_medoids, global_best_assigned, global_best_dist_matrix

# ---------------------------
# ANALYSIS MODE
# ---------------------------
if args.analysis:
    print("Running analysis mode for 1–20 swap stations...")
    start_time = time.time()
    results = []
    for k in range(1, 21):
        print(f"\nComputing P-Median for {k} facilities using PSO...")
        medoids, labels, dist_matrix = particle_swarm_p_median(k)
        min_distances = dist_matrix[np.arange(len(taxi_nodes)), labels]
        avg_distance = min_distances.mean()
        max_distance = min_distances.max()
        results.append((k, avg_distance, max_distance))
        print(f"{k} facilities: avg = {avg_distance:.2f} m, max = {max_distance:.2f} m")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal analysis runtime: {total_time:.2f} seconds")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_facilities", "average_route_distance_m", "max_route_distance_m"])
        writer.writerows(results)
        writer.writerow(["total_time_seconds", total_time, ""])
    print(f"\nAnalysis CSV saved to {OUTPUT_CSV}")
    exit()

# ---------------------------
# NORMAL MODE: OUTPUT GEOJSON
# ---------------------------
medoids, labels, dist_matrix = particle_swarm_p_median(NUM_LOCATIONS)

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
