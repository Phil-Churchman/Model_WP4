#!/usr/bin/env python3
"""
Genetic Algorithm P-Median for Swap Stations on Road Networks
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
parser = argparse.ArgumentParser(description="Genetic Algorithm P-Median for swap stations")
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
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "facility_analysis_pmedian_genetic.csv")
INPUT_GEOJSON = os.path.join(INPUT_DIR, "taxi_ranks.geojson")
OUTPUT_GEOJSON = os.path.join(OUTPUT_DIR, "swap_stations_genetic.geojson")
BOUNDARY_FILE = os.path.join(INPUT_DIR, "area.geojson")

POPULATION_SIZE = 60
MAX_GENERATIONS = 200
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 3

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
print(f"Using all {len(candidate_nodes)} road network nodes as candidate medoids.")

candidate_index = {node: idx for idx, node in enumerate(candidate_nodes)}

def compute_single_shortest_paths(tn):
    return tn, nx.single_source_dijkstra_path_length(G, tn, weight="length")

print("Precomputing taxi → node shortest paths (parallel)...")
results = Parallel(n_jobs=-1)(delayed(compute_single_shortest_paths)(tn) for tn in tqdm(taxi_nodes))
all_shortest_paths = {tn: paths for tn, paths in results}
print("Precomputation complete.")

def distance(tn, fn):
    return all_shortest_paths.get(tn, {}).get(fn, np.inf)

# ---------------------------
# GENETIC ALGORITHM P-MEDIAN
# ---------------------------
def compute_cost(medoid_indices):
    """Compute total distance from taxis to nearest facility for a set of medoids"""
    medoids = [candidate_nodes[i] for i in medoid_indices]
    dist_matrix = np.zeros((len(taxi_nodes), len(medoids)))
    for i, tn in enumerate(taxi_nodes):
        for j, fn in enumerate(medoids):
            dist_matrix[i, j] = distance(tn, fn)
    assigned = np.argmin(dist_matrix, axis=1)
    total_cost = dist_matrix[np.arange(len(taxi_nodes)), assigned].sum()
    return total_cost, assigned, dist_matrix

def initialize_population(k):
    population = [random.sample(range(len(candidate_nodes)), k) for _ in range(POPULATION_SIZE)]
    return population

def tournament_selection(population, fitnesses):
    selected = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
    selected.sort(key=lambda x: x[1])
    return selected[0][0]

def crossover(parent1, parent2):
    # Uniform crossover
    k = len(parent1)
    child = []
    chosen = set()
    for i in range(k):
        gene = parent1[i] if random.random() < 0.5 else parent2[i]
        if gene not in chosen:
            child.append(gene)
            chosen.add(gene)
    # Fill remaining genes randomly if duplicates removed some
    while len(child) < k:
        gene = random.randrange(len(candidate_nodes))
        if gene not in chosen:
            child.append(gene)
            chosen.add(gene)
    return child

def mutate(individual):
    k = len(individual)
    for i in range(k):
        if random.random() < MUTATION_RATE:
            new_gene = random.randrange(len(candidate_nodes))
            while new_gene in individual:
                new_gene = random.randrange(len(candidate_nodes))
            individual[i] = new_gene
    return individual

def genetic_algorithm(k):
    population = initialize_population(k)
    best_cost = float('inf')
    best_solution = None
    best_assigned = None
    best_dist_matrix = None

    for generation in range(1, MAX_GENERATIONS + 1):
        fitnesses = []
        for individual in population:
            cost, assigned, dist_matrix = compute_cost(individual)
            fitnesses.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_solution = individual.copy()
                best_assigned = assigned
                best_dist_matrix = dist_matrix

        new_population = []

        # Elitism: keep best individual
        new_population.append(best_solution.copy())

        # Generate rest of new population
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population
        print(f"Generation {generation}: best cost = {best_cost:.2f}")

    final_medoids = [candidate_nodes[i] for i in best_solution]
    return final_medoids, best_assigned, best_dist_matrix

# ---------------------------
# ANALYSIS MODE
# ---------------------------
if args.analysis:
    print("Running analysis mode for 1–20 swap stations...")
    start_time = time.time()
    results = []
    for k in range(1, 21):
        print(f"\nComputing P-Median for {k} facilities using GA...")
        medoids, labels, dist_matrix = genetic_algorithm(k)
        min_distances = dist_matrix[np.arange(len(taxi_nodes)), labels]
        avg_distance = min_distances.mean()
        max_distance = min_distances.max()
        results.append((k, avg_distance, max_distance))
        print(f"{k} facilities: avg = {avg_distance:.2f} m, max = {max_distance:.2f} m")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal analysis runtime: {total_time:.2f} seconds")

    # Save CSV
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
medoids, labels, dist_matrix = genetic_algorithm(NUM_LOCATIONS)

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
