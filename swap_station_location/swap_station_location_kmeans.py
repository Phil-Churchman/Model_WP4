#!/usr/bin/env python3
"""
Offline swap station KMeans + Voronoi analysis
Uses local GeoJSON files:
- Accra boundary: geojson_files/accra_polygon.geojson
- Taxi ranks: geojson_files/taxi_ranks.geojson
Generates swap station locations, Voronoi polygons, and links.
"""

import os
import sys
import json
import argparse
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, mapping
from shapely.ops import voronoi_diagram
from sklearn.cluster import KMeans
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import load_scenario, add_scenario_argument

# ---------------------------
# ARGUMENTS
# ---------------------------
parser = argparse.ArgumentParser(description="Offline swap station analysis")
parser.add_argument("num_locations", nargs="?", type=int, help="Number of swap stations")
parser.add_argument("-analysis", action="store_true", help="Run analysis for 1-20 facilities")
add_scenario_argument(parser)
args = parser.parse_args()

# ---------------------------
# FILES
# ---------------------------
SCENARIO = load_scenario(args.scenario)
FOLDER_NAME = SCENARIO.folder_name
INPUT_DIR = SCENARIO.input_dir
OUTPUT_DIR = SCENARIO.output_dir
NUM_LOCATIONS = args.num_locations if args.num_locations else 5
ROAD_NETWORK_FILE = os.path.join(INPUT_DIR, "road_network.graphml")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "facility_analysis_pmedian_kmeans.csv")
INPUT_GEOJSON = os.path.join(INPUT_DIR, "taxi_ranks.geojson")
OUTPUT_GEOJSON = os.path.join(OUTPUT_DIR, "swap_stations_pmedian_kmeans.geojson")
BOUNDARY_FILE = os.path.join(INPUT_DIR, "area.geojson")

# ---------------------------
# LOAD ACCRA BOUNDARY
# ---------------------------
accra_gdf = gpd.read_file(BOUNDARY_FILE)
accra_polygon = accra_gdf.geometry.iloc[0]

# ---------------------------
# LOAD TAXI RANKS
# ---------------------------
taxi_gdf = gpd.read_file(INPUT_GEOJSON)
taxi_points = []
for idx, row in taxi_gdf.iterrows():
    pt = row.geometry
    if accra_polygon.contains(pt):
        taxi_points.append([pt.x, pt.y])
taxi_points = np.array(taxi_points)
print(f"Total taxi ranks inside Accra boundary: {len(taxi_points)}")

# ---------------------------
# ANALYSIS MODE
# ---------------------------
if args.analysis:
    import osmnx as ox
    import networkx as nx
    from tqdm import tqdm
    import time 

    print("Running route-distance analysis for 1-20 facilities...")
    start_time = time.time()  # Start timer
    # ---------------------------
    # Build road network (lat/lon)
    # ---------------------------
    print("Downloading road network for Accra...")
    # G = ox.graph_from_polygon(accra_polygon, network_type='drive')  # unprojected (lat/lon)

    if os.path.exists(ROAD_NETWORK_FILE):
        G = ox.load_graphml(ROAD_NETWORK_FILE)
    else:
        G = ox.graph_from_place("Accra, Ghana", network_type="drive", simplify=True)
        G = ox.distance.add_edge_lengths(G)
        largest_cc_nodes = max(nx.strongly_connected_components(G), key=len)
        G = G.subgraph(largest_cc_nodes).copy()
        ox.save_graphml(G, ROAD_NETWORK_FILE)

    results = []

    # ---------------------------
    # Loop over number of facilities
    # ---------------------------
    for k in range(1, 21):
        print(f"\nClustering into {k} facilities...")

        # 1. Compute Euclidean KMeans centroids
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(taxi_points)
        centroids = kmeans.cluster_centers_

        # 2. Snap taxi points and centroids to nearest network nodes (lat/lon graph)
        taxi_nodes = [ox.nearest_nodes(G, X=pt[0], Y=pt[1]) for pt in taxi_points]
        facility_nodes = [ox.nearest_nodes(G, X=c[0], Y=c[1]) for c in centroids]

        # 3. Project graph to meters for accurate distance computation
        G_proj = ox.project_graph(G)

        # 4. Compute shortest route distances using multi-source Dijkstra
        lengths = nx.multi_source_dijkstra_path_length(G_proj, facility_nodes, weight='length')

        # 5. For each taxi rank, get distance to nearest facility
        min_distances = [lengths.get(t_node, float('inf')) for t_node in taxi_nodes]

        # 6. Compute average and max distances
        avg_distance = np.mean(min_distances)
        max_distance = np.max(min_distances)
        results.append((k, avg_distance, max_distance))
        print(f"{k} facilities: avg={avg_distance:.2f} m, max={max_distance:.2f} m")

    end_time = time.time()  # End timer
    total_time = end_time - start_time
    print(f"\nTotal analysis runtime: {total_time:.2f} seconds")

    # ---------------------------
    # Save results to CSV
    # ---------------------------
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_facilities", "average_distance_m", "max_distance_m"])
        writer.writerows(results)
        writer.writerow(["total_time_seconds", total_time, ""])  # Add runtime row

    print(f"\nAnalysis CSV saved to {OUTPUT_CSV}")
    exit()




# ---------------------------
# NORMAL MODE: generate swap stations + Voronoi
# ---------------------------
print(f"Clustering into {NUM_LOCATIONS} facilities...")
kmeans = KMeans(n_clusters=NUM_LOCATIONS, random_state=42)
kmeans.fit(taxi_points)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Create Voronoi polygons
centroid_points = [Point(c) for c in centroids]
bbox = Polygon([
    [taxi_points[:,0].min(), taxi_points[:,1].min()],
    [taxi_points[:,0].max(), taxi_points[:,1].min()],
    [taxi_points[:,0].max(), taxi_points[:,1].max()],
    [taxi_points[:,0].min(), taxi_points[:,1].max()]
])
multi_centroids = gpd.GeoSeries(centroid_points).union_all()
vor_polys = voronoi_diagram(multi_centroids, envelope=bbox)

# Build GeoJSON features
features = []

# Voronoi polygons clipped to Accra
for i, poly in enumerate(vor_polys.geoms if hasattr(vor_polys, "geoms") else [vor_polys]):
    poly_clipped = poly.intersection(accra_polygon)
    features.append({
        "type": "Feature",
        "properties": {"facility_id": i},
        "geometry": mapping(poly_clipped)
    })

# Facility points
for i, c in enumerate(centroids):
    features.append({
        "type": "Feature",
        "properties": {"facility_id": i, "type": "facility"},
        "geometry": mapping(Point(c))
    })

# Lines linking taxi ranks to their nearest facility
for idx, pt in enumerate(taxi_points):
    facility_idx = labels[idx]
    line = LineString([pt, centroids[facility_idx]])
    features.append({
        "type": "Feature",
        "properties": {"facility_id": int(facility_idx), "type": "link_to_facility"},
        "geometry": mapping(line)
    })

# Save GeoJSON
fc = {"type": "FeatureCollection", "features": features}
with open(OUTPUT_GEOJSON, "w") as f:
    json.dump(fc, f)
print(f"Saved {NUM_LOCATIONS} facilities, Voronoi polygons, and links to {OUTPUT_GEOJSON}")
