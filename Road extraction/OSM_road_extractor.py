import os
import geopandas as gpd
import osmnx as ox
from shapely.geometry import LineString, Point, mapping
import json

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import scenario_from_cli

SCENARIO = scenario_from_cli("Download the road network for a scenario's area")
scenario_cfg = SCENARIO.cfg

FOLDER_NAME = SCENARIO.folder_name
INPUT_DIR = SCENARIO.input_dir

ROAD_SPEEDS = scenario_cfg["road_speed_km-h"]   # Default speed for roads (in km/h) if not specified in OSM data

def is_allowed_highway(highway, road_speeds):
    if isinstance(highway, list):
        return any(road_speeds.get(h, 0) > 0 for h in highway)
    return road_speeds.get(highway, 0) > 0 # Change to road_speeds.get(highway, 1) to include unclassified highway types 

def get_roads(geojson_path="area.geojson", output_path="."):
    # -------------------------------------------------------------
    # 1. Load polygon
    # -------------------------------------------------------------
    gdf = gpd.read_file(os.path.join(INPUT_DIR, geojson_path))
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    polygon = gdf.union_all()

    # -------------------------------------------------------------
    # 2. Download road network from OSMnx
    # -------------------------------------------------------------
    print("Downloading road network from OSMnx...")
    G = ox.graph_from_polygon(
        polygon,
        network_type="all",   # or "all_private" if you want all roads
        # simplify=True
        simplify=False
    )

    # -------------------------------------------------------------
    # 2.1 Filter out disallowed roads (speed = 0)
    # -------------------------------------------------------------

    edges_to_remove = []

    # print("Filtering roads based on ROAD_SPEEDS...")
    # for u, v, key, data in G.edges(keys=True, data=True):
    #     highway = data.get("highway")

    #     if not is_allowed_highway(highway, ROAD_SPEEDS):
    #         edges_to_remove.append((u, v, key))

    # G.remove_edges_from(edges_to_remove)

    # Remove isolated nodes
    isolated_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]
    G.remove_nodes_from(isolated_nodes)

    # print(f"Removed {len(edges_to_remove)} edges with speed = 0")

    # -------------------------------------------------------------
    # 2.2 Keep only largest strongly connected component
    # -------------------------------------------------------------
    import networkx as nx

    print("Extracting largest strongly connected component...")

    largest_scc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_scc).copy()

    print(f"Retained {len(G.nodes)} nodes in largest SCC")

    if len(largest_scc) < 0.5 * len(G.nodes):
        print("Warning: SCC is much smaller than original graph")

    # -------------------------------------------------------------
    # 3. Save GraphML (OSMnx-compatible)
    # -------------------------------------------------------------
    graphml_file = os.path.join(output_path, "roads.graphml")
    ox.save_graphml(G, graphml_file)
    print(f"OSMnx GraphML saved to {graphml_file}")

    # -------------------------------------------------------------
    # 4. Create GeoJSON for visualization
    # -------------------------------------------------------------
    features = []
    for u, v, data in G.edges(data=True):
        # Coordinates
        if "geometry" in data:
            coords = list(data["geometry"].coords)
        else:
            # Straight line if no geometry
            coords = [(G.nodes[u]["x"], G.nodes[u]["y"]),
                      (G.nodes[v]["x"], G.nodes[v]["y"])]
        # Create LineString feature
        features.append({
            "type": "Feature",
            "geometry": mapping(LineString(coords)),
            "properties": {k: str(v) for k, v in data.items() if k != "geometry"}
        })

    geojson_file = os.path.join(output_path, "roads.geojson")
    with open(geojson_file, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print(f"GeoJSON saved to {geojson_file}")

    # -------------------------------------------------------------
    # 5. Done
    # -------------------------------------------------------------
    return G

if __name__ == "__main__":
    
    get_roads(output_path=INPUT_DIR)
