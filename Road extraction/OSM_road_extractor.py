import argparse
import collections
import os
import geopandas as gpd
import osmnx as ox
from shapely.geometry import LineString, Point, mapping
import json

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import load_scenario, add_scenario_argument

_parser = argparse.ArgumentParser(
    description="Download the road network for a scenario's area")
add_scenario_argument(_parser)
_parser.add_argument("--yes", action="store_true",
                     help="proceed without asking, even if the OSM data contains "
                          "highway types that road_speed_km-h does not list")
_parser.add_argument("--no-filter", action="store_true",
                     help="keep every highway type, including those set to 0 km/h")
_args = _parser.parse_args()

SCENARIO = load_scenario(_args.scenario)
scenario_cfg = SCENARIO.cfg

FOLDER_NAME = SCENARIO.folder_name
INPUT_DIR = SCENARIO.input_dir

ROAD_SPEEDS = scenario_cfg["road_speed_km-h"]   # Default speed for roads (in km/h) if not specified in OSM data

def is_allowed_highway(highway, road_speeds):
    if isinstance(highway, list):
        return any(road_speeds.get(h, 0) > 0 for h in highway)
    return road_speeds.get(highway, 0) > 0 # Change to road_speeds.get(highway, 1) to include unclassified highway types


def highway_types(G):
    """Edge count per highway type, flattening the lists OSM sometimes uses."""
    counts = collections.Counter()
    for _u, _v, _k, data in G.edges(keys=True, data=True):
        hw = data.get("highway")
        for h in (hw if isinstance(hw, list) else [hw]):
            counts[h if isinstance(h, str) else "(untagged)"] += 1
    return counts


def review_highway_types(G, road_speeds, assume_yes):
    """
    Report what the filter is about to drop, and let the user stop.

    Two different reasons get separated, because they mean different things: a
    type set to 0 km/h was excluded on purpose, whereas a type missing from
    road_speed_km-h is excluded only because nothing said what speed it has --
    which is usually an oversight, and silently losing those roads is the
    failure this check exists to prevent.

    Returns True to go ahead. Never prompts when stdin is not a terminal: the
    control panel runs this as a subprocess with no stdin, and a prompt there
    would hang forever rather than ask anyone anything.
    """
    counts = highway_types(G)
    known = set(road_speeds)
    zero = {h: n for h, n in counts.items() if h in known and road_speeds[h] == 0}
    unknown = {h: n for h, n in counts.items() if h not in known}

    total = sum(counts.values())
    print(f"\nOSM data contains {len(counts)} highway types across {total:,} edges.")

    if zero:
        print(f"\n  Set to 0 km/h in {os.path.basename(SCENARIO.path)} "
              f"-- will be removed:")
        for h, n in sorted(zero.items(), key=lambda x: -x[1]):
            print(f"    {h:22s} {n:>9,} edges")

    if not unknown:
        print("\n  Every type in the data is listed in road_speed_km-h.")
        return True

    unknown_edges = sum(unknown.values())
    print(f"\n  NOT LISTED in road_speed_km-h -- will also be removed "
          f"({unknown_edges:,} edges, {100 * unknown_edges / total:.1f}%):")
    for h, n in sorted(unknown.items(), key=lambda x: -x[1]):
        print(f"    {h:22s} {n:>9,} edges")
    print("\n  Add these to road_speed_km-h with a speed to keep them.")

    if assume_yes:
        print("  Proceeding anyway (--yes).")
        return True

    def no_answer():
        print("\nStopping: nothing has been written, and the existing road files "
              "are untouched.\nRe-run in a terminal to be asked, or pass --yes to "
              "accept the removals.")
        return False

    if not sys.stdin or not sys.stdin.isatty():
        return no_answer()

    try:
        reply = input("\nProceed and remove them? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        # isatty() can still be true where nothing will ever arrive -- a Git Bash
        # pty, a CI runner, a harness. Treat silence as no.
        return no_answer()

    if reply not in ("y", "yes"):
        print("Cancelled. The existing road files are untouched.")
        return False
    return True

def get_roads(geojson_path="area.geojson", output_path=".",
              assume_yes=False, no_filter=False):
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
    # 2.1 Filter out disallowed roads (speed = 0, or not configured)
    # -------------------------------------------------------------
    # Reviewed here, after the download but before anything is written, so
    # cancelling costs only the download and leaves the existing road files
    # exactly as they were. There is no backup, so that ordering matters.
    if not review_highway_types(G, ROAD_SPEEDS, assume_yes):
        return None

    edges_before = G.number_of_edges()

    if no_filter:
        print("\nKeeping every highway type (--no-filter).")
    else:
        print("\nFiltering roads based on ROAD_SPEEDS...")
        edges_to_remove = []
        for u, v, key, data in G.edges(keys=True, data=True):
            highway = data.get("highway")

            if not is_allowed_highway(highway, ROAD_SPEEDS):
                edges_to_remove.append((u, v, key))

        G.remove_edges_from(edges_to_remove)
        print(f"Removed {len(edges_to_remove):,} of {edges_before:,} edges "
              f"({100 * len(edges_to_remove) / edges_before:.1f}%) "
              "with speed 0 or no configured speed")

    # Remove isolated nodes
    isolated_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]
    G.remove_nodes_from(isolated_nodes)
    print(f"Removed {len(isolated_nodes):,} isolated nodes")

    # -------------------------------------------------------------
    # 2.2 Keep only largest strongly connected component
    # -------------------------------------------------------------
    import networkx as nx

    print("Extracting largest strongly connected component...")

    nodes_before = G.number_of_nodes()
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_scc).copy()

    print(f"Retained {len(G.nodes):,} of {nodes_before:,} nodes in largest SCC")

    # Compared against the count before subgraphing. The old test compared
    # against G after the subgraph, where the two are equal by construction, so
    # it could never fire.
    if len(largest_scc) < 0.5 * nodes_before:
        print(f"Warning: the largest connected component is only "
              f"{100 * len(largest_scc) / nodes_before:.0f}% of the filtered "
              "network -- filtering may have cut it into pieces")

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

    result = get_roads(output_path=INPUT_DIR,
                       assume_yes=_args.yes, no_filter=_args.no_filter)
    # Cancelled at the review step: nothing was written, so say so with a
    # non-zero exit rather than looking like a successful run.
    if result is None:
        sys.exit(1)
