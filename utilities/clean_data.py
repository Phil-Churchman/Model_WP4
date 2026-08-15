import pandas as pd
import math, os, json
import numpy as np
import networkx as nx
import uuid

from shapely.geometry import LineString, Point
from shapely.wkt import loads as load_wkt
from shapely import STRtree

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from shapely.geometry import box
import xml.etree.ElementTree as ET

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import scenario_from_cli

SCENARIO = scenario_from_cli("Clean captured GPS data")
scenario_data = SCENARIO.cfg
FOLDER_NAME = SCENARIO.folder_name

OUTPUT_DIR = SCENARIO.captured_dir
INPUT_DIR = SCENARIO.captured_dir
GRAPHML_FILE = os.path.join(SCENARIO.input_dir, "roads.graphml")

# ============================================================
# HIGHWAY SPEED LIMIT DICTIONARY (KM/H)
# Used to dynamically flag unrealistic vehicle velocities
# ============================================================
HIGHWAY_SPEED_LIMITS = {
    "motorway": 130,
    "motorway_link": 90,
    "trunk": 110,
    "trunk_link": 80,
    "primary": 90,
    "primary_link": 70,
    "secondary": 80,
    "secondary_link": 60,
    "tertiary": 70,
    "tertiary_link": 50,
    "residential": 50,
    "living_street": 30,
    "service": 40,
    "unclassified": 60
}

DEFAULT_SPEED_LIMIT = 50  # Default speed limit for unknown highway types (km/h)

forbidden_types = [
    "steps",
    "platform",
    "corridor",
    "elevator",
    "escalator"
]

pavement_tags = [
    "footway",
    "pedestrian",
    "sidewalk",
    'cycleway',
    'bridleway',
    'path'
]


UNREALISTIC_SPEED_FACTOR = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# PARAMETERS
# ============================================================

loop_time_threshold_minutes = 20
cluster_distance_threshold_m = 20
centroid_merge_distance_m = 100
cluster_min_points_in_5min = 20
search_radius_m = 20.0
pavement_penalty = 100
persistence_bias = 10
all_roads_2_way = True
orientation_limit = 30.0

max_route_patch_length = 50
max_patch_mins = 5

G = None
G_drivable = None
spatial_tree = None
lat_key = None
lon_key = None
edge_geometries = None
edge_metadata = None
edge_lookup = None

edge_speeds_map = {}

# ============================================================
# LOAD DATA
# ============================================================

file_path = os.path.join(INPUT_DIR, "device_locations.xlsx")

df = pd.read_excel(file_path)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

if df["Timestamp"].dt.tz is not None:
    df["Timestamp"] = df["Timestamp"].dt.tz_localize(None)

df = df.sort_values("Timestamp").reset_index(drop=True)

# ============================================================
# OPTIONAL ACCURACY COLUMN
# ============================================================

has_accuracy = False
accuracy_col = None

for col in ["Accuracy", "accuracy"]:

    if col in df.columns:

        valid_values = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if len(valid_values) > 0:

            has_accuracy = True
            accuracy_col = col

            df[accuracy_col] = pd.to_numeric(
                df[accuracy_col],
                errors="coerce"
            )

            print(f"Using accuracy column: {accuracy_col}")

            break

# ============================================================
# GEO HELPERS (Fixed to use Meters directly)
# ============================================================

# 111,320 meters per degree of latitude
METERS_PER_DEG_LAT = 111320.0 

# If your data is highly localized, a fixed constant is fine:
avg_lat = df["Latitude"].mean()
METERS_PER_DEG_LON_FIXED = METERS_PER_DEG_LAT * math.cos(math.radians(avg_lat))

def distance_m(lat1, lon1, lat2, lon2):
    # Dynamic calculation if data spans large areas; 
    # otherwise replace with METERS_PER_DEG_LON_FIXED
    current_avg_lat = (lat1 + lat2) / 2.0
    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(current_avg_lat))
    
    dlat = (lat2 - lat1) * METERS_PER_DEG_LAT
    dlon = (lon2 - lon1) * meters_per_deg_lon
    
    return math.sqrt(dlat**2 + dlon**2)

def calculate_heading(lat1, lon1, lat2, lon2):

    d_lon = math.radians(lon2 - lon1)

    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)

    y = math.sin(d_lon) * math.cos(lat2_r)

    x = (
        math.cos(lat1_r) * math.sin(lat2_r)
        - math.sin(lat1_r)
        * math.cos(lat2_r)
        * math.cos(d_lon)
    )

    return (math.degrees(math.atan2(y, x)) + 360) % 360


def clean_attributes_for_graphml(data_dict, df_ref):

    cleaned = {
        "df_row_reference": str(df_ref)
    }

    for key, val in data_dict.items():
        cleaned[str(key)] = str(val)

    return cleaned


def add_edge_to_output(u, v, k, data_dict, df_ref, OutputGraph):
    str_u = str(u)
    str_v = str(v)

    # Ensure nodes exist
    for n_id, native_n in [(str_u, u), (str_v, v)]:
        if not OutputGraph.has_node(n_id):
            node_data = G.nodes[native_n]
            OutputGraph.add_node(
                n_id,
                lat=float(node_data[lat_key]),
                lon=float(node_data[lon_key])
            )

    attrs = clean_attributes_for_graphml(data_dict, df_ref)

    # FIX 2: Aggregate instead of generating random parallel keys
    osmid = data_dict.get("osmid", f"{u}_{v}_{k}")
    if isinstance(osmid, list):
        osmid = str(osmid[0])
    else:
        osmid = str(osmid)
        
    edge_key = f"{osmid}"
    
    # if OutputGraph.has_edge(str_u, str_v, key=edge_key):
    #     current_attrs = OutputGraph[str_u][str_v][edge_key]
        # current_count = int(current_attrs.get("traversal_count", 1))
        # OutputGraph[str_u][str_v][edge_key]["traversal_count"] = str(current_count + 1)
    # else:
    if not OutputGraph.has_edge(str_u, str_v, key=edge_key):
        # attrs["traversal_count"] = "1"
        OutputGraph.add_edge(
            str_u,
            str_v,
            key=edge_key,
            **attrs
        )

def find_route_patch(source, target, G, G_drivable):
    path = None

    # Tier 1: Drivable
    if (
        G_drivable.has_node(source)
        and G_drivable.has_node(target)
    ):

        try:

            path = nx.shortest_path(
                G_drivable,
                source=source,
                target=target,
                weight=road_travel_time_weight
            )

        except nx.NetworkXNoPath:
            pass

    # Tier 2: Full graph fallback
    if path == None:

        try:

            path = nx.shortest_path(
                G,
                source=source,
                target=target,
                weight=road_travel_time_weight
            )


        except (
            nx.NetworkXNoPath,
            nx.NodeNotFound
        ):
            pass
    
    
    return path

def road_travel_time_weight(u, v, edge_dict):
    """
    Returns minimum travel time (seconds) among parallel edges.
    """

    min_time_sec = float("inf")

    for key, data in edge_dict.items():

        length_m = float(data.get("length", 1.0))

        highway = data.get("highway", "")

        if isinstance(highway, list):
            highway_types = [str(h).lower() for h in highway]
        else:
            highway_types = [str(highway).lower()]

        speed_limit = DEFAULT_SPEED_LIMIT

        for h in highway_types:
            if h in HIGHWAY_SPEED_LIMITS:
                speed_limit = HIGHWAY_SPEED_LIMITS[h]
                break

        speed_mps = speed_limit / 3.6

        travel_time_sec = length_m / speed_mps

        if travel_time_sec < min_time_sec:
            min_time_sec = travel_time_sec

    return min_time_sec


# ============================================================
# STEP 1: CLUSTER DETECTION
# ============================================================

print("Step 1: Detecting high-density cluster centroids...")

lats = df["Latitude"].to_numpy()
lons = df["Longitude"].to_numpy()
times = df["Timestamp"].to_numpy()

five_min = np.timedelta64(5, 'm')

end_indices = np.searchsorted(
    times,
    times + five_min,
    side='right'
)

candidate_centroids = []

for i in range(len(df)):

    j = end_indices[i]

    count = j - i

    if count >= cluster_min_points_in_5min:

        sub_lats = lats[i:j]
        sub_lons = lons[i:j]

        med_lat = np.median(sub_lats)

        max_dlat = (
            np.max(np.abs(sub_lats - med_lat))
            * METERS_PER_DEG_LAT
        )

        if max_dlat > cluster_distance_threshold_m:
            continue

        med_lon = np.median(sub_lons)

        max_dlon = (
            np.max(np.abs(sub_lons - med_lon))
            * METERS_PER_DEG_LON_FIXED
        )

        if (
            math.sqrt(max_dlat**2 + max_dlon**2)
            <= cluster_distance_threshold_m
        ):
            candidate_centroids.append(
                Point(med_lon, med_lat)
            )

distinct_centroids = []

if candidate_centroids:

    raw_coords = np.array([
        [p.x, p.y]
        for p in candidate_centroids
    ])

    unique_coords = np.unique(
        np.round(raw_coords, 5),
        axis=0
    )

    filtered_candidates = [
        Point(lon, lat)
        for lon, lat in unique_coords
    ]

    print(
        f"Reduced candidates from "
        f"{len(candidate_centroids)} "
        f"to {len(filtered_candidates)} "
        f"unique regions."
    )

    merge_radius_deg = centroid_merge_distance_m / METERS_PER_DEG_LAT

    tree = STRtree(filtered_candidates)

    left_list = []
    right_list = []

    for idx, geom in enumerate(filtered_candidates):

        matches = tree.query(
            geom,
            predicate="dwithin",
            distance=merge_radius_deg
        )

        left_list.extend([idx] * len(matches))
        right_list.extend(matches)

    left_idx = np.array(left_list, dtype=np.int32)
    right_idx = np.array(right_list, dtype=np.int32)

    n_points = len(filtered_candidates)

    adjacency_matrix = csr_matrix(
        (
            np.ones(len(left_idx), dtype=bool),
            (left_idx, right_idx)
        ),
        shape=(n_points, n_points)
    )

    n_components, labels = connected_components(
        csgraph=adjacency_matrix,
        directed=False,
        return_labels=True
    )

    for comp_id in range(n_components):

        member_indices = np.where(labels == comp_id)[0]

        group_lats = [
            filtered_candidates[idx].y
            for idx in member_indices
        ]

        group_lons = [
            filtered_candidates[idx].x
            for idx in member_indices
        ]

        distinct_centroids.append((
            np.mean(group_lats),
            np.mean(group_lons)
        ))

print(
    f"Found {len(distinct_centroids)} "
    f"distinct consolidated cluster location(s)."
)

# ============================================================
# STEP 2 & 3: LOOP FLAGS
# ============================================================

print(
    "Steps 2 & 3: Identifying excursion loops "
    "and flagging tracking points..."
)

df["inside_cluster"] = False
df["closest_centroid_idx"] = -1

if distinct_centroids:

    c_arr = np.array(distinct_centroids)

    dlat_mat = lats[:, np.newaxis] - c_arr[:, 0]
    dlon_mat = lons[:, np.newaxis] - c_arr[:, 1]

    dist_matrix = np.sqrt(
        (dlat_mat * METERS_PER_DEG_LAT)**2
        + (dlon_mat * METERS_PER_DEG_LON_FIXED)**2
    )

    within_thresh = (
        dist_matrix <= cluster_distance_threshold_m
    )

    any_inside = np.any(within_thresh, axis=1)

    if np.any(any_inside):

        df.loc[any_inside, "inside_cluster"] = True

        df.loc[
            any_inside,
            "closest_centroid_idx"
        ] = np.argmax(
            within_thresh[any_inside],
            axis=1
        )

in_loop = np.zeros(len(df), dtype=bool)

df["assigned_centroid_idx"] = (
    df["closest_centroid_idx"]
)

inside_arr = df["inside_cluster"].to_numpy()
closest_arr = df["closest_centroid_idx"].to_numpy()

is_outside = ~inside_arr

runs = np.diff(
    np.pad(is_outside, (1, 1), mode='constant')
).nonzero()[0]

for idx in range(0, len(runs), 2):

    start_outside = runs[idx]
    end_outside = runs[idx + 1] - 1

    idx_before = start_outside - 1
    idx_after = end_outside + 1

    if idx_before >= 0 and idx_after < len(df):

        c_before = closest_arr[idx_before]
        c_after = closest_arr[idx_after]

        if c_before == c_after and c_before != -1:

            duration_mins = (
                (times[idx_after] - times[idx_before])
                .astype('timedelta64[s]')
                .astype(int) / 60.0
            )

            if duration_mins <= loop_time_threshold_minutes:

                loop_dists = dist_matrix[
                    start_outside:end_outside + 1,
                    c_before
                ]

                if np.any(
                    loop_dists > cluster_distance_threshold_m
                ):
                    in_loop[
                        start_outside:end_outside + 1
                    ] = True

                    df.loc[
                        start_outside:end_outside + 1,
                        "assigned_centroid_idx"
                    ] = c_before

df["flagged"] = (
    df["inside_cluster"] | in_loop
)

# ============================================================
# STEP 4: COLLAPSE FLAGGED SEQUENCES
# ============================================================

print(
    "Step 4: Collapsing continuous flagged sequences into Start/End pairs..."
)

flagged_arr = df["flagged"].to_numpy()
assigned_arr = df["assigned_centroid_idx"].to_numpy()

condition_change = (
    (flagged_arr[:-1] != flagged_arr[1:])
    |
    (assigned_arr[:-1] != assigned_arr[1:])
)

block_ids = np.concatenate((
    [0],
    np.cumsum(condition_change)
))

df["_block_id"] = block_ids

processed_blocks = []

for _, block_df in df.groupby("_block_id"):

    if not block_df["flagged"].iloc[0]:

        block_df = block_df.assign(
            GeneratedPoint=False,
            StationaryCluster=False
        )

        processed_blocks.append(block_df)

    else:

        c_idx = (
            block_df["assigned_centroid_idx"].iloc[0]
        )

        c_lat, c_lon = distinct_centroids[c_idx]

        start_pt = block_df.iloc[[0]].copy()

        start_pt[["Latitude", "Longitude"]] = [
            c_lat,
            c_lon
        ]

        start_pt = start_pt.assign(
            GeneratedPoint=True,
            StationaryCluster=True
        )

        processed_blocks.append(start_pt)

        if len(block_df) > 1:

            end_pt = block_df.iloc[[-1]].copy()

            end_pt[["Latitude", "Longitude"]] = [
                c_lat,
                c_lon
            ]

            end_pt = end_pt.assign(
                GeneratedPoint=True,
                StationaryCluster=True
            )

            processed_blocks.append(end_pt)

final_df = pd.concat(
    processed_blocks,
    ignore_index=True
)

final_df.drop(
    columns=[
        "_block_id",
        "inside_cluster",
        "closest_centroid_idx",
        "assigned_centroid_idx",
        "flagged"
    ],
    errors="ignore",
    inplace=True
)

def load_graph(graphml_file):

    global G, G_drivable, spatial_tree, lat_key, lon_key, edge_geometries, edge_metadata, edge_lookup
    
    # ============================================================
    # STEP 5: LOAD GRAPH
    # ============================================================

    if not os.path.exists(graphml_file):

        raise FileNotFoundError(
            f"The required network file was not found: "
            f"{GRAPHML_FILE}"
        )

    G = nx.read_graphml(graphml_file)
    G = nx.MultiDiGraph(G)

    first_node_id = list(G.nodes())[0]
    first_node_data = G.nodes[first_node_id]

    lon_key = next(
        (
            c for c in
            ["lon", "x", "longitude", "Lon", "X", "Longitude"]
            if c in first_node_data
        ),
        None
    )

    lat_key = next(
        (
            c for c in
            ["lat", "y", "latitude", "Lat", "Y", "Latitude"]
            if c in first_node_data
        ),
        None
    )

    if lon_key is None or lat_key is None:

        raise KeyError(
            f"Could not find coordinate attributes "
            f"on nodes. "
            f"Available attributes: "
            f"{list(first_node_data.keys())}"
        )

    # ============================================================
    # BUILD SPATIAL INDEX
    # ============================================================

    # print(
    #     "Building STRtree spatial index for fast geometry lookups..."
    # )

    edge_geometries = []
    edge_metadata = []
    edge_lookup = {}

    for edge_tuple in G.edges(data=True, keys=True):

        u, v, k, data = edge_tuple

        edge_id = (
            data.get("osmid")
            or data.get("id")
            or f"{u}_{v}_{k}"
        )

        edge_lookup[str(edge_id)] = {
            "length_m": float(data.get("length", 0.0)),
            "highway": data.get("highway", "unclassified")
        }

        if "geometry" in data:

            try:
                line = load_wkt(data["geometry"])

            except:

                u_data = G.nodes[u]
                v_data = G.nodes[v]

                line = LineString([
                    (
                        float(u_data[lon_key]),
                        float(u_data[lat_key])
                    ),
                    (
                        float(v_data[lon_key]),
                        float(v_data[lat_key])
                    )
                ])

        else:

            u_data = G.nodes[u]
            v_data = G.nodes[v]

            line = LineString([
                (
                    float(u_data[lon_key]),
                    float(u_data[lat_key])
                ),
                (
                    float(v_data[lon_key]),
                    float(v_data[lat_key])
                )
            ])

        edge_geometries.append(line)

        edge_metadata.append({
            "u": u,
            "v": v,
            "k": k,
            "edge_id": edge_id,
            "data": data
        })

    spatial_tree = STRtree(edge_geometries)

    print(
        f"Spatial index complete. "
        f"Indexed {len(edge_geometries)} edge segments."
    )

    # ============================================================
    # DRIVABLE SUBGRAPH GENERATION
    # ============================================================

    drivable_edges = [
        (u, v, k)
        for u, v, k, d
        in G.edges(keys=True, data=True)
        if not any(
            tag in str(d.get("highway", "")).lower()
            for tag in pavement_tags
        )
    ]

    G_drivable = G.edge_subgraph(drivable_edges)

def map_matching(final_df, run):

    global edge_speeds_map

    edge_speeds_map.clear()

    # ============================================================
    # MAP MATCHING (WITH TRUE NETWORK ROUTE DISTANCE ENGINE)
    # ============================================================

    if run == 1:
        load_graph(GRAPHML_FILE)
    else:
        load_graph(os.path.join(INPUT_DIR, "matched_routes.graphml"))

    print("Executing sequential mapping and edge tracking...")

    OutputGraph = nx.MultiDiGraph()

    degree_buffer = search_radius_m / METERS_PER_DEG_LAT

    prev_time = None
    prev_edge_meta = None

    all_trips = []
    current_trip = []

    # Ensure strict chronological order
    final_df = (
        final_df
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    OutputGraph = nx.MultiDiGraph()
    degree_buffer = search_radius_m / METERS_PER_DEG_LAT

    forbidden_types = [
        "steps",
        "platform",
        "corridor",
        "elevator",
        "escalator"
    ]

    prev_time = None
    prev_edge_meta = None

    all_trips = []
    current_trip = []

    # ------------------------------------------------------------
    # INITIALIZE ALL COLUMNS IN THE DATAFRAME
    # ------------------------------------------------------------
    final_df["Match"] = True
    final_df["matched_edge_id"] = None
    final_df["snapped_lat"] = np.nan
    final_df["snapped_lon"] = np.nan
    final_df["dist_from_prev_snapped_m"] = 0.0
    final_df["speed_snapped_mps"] = 0.0
    final_df["speed_snapped_kmh"] = 0.0
    final_df["route_time_sec"] = 0.0
    final_df["route_time_weighted_sec"] = 0.0
    final_df["prev_snapped_index"] = -1
    final_df["traversed_edge_ids_json"] = "[]"
    final_df["is_unrealistic_speed"] = False  # New speed validation flag

    # Persistent states to trace distance calculations along network edges
    last_snapped_lat = None
    last_snapped_lon = None
    last_snapped_idx = -1
    prev_edge_id = None
    prev_true_u = None
    prev_true_v = None
    prev_edge_len_m = 0.0
    prev_dist_from_true_u_m = 0.0

    # ============================================================
    # MAIN MATCH LOOP
    # ============================================================
    lats = final_df["Latitude"].to_numpy()
    lons = final_df["Longitude"].to_numpy()
    timestamps = final_df["Timestamp"].to_numpy()

    for idx in range(len(final_df)):
        p_lat = lats[idx]
        p_lon = lons[idx]
        curr_time = timestamps[idx]

        print(f"Processing point {idx + 1} / {len(final_df)}", end="\r")

        next_row = (
            final_df.iloc[idx + 1]
            if idx < len(final_df) - 1
            else None
        )

        next_time = (
            next_row["Timestamp"]
            if next_row is not None
            else None
        )

        time_gap_prev_min = (
            (curr_time - prev_time) / np.timedelta64(1, 'm')
            if prev_time is not None
            else float('inf')
        )

        time_gap_next_min = (
            (next_time - curr_time) / np.timedelta64(1, 'm')
            if next_time is not None
            else float('inf')
        )

        point_geom = Point(p_lon, p_lat)

        minx = p_lon - degree_buffer
        maxx = p_lon + degree_buffer
        miny = p_lat - degree_buffer
        maxy = p_lat + degree_buffer

        query_box = box(minx, miny, maxx, maxy)
        nearby_indices = spatial_tree.query(query_box)

        candidates = []

        # ========================================================
        # BUILD CANDIDATES
        # ========================================================
        for edge_idx in nearby_indices:
            line = edge_geometries[edge_idx]
            meta = edge_metadata[edge_idx]
            data = meta["data"]

            snap = line.interpolate(line.project(point_geom))
            dist = distance_m(p_lat, p_lon, snap.y, snap.x)

            if dist > search_radius_m:
                continue

            highway_str = str(data.get("highway", "")).lower()
            name_str = str(data.get("name", "")).lower()
            
            if next_row is not None:
                dist_to_next = distance_m(p_lat, p_lon, next_row["Latitude"], next_row["Longitude"])
                if dist_to_next > 1.0:
                    route_heading = calculate_heading(p_lat, p_lon, next_row["Latitude"], next_row["Longitude"])
                else:
                    route_heading = prev_edge_meta["heading"] if prev_edge_meta is not None else None
            else:
                route_heading = prev_edge_meta["heading"] if prev_edge_meta is not None else None

            coords = list(line.coords)
            if len(coords) >= 2:
                geom_heading = calculate_heading(coords[0][1], coords[0][0], coords[-1][1], coords[-1][0])
            else:
                geom_heading = route_heading

            if all_roads_2_way:
                is_oneway = False
            else:
                is_oneway = str(data.get("oneway", "")).lower() in ["true", "1", "yes"]
            
            if route_heading is not None:
                diff_forward = abs(route_heading - geom_heading)
                if diff_forward > 180: 
                    diff_forward = 360 - diff_forward
            else:
                diff_forward = 0.0
            
            directions_to_try = [("forward", diff_forward, meta["u"], meta["v"])]
            
            if not is_oneway:
                if route_heading is not None:
                    diff_backward = abs(route_heading - ((geom_heading + 180) % 360))
                    if diff_backward > 180: 
                        diff_backward = 360 - diff_backward
                else:
                    diff_backward = 0.0
                    
                directions_to_try.append(("backward", diff_backward, meta["v"], meta["u"]))

            pavement_pen = pavement_penalty if any(t in highway_str for t in pavement_tags) else 0.0

            if any(ft in highway_str for ft in forbidden_types):
                continue

            proj_deg = line.project(point_geom)
            line_len_deg = line.length
            fraction = proj_deg / line_len_deg if line_len_deg > 0 else 0.0
            edge_len_m = float(data.get("length", 0.0))
            dist_from_start_m = fraction * edge_len_m

            for direction, heading_diff, true_u, true_v in directions_to_try:
                score = dist + (heading_diff * 0.5) + pavement_pen
                
                if prev_edge_meta is not None:
                    if prev_edge_meta["data"].get("osmid") == data.get("osmid"):
                        score -= persistence_bias
                    if data.get("name") == prev_edge_meta["data"].get("name") and data.get("name") is not None:
                        score -= 10.0

                if direction == "forward":
                    dist_from_true_u_m = dist_from_start_m
                else:
                    dist_from_true_u_m = edge_len_m - dist_from_start_m

                edge_id_str = data.get("osmid", f"{true_u}_{true_v}_{meta['k']}")

                candidates.append({
                    "meta": meta,
                    "u": true_u,
                    "v": true_v,
                    "dist": dist,
                    "snap_x": snap.x,
                    "snap_y": snap.y,
                    "edge_id": edge_id_str,
                    "edge_length_m": edge_len_m,
                    "dist_from_true_u_m": dist_from_true_u_m,
                    "heading": geom_heading if direction == "forward" else (geom_heading + 180) % 360,
                    "heading_diff": heading_diff,
                    "name": data.get("name"),
                    "score": score
                })

        if not candidates:
            final_df.at[idx, "Match"] = False
            continue

        selected_candidate = None

        # ========================================================
        # SAME ROAD CONTINUATION STRATEGY
        # ========================================================
        if (
            time_gap_prev_min < 1.0
            and prev_edge_meta is not None
        ):
            prev_name = prev_edge_meta["data"].get("name")
            if prev_name is not None:
                name_matches = [
                    c for c in candidates 
                    if c["name"] == prev_name and c["heading_diff"] <= orientation_limit
                ]
                if name_matches:
                    selected_candidate = min(name_matches, key=lambda x: x["score"])

        if selected_candidate is None:
            selected_candidate = min(candidates, key=lambda x: x["score"])

        curr_meta = selected_candidate["meta"]

        curr_snap_lat = selected_candidate["snap_y"]
        curr_snap_lon = selected_candidate["snap_x"]
        curr_edge_id = selected_candidate["edge_id"]
        curr_true_u = selected_candidate["u"]
        curr_true_v = selected_candidate["v"]
        curr_edge_len_m = selected_candidate["edge_length_m"]
        curr_dist_from_true_u_m = selected_candidate["dist_from_true_u_m"]

        final_df.at[idx, "matched_edge_id"] = curr_edge_id
        final_df.at[idx, "snapped_lat"] = curr_snap_lat
        final_df.at[idx, "snapped_lon"] = curr_snap_lon

        patch_distance_m = 0.0
        patch_edge_ids = []
        patch_highways = []
        patch_distance_m = 0.0
        route_link_valid = False
        same_edge_movement = False
        
        if last_snapped_idx != -1:
            if prev_edge_id == curr_edge_id and prev_true_u == curr_true_u:
                same_edge_movement = True
                route_link_valid = True

        # ========================================================
        # ROUTE PATCHING & INTERMEDIATE IDENTIFIER EXTRACTION
        # ========================================================
        new_edge = {
            "u": curr_true_u,
            "v": curr_true_v,
            "k": selected_candidate["meta"]["k"],
            "data": selected_candidate["meta"]["data"],
            "ref": idx
        }

        if current_trip == []:
            current_trip = [new_edge]
        elif current_trip[-1]["u"] == new_edge["u"] and current_trip[-1]["v"] == new_edge["v"]:
            pass
        elif time_gap_prev_min > max_patch_mins or current_trip[-1]["v"] == new_edge["u"]:
            current_trip.append(new_edge)
            if time_gap_prev_min <= max_patch_mins:
                route_link_valid = True
        else:
            source_node = current_trip[-1]["v"]
            target_node = new_edge["u"]

            path = find_route_patch(source_node, target_node, G, G_drivable)

            if path and len(path) <= max_route_patch_length:
            # if path:    
                for p_idx in range(len(path) - 1):
                    pu = path[p_idx]
                    pv = path[p_idx + 1]

                    if G.has_edge(pu, pv):
                        pk = min(
                            G[pu][pv].keys(),
                            key=lambda k:
                                road_travel_time_weight(
                                    pu,
                                    pv,
                                    {k: G[pu][pv][k]}
                                )
                        )

                        pdata = G[pu][pv][pk]

                        patch_distance_m += float(
                            pdata.get("length", 0.0)
                        )
                        
                        p_id = pdata.get("osmid", f"{pu}_{pv}_{pk}")
                        patch_edge_ids.append(p_id)
                        patch_highways.append(pdata.get("highway", "unclassified"))

                        current_trip.append({
                            "u": pu, "v": pv, "k": pk, "data": pdata, "ref": f"{idx}_route_patch"
                        })
                
                current_trip.append(new_edge)
                route_link_valid = True
            else:
                backup_applied = False
                for count in range(8):
                    if count + 2 > len(current_trip):
                        break
                    
                    source = current_trip[-(count + 2)]["v"]
                    backup_path = find_route_patch(source, new_edge["u"], G, G_drivable)

                    if backup_path and len(backup_path) <= max_route_patch_length:
                    # if backup_path:    
                        for _ in range(count + 1):
                            current_trip.pop()
                        
                        patch_edge_ids = []
                        patch_highways = []
                        for p_idx in range(len(backup_path) - 1):
                            pu = backup_path[p_idx]
                            pv = backup_path[p_idx + 1]

                            if G.has_edge(pu, pv):
                                pk = min(
                                    G[pu][pv].keys(),
                                    key=lambda k:
                                        road_travel_time_weight(
                                            pu,
                                            pv,
                                            {k: G[pu][pv][k]}
                                        )
                                )
                                pdata = G[pu][pv][pk]
                                patch_distance_m += float(
                                    pdata.get("length", 0.0)
                                )                                
                                p_id = pdata.get("osmid", f"{pu}_{pv}_{pk}")
                                patch_edge_ids.append(p_id)
                                patch_highways.append(pdata.get("highway", "unclassified"))
                                
                                current_trip.append({
                                    "u": pu, "v": pv, "k": pk, "data": pdata, "ref": f"{idx}_backup_patch"
                                })

                        current_trip.append(new_edge)
                        backup_applied = True
                        route_link_valid = True
                        break
                
                if not backup_applied:
                    current_trip.append(new_edge)

        # ========================================================
        # CALCULATE NETWORK DISTANCE, SPEED & HIGHLIGHT ANOMALIES
        # ========================================================

        route_distance = 0.0
        weighted_time_sec = 0.0
        traversed_edges = []
        step_highways = []
        
        if last_snapped_idx != -1:
            if same_edge_movement:
                route_distance = abs(curr_dist_from_true_u_m - prev_dist_from_true_u_m)
                traversed_edges = [prev_edge_id]
                step_highways = [curr_meta["data"].get("highway", "unclassified")]
            elif route_link_valid:
                middle_distance_m = 0.0

                for edge_id in patch_edge_ids:
                    edge_info = edge_lookup.get(str(edge_id))

                    if edge_info:
                        middle_distance_m += edge_info["length_m"]
                
                route_distance = (
                    (prev_edge_len_m - prev_dist_from_true_u_m)
                    + middle_distance_m
                    + curr_dist_from_true_u_m
                )

                # 1. Deduplicate while preserving chronological sequence order using dict.fromkeys
                traversed_edges = (
                    [str(prev_edge_id)]
                    + [str(e) for e in patch_edge_ids]
                    + [str(curr_edge_id)]
                )
                
                step_highways = [prev_edge_meta["data"].get("highway", "unclassified")] + patch_highways + [curr_meta["data"].get("highway", "unclassified")]
            else:
                route_distance = distance_m(last_snapped_lat, last_snapped_lon, curr_snap_lat, curr_snap_lon)
                traversed_edges = []
                step_highways = [curr_meta["data"].get("highway", "unclassified")]

            time_diff_sec = (curr_time - timestamps[last_snapped_idx]) / np.timedelta64(1, 's')
            if time_diff_sec > 0 and (route_link_valid or same_edge_movement):
                mps = route_distance / time_diff_sec
                kmh = mps * 3.6
            else:
                mps = 0.0
                kmh = 0.0
                traversed_edges = []

            # 2. Extract and flatten all highway types traversed in this single frame
            flat_highways = []
            for hw in step_highways:
                if isinstance(hw, list):
                    flat_highways.extend([str(h).lower() for h in hw])
                else:
                    flat_highways.append(str(hw).lower())

            # 3. Determine the maximum speed limit assigned to this group of roads
            max_allowed_speed_kmh = 0
            for hwt in flat_highways:
                limit = HIGHWAY_SPEED_LIMITS.get(hwt, 90)  # Standard fallback default
                if limit > max_allowed_speed_kmh:
                    max_allowed_speed_kmh = limit

            # 4. Set validation ceiling
            unrealistic_threshold_kmh = max_allowed_speed_kmh * UNREALISTIC_SPEED_FACTOR
            is_unrealistic = (kmh > unrealistic_threshold_kmh) if route_link_valid or same_edge_movement else False
            if is_unrealistic:
                route_distance = 0.0
                mps = 0.0
                kmh = 0.0
                # traversed_edges = []
                route_link_valid = False

            if same_edge_movement:
                if curr_edge_id in edge_speeds_map:
                    edge_speeds_map[curr_edge_id].append([route_distance, kmh])
                else:
                    edge_speeds_map[curr_edge_id] = [[route_distance, kmh]]
            elif route_link_valid:
                for edge_id in patch_edge_ids:
                    edge_info = edge_lookup.get(str(edge_id))
                    if edge_id in edge_speeds_map:
                        edge_speeds_map[edge_id].append([edge_info["length_m"], kmh])
                    else:
                        edge_speeds_map[edge_id] = [[edge_info["length_m"], kmh]]

                if prev_edge_id is not None:
                    if prev_edge_id in edge_speeds_map:
                        edge_speeds_map[prev_edge_id].append([prev_edge_len_m - prev_dist_from_true_u_m, kmh])
                    else:
                        edge_speeds_map[prev_edge_id] = [[prev_edge_len_m - prev_dist_from_true_u_m, kmh]]
                if curr_edge_id is not None:
                    if curr_edge_id in edge_speeds_map:
                        edge_speeds_map[curr_edge_id].append([curr_dist_from_true_u_m, kmh])
                    else:
                        edge_speeds_map[curr_edge_id] = [[curr_dist_from_true_u_m, kmh]]



            # Commit metric calculations back to the master dataframe
            final_df.at[idx, "dist_from_prev_snapped_m"] = route_distance
            final_df.at[idx, "speed_snapped_mps"] = mps
            final_df.at[idx, "speed_snapped_kmh"] = kmh
            final_df.at[idx, "prev_snapped_index"] = last_snapped_idx
            final_df.at[idx, "traversed_edge_ids_json"] = json.dumps(traversed_edges, default=str)
            final_df.at[idx, "is_unrealistic_speed"] = bool(is_unrealistic)
            final_df.at[idx, "route_link_valid"] = bool(route_link_valid)
            final_df.at[idx, "highway"] = highway_str
            final_df.at[idx, "name"] = name_str
            final_df.at[idx, "route_time_sec"] = time_diff_sec

            weighted_time_sec = 0.0

            for edge_id in traversed_edges:

                edge_info = edge_lookup.get(str(edge_id))

                if edge_info is None:
                    continue

                highway = edge_info["highway"]

                if isinstance(highway, list):
                    highway = highway[0]

                speed_limit = HIGHWAY_SPEED_LIMITS.get(
                    str(highway).lower(),
                    DEFAULT_SPEED_LIMIT
                )

                weighted_time_sec += (
                    edge_info["length_m"]
                    / (speed_limit / 3.6)
                )
        final_df.at[idx, "route_time_weighted_sec"] = weighted_time_sec

        # Step state updates forward for the following iteration
        last_snapped_lat, last_snapped_lon, last_snapped_idx = curr_snap_lat, curr_snap_lon, idx
        prev_edge_id, prev_true_u, prev_true_v = curr_edge_id, curr_true_u, curr_true_v
        prev_edge_len_m, prev_dist_from_true_u_m = curr_edge_len_m, curr_dist_from_true_u_m
        prev_edge_meta = {"data": curr_meta["data"], "heading": selected_candidate["heading"]}
        prev_time = curr_time

    #REMOVE ANOMOLOUS POINTS BASED ON TWO CONSECUTIVE UNREALISTIC SPEEDS

    # 1. Identify the rows where the condition is True
    true_mask = final_df["is_unrealistic_speed"] == True

    # 2. Identify the rows immediately *following* the True rows
    # Shifting the mask down fills the next row with True
    following_mask = true_mask.shift(1, fill_value=False)

    # 3. Combine both masks using the bitwise OR operator (|)
    # This flags both the unrealistic speed row AND the row after it
    rows_to_drop = true_mask & following_mask

    # 4. Filter the dataframe to keep only the rows where the combined mask is False
    final_df = final_df[~rows_to_drop].reset_index(drop=True)



    all_trips.append(current_trip)


    # ============================================================
    # STEP 6: CLEANING
    # ============================================================

    print(
        "Cleaning route sequences "
        "and removing jitters..."
    )

    OutputGraph = nx.MultiDiGraph()

    for trip_idx, trip in enumerate(all_trips):

        cleaned_trip = []

        for edge in trip:

            # ====================================================
            # REMOVE CONSECUTIVE DUPLICATES
            # ====================================================

            if (
                cleaned_trip
                and cleaned_trip[-1]["u"] == edge["u"]
                and cleaned_trip[-1]["v"] == edge["v"]
            ):
                continue

            # ====================================================
            # REMOVE IMMEDIATE REVERSALS
            # ====================================================

            if (
                len(cleaned_trip) >= 1
                and cleaned_trip[-1]["u"] == edge["v"]
                and cleaned_trip[-1]["v"] == edge["u"]
            ):

                cleaned_trip.pop()
                continue

            # ====================================================
            # REMOVE TINY ISLAND OSCILLATION LOOPS
            # ====================================================

            if len(cleaned_trip) >= 3:

                e1 = cleaned_trip[-3]
                e2 = cleaned_trip[-2]
                e3 = cleaned_trip[-1]

                if (
                    e1["u"] == e3["v"]
                    and e1["v"] == e3["u"]
                ):

                    cleaned_trip.pop()
                    cleaned_trip.pop()

                    continue

            cleaned_trip.append(edge)

        # ========================================================
        # WRITE CLEANED TRIP
        # ========================================================

        for edge in cleaned_trip:

            add_edge_to_output(
                edge["u"],
                edge["v"],
                edge["k"],
                edge["data"],
                edge["ref"],
                OutputGraph
            )

    # ============================================================
    # SAVE OUTPUT
    # ============================================================

    graphml_out_path = os.path.join(
        OUTPUT_DIR,
        f"matched_routes.graphml"
    )

    nx.write_graphml(
        OutputGraph,
        graphml_out_path
    )

    print(
        f"Cleaned mapping complete. "
        f"Saved to: {graphml_out_path}"
    )

    print(f"Final Graph Validation:")
    print(f"Nodes mapped: {OutputGraph.number_of_nodes()}")
    print(f"Edges mapped: {OutputGraph.number_of_edges()}")

    print(
        f"Saved generated tracking network to: "
        f"{graphml_out_path}"
    )

    # ============================================================
    # EXPORT WITH EXPLICIT INDEX FIELD
    # ============================================================

    # Convert structural index to an explicit, scannable column name
    final_df["point_index"] = final_df.index

    # Shift the point index column to position 0
    ordered_columns = ["point_index"] + [col for col in final_df.columns if col != "point_index"]
    final_df = final_df[ordered_columns]

    # Write out clean spreadsheet to disk
    excel_out_path = os.path.join(OUTPUT_DIR, f"gps_noise_reduced.xlsx")
    final_df.to_excel(excel_out_path, index=False)
    print(f"Saved modified location dataframe to: {excel_out_path}")

    return final_df

print("\nRUN 1")

final_df = map_matching(final_df, 1)

for run in range(2, 4):
    print(f"\nRUN {run}")
    is_matched = final_df["Match"] == True
    is_realistic = final_df["is_unrealistic_speed"] == False
    # valid_mask = is_matched & is_realistic
    valid_mask = is_matched
    final_df = final_df[valid_mask].copy().reset_index(drop=True)
    final_df = map_matching(final_df, run)

# ============================================================
# POST-PROCESSING: EMBED MAX & AVERAGE SPEEDS IN GRAPHML
# ============================================================
print("\nRunning post-processing to calculate and embed edge speed statistics...")

graphml_out_path = os.path.join(OUTPUT_DIR, "matched_routes.graphml")

if os.path.exists(graphml_out_path):
    print(f"Loading {graphml_out_path} for speed attribute embedding...")
    PostGraph = nx.read_graphml(graphml_out_path)
    PostGraph = nx.MultiDiGraph(PostGraph)
    
    updated_count = 0
    
    osmid_sequence = []
    osmid_traversals = {}
    for index, row in final_df.iterrows():
        for i in json.loads(row["traversed_edge_ids_json"]):
            osmid_sequence.append(i)

    osmid_last = None
    for i in osmid_sequence:
        if i != osmid_last:
            if i not in list(osmid_traversals.keys()):
                osmid_traversals[i] = 1
            else:
                osmid_traversals[i] += 1
        osmid_last = i


    # Update edge attributes based on mapping
    for u, v, k, data in list(PostGraph.edges(data=True, keys=True)):
        edge_id = data.get("osmid")
        
       
        # Cleanly convert to string if the key exists, otherwise keep it as None
        edge_key = str(edge_id) if edge_id is not None else None

        if edge_key in list(osmid_traversals.keys()):
            data["traversal_count"] = osmid_traversals[edge_key]
        else:
            data["traversal_count"] = 0
        # Check if we have tracking points and safe non-zero entries
        if edge_key and edge_key in edge_speeds_map.keys():

            valid_entries = [entry for entry in edge_speeds_map[edge_key] if entry[1] > 0]
            
            if valid_entries:
                speeds = [entry[1] for entry in valid_entries]
                times = [entry[0]/1000 / entry[1] for entry in valid_entries]
                distances = [entry[0]/1000 for entry in valid_entries]
                
              
                # # Compute maximum speed
                # data["max_speed_kmh"] = str(round(max(speeds), 0))
                
                # Compute accurate space-mean average speed (Total Distance / Total Time)
                total_time = sum(times)
                average_speed = sum(distances) / total_time if total_time > 0 else 0.0
                data["average_speed_kmh"] = str(round(average_speed, 0))
                
                updated_count += 1
            else:
                # Fallback if entries exist but all recorded speeds were 0 km/h
                # data["max_speed_kmh"] = "0.0"
                data["average_speed_kmh"] = "0.0"
        else:
            # Fallback values if an edge exists in the base network but had no track passes
            # data["max_speed_kmh"] = "0.0"
            data["average_speed_kmh"] = "0.0"
        


        # Explicitly write the modified dictionary properties back into the edge object
        speed_check = float(data.get("average_speed_kmh", "0.0")) < HIGHWAY_SPEED_LIMITS.get(str(data.get("highway", "unclassified")).lower(), DEFAULT_SPEED_LIMIT) * 1.5
        # if data["traversal_count"] != 0 and speed_check:
        if data["traversal_count"] != 0:    
            PostGraph[u][v][k].update(data)
        else:
            PostGraph.remove_edge(u, v, k)
        # PostGraph[u][v][k].update(data)
            
    # Save the updated graph with the newly embedded XML attributes
    nx.write_graphml(PostGraph, graphml_out_path)
    print(f"Successfully embedded speed attributes into {updated_count} edges in: {graphml_out_path}")
else:
    print(f"Error: GraphML file not found at {graphml_out_path}. Post-processing skipped.")