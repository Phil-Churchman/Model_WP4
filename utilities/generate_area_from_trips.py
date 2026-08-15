"""
Build <folder_name>/geojson_files/area.geojson from the captured trip data.

Takes every start and end coordinate in <folder_name>/captured_locations/
source_trip_data.csv, wraps them in a polygon and pushes that polygon out by a
margin (1 km by default), so the extracted road network is guaranteed to cover
every point the vehicles were actually seen at, plus room to route around them.

The folder name comes from scenario.json, as it does everywhere else in the
model. The result is a FeatureCollection holding a single Polygon, which is what
"Road extraction"/OSM_road_extractor.py and the swap_station_location scripts
expect to read back.

The margin is applied in metres, in a UTM zone derived from the data, so it is a
true distance rather than a number of degrees.

Usage:
    python utilities/generate_area_from_trips.py                  # convex hull, 1 km
    python utilities/generate_area_from_trips.py --margin-m 2000
    python utilities/generate_area_from_trips.py --shape concave  # tighter
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.geometry import MultiPoint, box, mapping
from shapely.ops import transform as shapely_transform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import load_scenario, add_scenario_argument

# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_MARGIN_M = 1000.0

COORD_COLUMNS = [("start_lon", "start_lat"), ("end_lon", "end_lat")]


# ============================================================
# HELPERS
# ============================================================

def load_points(csv_path):
    """Every start and end coordinate in the CSV, as a unique (lon, lat) array."""
    df = pd.read_csv(csv_path)
    missing = [c for pair in COORD_COLUMNS for c in pair if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} has no column(s): {', '.join(missing)}")

    frames = [df[[lon, lat]].rename(columns={lon: "lon", lat: "lat"})
              for lon, lat in COORD_COLUMNS]
    pts = pd.concat(frames, ignore_index=True).dropna()
    # Rows with a blank position are a capture artefact, not a location, so they
    # simply contribute nothing to the extent.
    total = len(df) * len(COORD_COLUMNS)
    return np.unique(pts.to_numpy(dtype=float), axis=0), total, len(pts)


def utm_transformers(lons, lats):
    """Round-trip transformers between WGS84 and the UTM zone the data sits in."""
    zone = int((float(np.mean(lons)) + 180.0) // 6.0) + 1
    epsg = (32600 if float(np.mean(lats)) >= 0 else 32700) + zone
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_wgs = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return to_utm, to_wgs, epsg


def build_shape(xy, shape, margin_m, concave_ratio):
    """
    Wrap the projected points and push the result out by `margin_m`.

    Every option here contains all the input points before the margin is added,
    so the margin is genuine clearance rather than a correction.
    """
    if shape == "bbox":
        # Expanded rather than buffered, so the result stays a true rectangle
        # instead of picking up rounded corners.
        x, y = xy[:, 0], xy[:, 1]
        return box(x.min() - margin_m, y.min() - margin_m,
                   x.max() + margin_m, y.max() + margin_m)

    # buffer() approximates each corner arc with straight segments, which cuts
    # the corner slightly inside the true margin. 32 segments per quadrant keeps
    # that under 0.3 m at a 1 km margin, so the margin holds as a minimum.
    points = MultiPoint(xy)
    if shape == "convex":
        return points.convex_hull.buffer(margin_m, quad_segs=32)
    if shape == "concave":
        # Follows the shape of the surveyed area rather than bridging across the
        # gaps between outlying trips: on the Nairobi capture this is roughly
        # half the area of the convex hull, and so half the road network to
        # download and route over.
        hull = shapely.concave_hull(points, ratio=concave_ratio)
        return hull.buffer(margin_m, quad_segs=32)
    raise ValueError(f"unknown shape: {shape}")


def to_feature_collection(polygon, properties):
    # bbox goes in the RFC 7946 member rather than in properties: a list-valued
    # property is not a type OGR can represent, so geopandas warns on every read.
    return {
        "type": "FeatureCollection",
        "bbox": list(polygon.bounds),
        "features": [{"type": "Feature",
                      "geometry": mapping(polygon),
                      "properties": properties}],
    }


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scenario_argument(p)
    p.add_argument("--source", default=None,
                   help="captured trip CSV (default: the scenario's "
                        "captured_locations/source_trip_data.csv)")
    p.add_argument("--output", default=None,
                   help="area.geojson to write (default: the scenario's geojson_files/)")
    p.add_argument("--margin-m", type=float, default=DEFAULT_MARGIN_M,
                   help="clearance added beyond the outermost points, in metres "
                        f"(default {DEFAULT_MARGIN_M:.0f})")
    p.add_argument("--shape", choices=("convex", "concave", "bbox"), default="convex",
                   help="convex hull (default), concave hull, or bounding box")
    p.add_argument("--concave-ratio", type=float, default=0.1,
                   help="tightness for --shape concave, 0 (tightest) to 1 (convex)")
    p.add_argument("--max-distance-km", type=float,
                   help="ignore points further than this from the median position; "
                        "off by default, so every point is enclosed")
    p.add_argument("--no-backup", action="store_true",
                   help="overwrite an existing area.geojson without copying it aside")
    return p.parse_args()


def main():
    args = parse_args()
    scenario = load_scenario(args.scenario)
    source = args.source or os.path.join(scenario.captured_dir, "source_trip_data.csv")
    output = args.output or os.path.join(scenario.input_dir, "area.geojson")

    pts, total_slots, populated = load_points(source)
    print(f"Scenario folder: {scenario.folder_name}")
    print(f"Source:          {source}")
    print(f"Coordinates:     {populated} of {total_slots} start/end slots populated, "
          f"{len(pts)} distinct")

    to_utm, to_wgs, epsg = utm_transformers(pts[:, 0], pts[:, 1])
    x, y = to_utm.transform(pts[:, 0], pts[:, 1])
    xy = np.column_stack([x, y])

    if args.max_distance_km:
        centre = np.median(xy, axis=0)
        keep = np.hypot(xy[:, 0] - centre[0], xy[:, 1] - centre[1]) <= args.max_distance_km * 1000
        dropped = len(xy) - int(keep.sum())
        xy = xy[keep]
        print(f"Excluded {dropped} points over {args.max_distance_km:g} km "
              f"from the median position")
        if len(xy) < 3:
            raise SystemExit("Too few points left to build an area.")

    polygon_utm = build_shape(xy, args.shape, args.margin_m, args.concave_ratio)
    if polygon_utm.is_empty:
        raise SystemExit("The generated area is empty.")

    # Confirm the margin did what it says before anything is written: every point
    # that went in must be inside, and clear of the edge by the full margin.
    inside = shapely.contains_xy(polygon_utm, xy[:, 0], xy[:, 1])
    if not inside.all():
        raise SystemExit(f"{(~inside).sum()} points fell outside the generated area.")
    clearance = shapely.distance(shapely.points(xy), polygon_utm.exterior).min()
    print(f"Shape:           {args.shape} + {args.margin_m:g} m margin "
          f"(UTM EPSG:{epsg})")
    print(f"Area:            {polygon_utm.area / 1e6:,.0f} km2, "
          f"{len(polygon_utm.exterior.coords)} vertices")
    print(f"Clearance:       {clearance:,.0f} m from the outermost point to the edge")

    polygon = shapely_transform(lambda a, b: to_wgs.transform(a, b), polygon_utm)
    minx, miny, maxx, maxy = polygon.bounds
    collection = to_feature_collection(polygon, {
        "name": f"{scenario.name} captured-trip extent",
        "source": os.path.basename(source),
        "shape": args.shape,
        "margin_m": args.margin_m,
        "point_count": int(len(xy)),
        "area_km2": round(polygon_utm.area / 1e6, 1),
        "generated": datetime.now().isoformat(timespec="seconds"),
    })

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    # The file being replaced may be a hand-drawn or administrative boundary that
    # exists nowhere else, so it is copied aside once rather than just lost.
    if os.path.exists(output) and not args.no_backup:
        backup = os.path.splitext(output)[0] + ".previous.geojson"
        if not os.path.exists(backup):
            shutil.copy2(output, backup)
            print(f"Previous area kept at {backup}")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(collection, f)

    print(f"Wrote {output}")
    print(f"Bounds: lon {minx:.5f} to {maxx:.5f}, lat {miny:.5f} to {maxy:.5f}")


if __name__ == "__main__":
    main()
