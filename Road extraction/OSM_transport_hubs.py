"""
Extract transport hubs from OpenStreetMap for a scenario's area.

Downloads everything OSM tags as public-transport infrastructure inside
<folder_name>/geojson_files/area.geojson and writes it to
<folder_name>/geojson_files/transport_hubs.geojson.

    python "Road extraction/OSM_transport_hubs.py" --scenario scenario.json

What counts as a hub
--------------------
OSM has no single "transport hub" tag, so several schemes are queried and the
results classified afterwards. Two properties are added to every feature:

    hub_type   bus | rail | taxi | air | ferry | other
    hub_scale  interchange -- a station, terminal or depot people change at
               stop        -- a single roadside stop or platform

The distinction matters for modelling. A bus station and a bus stop are both
"public transport" to OSM but only one is somewhere a vehicle would wait for a
fare. Both are extracted; hub_scale lets you use one, the other, or both, rather
than having that decision baked into the query.

Geometry
--------
OSM returns stops as points and stations as building footprints. Both are kept
as they are, and every feature also gets centroid_lat / centroid_lon so a
consumer that needs a single position -- as the simulation does when it snaps
facilities to road nodes -- has one without doing the geometry itself.
--as-points replaces the geometry with that centroid throughout.

Attributes
----------
OSM features carry whatever tags a mapper happened to add, so a full extract
runs to well over a hundred columns of which most are empty for most features.
By default only the core set is written -- identity, classification, name,
operator and position -- which is what almost every consumer wants. --full
keeps every tag that at least one feature populated, discarding nothing.
"""

import argparse
import json
import os
import random
import shutil
import sys
import time

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import load_scenario, add_scenario_argument

DEFAULT_RETRIES = 10
DEFAULT_TIMEOUT_S = 300

# Worth another attempt: the public Overpass instance drops long queries, rate
# limits, and returns 429/504 under load. All of those succeed on a retry.
# InsufficientResponseError is NOT here -- a well-formed query that found
# nothing will find nothing again, so retrying only wastes time.
RETRYABLE = (requests.exceptions.Timeout,
             requests.exceptions.ConnectionError,
             requests.exceptions.ChunkedEncodingError,
             ResponseStatusCodeError)

# The tag schemes OSM uses for transport infrastructure. Several overlap -- a
# bus station is often tagged amenity=bus_station AND public_transport=station --
# so features are de-duplicated by OSM id afterwards.
HUB_TAGS = {
    "amenity": ["bus_station", "taxi", "ferry_terminal", "car_sharing",
                "bicycle_rental", "motorcycle_taxi"],
    "railway": ["station", "halt", "tram_stop", "subway_entrance"],
    "public_transport": ["station", "stop_position", "platform", "stop_area"],
    "highway": ["bus_stop"],
    "aeroway": ["aerodrome", "terminal"],
    "building": ["train_station", "transportation"],
}

# (column, value) -> (hub_type, hub_scale). First match wins, so the more
# specific schemes are listed before the generic public_transport ones.
CLASSIFY = [
    ("aeroway", "aerodrome", "air", "interchange"),
    ("aeroway", "terminal", "air", "interchange"),
    ("amenity", "ferry_terminal", "ferry", "interchange"),
    ("amenity", "bus_station", "bus", "interchange"),
    ("amenity", "taxi", "taxi", "interchange"),
    ("amenity", "car_sharing", "other", "stop"),
    ("amenity", "bicycle_rental", "other", "stop"),
    ("amenity", "motorcycle_taxi", "taxi", "interchange"),
    ("railway", "station", "rail", "interchange"),
    ("railway", "halt", "rail", "stop"),
    ("railway", "tram_stop", "rail", "stop"),
    ("railway", "subway_entrance", "rail", "stop"),
    ("building", "train_station", "rail", "interchange"),
    ("building", "transportation", "other", "interchange"),
    ("highway", "bus_stop", "bus", "stop"),
    ("public_transport", "station", "bus", "interchange"),
    ("public_transport", "stop_area", "bus", "interchange"),
    ("public_transport", "platform", "bus", "stop"),
    ("public_transport", "stop_position", "bus", "stop"),
]

CORE_COLUMNS = ["osmid", "element", "name", "hub_type", "hub_scale",
                "amenity", "railway", "public_transport", "highway", "aeroway",
                "building", "operator", "network", "centroid_lat",
                "centroid_lon", "geometry"]


def classify(row):
    for column, value, hub_type, hub_scale in CLASSIFY:
        if column in row and str(row.get(column)) == value:
            return hub_type, hub_scale
    return "other", "stop"


def scalarise(value):
    """
    GeoJSON properties must be scalars. OSM occasionally yields a list (a way
    belonging to several relations, say), which json.dump would emit as a nested
    array -- valid JSON but awkward for every consumer downstream.
    """
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value)
    return value


def fetch(polygon, tags, retries=DEFAULT_RETRIES, timeout_s=DEFAULT_TIMEOUT_S):
    """
    Query Overpass, retrying on the failures that are worth retrying.

    The public Overpass instance is shared and rate limited, and a query over
    several thousand square kilometres is exactly the kind it drops. Backoff is
    exponential with jitter: exponential so a struggling server is not hammered,
    jittered so several clients that failed together do not all return together.

    A query that succeeds but finds nothing is NOT retried -- it would find
    nothing again. Nor is a malformed one; only transport failures.
    """
    ox.settings.requests_timeout = timeout_s
    print(f"Querying OpenStreetMap (timeout {timeout_s}s, up to {retries} attempts) ...")

    for attempt in range(1, retries + 1):
        try:
            gdf = ox.features_from_polygon(polygon, tags)
            if attempt > 1:
                print(f"  succeeded on attempt {attempt}")
            break
        except InsufficientResponseError:
            raise SystemExit(
                "OSM returned no transport features for this area. The query "
                "worked; there is simply nothing tagged here.")
        except RETRYABLE as e:
            if attempt == retries:
                raise SystemExit(
                    f"Overpass failed {retries} times; giving up.\n"
                    f"Last error: {type(e).__name__}: {e}\n"
                    "The service may be busy -- try again later, use a smaller "
                    "--area, or raise --timeout.")
            # 8, 16, 32 ... capped, plus up to 25% jitter.
            delay = min(8 * 2 ** (attempt - 1), 120)
            delay *= 1 + random.random() * 0.25
            print(f"  attempt {attempt}/{retries} failed "
                  f"({type(e).__name__}); retrying in {delay:.0f}s")
            time.sleep(delay)

    if gdf.empty:
        raise SystemExit("OSM returned no transport features for this area.")

    # The index carries the element type and id; both are worth keeping, and
    # both have to be normalised because osmnx has renamed them between
    # versions -- 1.x indexed on (element_type, osmid), 2.x on (element, id).
    # Assuming either name silently loses the identifier, and anything guarded
    # on its presence then never runs.
    gdf = gdf.reset_index()
    gdf = gdf.rename(columns={"element_type": "element", "id": "osmid"})
    if "element" not in gdf.columns:
        gdf["element"] = gdf.geometry.geom_type
    if "osmid" not in gdf.columns:
        raise SystemExit(
            "OSM features came back with no identifier column. osmnx "
            f"{ox.__version__} returned: {list(gdf.columns)[:8]} ...\n"
            "The index naming has changed again; update the rename above.")
    return gdf


def main():
    args = parse_args()
    scenario = load_scenario(args.scenario)

    area_path = args.area or os.path.join(scenario.input_dir, "area.geojson")
    out_path = args.output or os.path.join(scenario.input_dir,
                                           "transport_hubs.geojson")
    if not os.path.exists(area_path):
        raise SystemExit(f"No area file at {area_path}")

    print(f"Scenario : {scenario.name}")
    print(f"Area     : {area_path}")

    area = gpd.read_file(area_path)
    if area.crs is None or area.crs.to_epsg() != 4326:
        area = area.to_crs("EPSG:4326")
    polygon = area.union_all()
    km2 = gpd.GeoSeries([polygon], crs=4326).to_crs(3857).area.iloc[0] / 1e6
    print(f"           {km2:,.0f} km2")

    gdf = fetch(polygon, HUB_TAGS, args.retries, args.timeout)
    print(f"  {len(gdf):,} features returned")

    # A feature matching several schemes comes back once per scheme. fetch()
    # guarantees osmid exists, so this is unconditional -- guarding it on the
    # column's presence is how it came to be skipped entirely.
    before = len(gdf)
    gdf = gdf.drop_duplicates(subset=["element", "osmid"])
    if before != len(gdf):
        print(f"  {before - len(gdf):,} duplicate(s) removed "
              "(one feature matching several tag schemes)")

    classified = gdf.apply(classify, axis=1, result_type="expand")
    gdf["hub_type"], gdf["hub_scale"] = classified[0], classified[1]

    # A single position for every feature, whatever its geometry. Computed in a
    # projected CRS: a centroid taken in degrees is wrong by a little everywhere
    # and by a lot away from the equator.
    projected = gdf.to_crs(gdf.estimate_utm_crs())
    centroids = projected.geometry.centroid.to_crs("EPSG:4326")
    gdf["centroid_lat"] = centroids.y.round(7)
    gdf["centroid_lon"] = centroids.x.round(7)

    if args.as_points:
        gdf = gdf.set_geometry(centroids)
        print("  geometry replaced with centroids (--as-points)")

    if args.full:
        # Everything a mapper populated. Drop columns no feature filled in --
        # OSM returns a union of every tag seen, so most are entirely empty.
        empty = [c for c in gdf.columns
                 if c != "geometry" and gdf[c].isna().all()]
        gdf = gdf.drop(columns=empty)
        print(f"  keeping all {len(gdf.columns)} populated OSM tags (--full)")
    else:
        # The core set by default: 125 sparse columns make the file large and
        # awkward to read, and the identity, classification and position are
        # what almost every consumer wants.
        keep = [c for c in CORE_COLUMNS if c in gdf.columns]
        gdf = gdf[keep]
    for column in gdf.columns:
        if column != "geometry":
            gdf[column] = gdf[column].map(scalarise)

    print(f"\n  {'hub_type':10s} {'interchange':>12s} {'stop':>8s} {'total':>8s}")
    summary = gdf.groupby(["hub_type", "hub_scale"]).size().unstack(fill_value=0)
    for hub_type in summary.index:
        inter = int(summary.loc[hub_type].get("interchange", 0))
        stop = int(summary.loc[hub_type].get("stop", 0))
        print(f"  {hub_type:10s} {inter:>12,} {stop:>8,} {inter + stop:>8,}")
    print(f"  {'TOTAL':10s} "
          f"{int(summary.get('interchange', pd.Series(dtype=int)).sum()):>12,} "
          f"{int(summary.get('stop', pd.Series(dtype=int)).sum()):>8,} "
          f"{len(gdf):>8,}")
    print(f"\n  geometry: {gdf.geom_type.value_counts().to_dict()}")
    print(f"  columns : {len(gdf.columns)}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # The file being replaced may have been edited by hand in the facility
    # editor, so one copy is kept aside rather than silently lost.
    if os.path.exists(out_path) and not args.no_backup:
        backup = os.path.splitext(out_path)[0] + ".previous.geojson"
        if not os.path.exists(backup):
            shutil.copy2(out_path, backup)
            print(f"\n  previous file kept at {os.path.basename(backup)}")
    gdf.to_file(out_path, driver="GeoJSON")

    print(f"\nWrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scenario_argument(p)
    p.add_argument("--area", default=None,
                   help="boundary to query inside (default: the scenario's "
                        "geojson_files/area.geojson)")
    p.add_argument("--output", default=None,
                   help="GeoJSON to write (default: the scenario's "
                        "geojson_files/transport_hubs.geojson)")
    p.add_argument("--as-points", action="store_true",
                   help="replace station footprints with their centroid, so "
                        "every feature is a Point")
    p.add_argument("--full", action="store_true",
                   help="keep every populated OSM tag instead of the core "
                        "identifying columns; over a hundred mostly-empty "
                        "columns, but nothing is discarded")
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                   help="attempts before giving up when Overpass times out or "
                        f"rate limits (default {DEFAULT_RETRIES})")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S,
                   help="seconds to wait for each Overpass response "
                        f"(default {DEFAULT_TIMEOUT_S})")
    p.add_argument("--no-backup", action="store_true",
                   help="overwrite an existing file without copying it aside")
    return p.parse_args()


if __name__ == "__main__":
    main()
