from datetime import datetime
import copy

def parse_time(t):
    return datetime.fromisoformat(t)

def create_idle_point(prev_feat, next_feat):
    """
    Create an 'idle' point feature between two features.
    Uses the last coordinate of the previous feature.
    """
    # Get coordinates for idle point
    if prev_feat["geometry"]["type"] == "Point":
        coords = prev_feat["geometry"]["coordinates"]
    elif prev_feat["geometry"]["type"] == "LineString":
        coords = prev_feat["geometry"]["coordinates"][-1]
    else:
        return None

    start_time = prev_feat["properties"]["end_time"]
    end_time = next_feat["properties"]["start_time"]

    start_dt = parse_time(start_time)
    end_dt = parse_time(end_time)

    duration_s = (end_dt - start_dt).total_seconds()

    if duration_s <= 0:
        return None

    idle_feat = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": coords
        },
        "properties": {
            "agent": prev_feat["properties"]["agent"],
            "type": "idle",
            "duration_s": duration_s,
            "start_time": start_time,
            "end_time": end_time
        }
    }

    return idle_feat


def insert_idle_points(geojson):
    features = geojson["features"]

    # Group by agent
    by_agent = {}
    for f in features:
        agent = f["properties"]["agent"]
        by_agent.setdefault(agent, []).append(f)

    new_features = []

    for agent, feats in by_agent.items():
        # Sort by start_time
        feats_sorted = sorted(feats, key=lambda f: parse_time(f["properties"]["start_time"]))

        for i in range(len(feats_sorted)):
            current_feat = feats_sorted[i]
            new_features.append(current_feat)

            if i < len(feats_sorted) - 1:
                next_feat = feats_sorted[i + 1]

                current_end = parse_time(current_feat["properties"]["end_time"])
                next_start = parse_time(next_feat["properties"]["start_time"])

                # Check for gap
                if next_start > current_end:
                    idle_feat = create_idle_point(current_feat, next_feat)
                    if idle_feat:
                        new_features.append(idle_feat)

    return {
        "type": "FeatureCollection",
        "features": new_features
    }