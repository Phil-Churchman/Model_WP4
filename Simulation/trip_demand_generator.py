import os, sys, json, random
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import load_scenario, scenario_from_cli


# ============================================================
# CONFIGURATION
# ============================================================
# Resolved when generate_trips() is called, not at import. The simulation
# imports this module before it has settled which scenario is in play, so
# reading the config at import time would bind to the wrong one.


def generate_trips(scenario=None):
    scenario = scenario or load_scenario()

    INPUT_DIR = scenario.input_dir
    OUTPUT_DIR = scenario.output_dir
    FREQUENCY_FILE = os.path.join(INPUT_DIR, "demand_frequencies.json")
    DEMAND_POINTS = os.path.join(INPUT_DIR, "demand_points.geojson")

    d_start, d_end = scenario["start_time"], scenario["end_time"]
    DAY_START = datetime(d_start[0], d_start[1], d_start[2], d_start[3], d_start[4], d_start[5])
    DAY_END = datetime(d_end[0], d_end[1], d_end[2], d_end[3], d_end[4], d_end[5])

    # Load Data
    with open(FREQUENCY_FILE, 'r') as f:
        profiles = json.load(f)

    with open(DEMAND_POINTS, 'r') as f:
        points_data = json.load(f)

    # 1. Map facilities by category for weighted random selection
    category_map = {}
    for feature in points_data['features']:
        cat = feature['properties']['category']
        if cat not in category_map:
            category_map[cat] = {"features": [], "weights": []}
        
        category_map[cat]["features"].append(feature)
        # Use weight_in_category from properties
        category_map[cat]["weights"].append(feature['properties'].get('weight_in_category', 1))

    trip_features = []
    
    # 2. Step through each one-hour window from START to END
    current_window_start = DAY_START
    
    while current_window_start < DAY_END:
        # Determine the current window's boundary (1 hour later or DAY_END, whichever is first)
        next_hour = current_window_start + timedelta(hours=1)
        window_end = min(next_hour, DAY_END)
        
        # Get indices for frequency arrays
        day_idx = current_window_start.weekday() # 0=Mon, 6=Sun
        hour_idx = current_window_start.hour     # 0-23
        
        for profile in profiles:
            src_cat = profile['source']
            dest_cat = profile['destination']
            
            if src_cat not in category_map or dest_cat not in category_map:
                continue

            # TRIPS = Weekly Frequency * Hourly Frequency
            num_trips = int(profile['weekly'][day_idx] * profile['hourly'][hour_idx])
            
            for _ in range(num_trips):
                # Weighted random selection of points
                while True:
                    src_feat = random.choices(
                        category_map[src_cat]["features"], 
                        weights=category_map[src_cat]["weights"]
                    )[0]
                    
                    dest_feat = random.choices(
                        category_map[dest_cat]["features"], 
                        weights=category_map[dest_cat]["weights"]
                    )[0]

                    if src_feat != dest_feat: break

                # Allocate to a random time within the current 1-hour window
                # We calculate the delta in seconds to ensure it stays within the hour/bounds
                max_seconds = int((window_end - current_window_start).total_seconds())
                random_offset = random.randint(0, max_seconds)
                trip_time = current_window_start + timedelta(seconds=random_offset)

                # Build GeoJSON Feature (LineString)
                trip_feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            src_feat['geometry']['coordinates'],
                            dest_feat['geometry']['coordinates']
                        ]
                    },
                    "properties": {
                        "departure_time": trip_time.isoformat(),
                        "source_facility": src_feat['properties']['facility_id'],
                        "dest_facility": dest_feat['properties']['facility_id'],
                        "source_category": src_cat,
                        "dest_category": dest_cat
                    }
                }
                trip_features.append(trip_feature)

        # Advance to the next hour
        current_window_start = next_hour

    # 3. Save to Output
    output_collection = {
        "type": "FeatureCollection",
        "features": trip_features
    }

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    output_path = os.path.join(OUTPUT_DIR, "trip_demand.geojson")
    with open(output_path, 'w') as f:
        json.dump(output_collection, f, indent=2)
    
    print(f"Generated {len(trip_features)} trips to {output_path}")

    return output_collection

if __name__ == "__main__":
    generate_trips(scenario_from_cli("Generate synthetic trip demand"))