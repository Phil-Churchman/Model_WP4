import os
import sys
import json
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import scenario_from_cli


def collect_swap_records(input_path):
    swap_records = []

    # Verify directory exists and gather relevant files
    if os.path.exists(input_path):
        files_to_process = [
            f for f in os.listdir(input_path)
            if f.endswith(".json") or f.endswith(".geojson")
        ]

        # Process files with a progress bar
        for file_name in tqdm(files_to_process, desc="Processing GeoJSON files", unit="file"):
            file_filepath = os.path.join(input_path, file_name)

            with open(file_filepath, "r") as f:
                data = json.load(f)

            # Filter and parse GeoJSON features
            features = data.get("features", [])
            for feature in features:
                geometry_type = feature.get("geometry", {}).get("type")
                properties = feature.get("properties", {})

                # Check for Point features with type "to_swap"
                if geometry_type == "Point" and properties.get("type") == "to_swap":
                    record = {
                        "agent_id": properties.get("agent"),
                        "swap_station": properties.get("facility_id"),
                        "arrival_distance": properties.get("total_distance_m"),
                        "arrival_time": properties.get("start_time"),
                        "departure_time": properties.get("end_time")
                    }
                    swap_records.append(record)

    return swap_records


def main():
    scenario = scenario_from_cli("Export swap station visits to Excel")
    input_path = scenario.trips_time_dir
    output_path = scenario.output_dir

    swap_records = collect_swap_records(input_path)

    # Create DataFrame and export to Excel
    print("\nExporting to Excel...")
    df_swaps = pd.DataFrame(swap_records)

    os.makedirs(output_path, exist_ok=True)
    excel_output_path = os.path.join(output_path, "swap_station_visits.xlsx")
    df_swaps.to_excel(excel_output_path, index=False)

    print(f"Done! Exported {len(df_swaps)} records to {excel_output_path}")


if __name__ == "__main__":
    main()