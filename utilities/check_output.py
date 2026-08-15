import os
import sys
import json
import glob
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import scenario_from_cli


def check_agent_integrity(directory):
    files = glob.glob(os.path.join(directory, "*.geojson"))
    if not files:
        print("No files found in directory.")
        return

    global_earliest = None
    global_latest = None
    discontinuity_log = []
    
    # We'll allow a small tolerance (e.g., 1 second) for floating point/string conversion issues
    TOLERANCE_SECONDS = 1.0 

    for fpath in tqdm(sorted(files), desc="Reviewing Agents"):
        with open(fpath, "r") as f:
            data = json.load(f)
        
        features = data.get("features", [])
        if not features:
            continue

        for i in range(len(features)):
            props = features[i]["properties"]
            start_t = datetime.fromisoformat(props["start_time"])
            end_t = datetime.fromisoformat(props["end_time"])

            # 1. Update Global Bounds
            if global_earliest is None or start_t < global_earliest:
                global_earliest = start_t
            if global_latest is None or end_t > global_latest:
                global_latest = end_t

            # 2. Check Sequence Continuity with the NEXT feature
            if i < len(features) - 1:
                next_props = features[i+1]["properties"]
                next_start_t = datetime.fromisoformat(next_props["start_time"])
                
                gap = (next_start_t - end_t).total_seconds()
                
                if abs(gap) > TOLERANCE_SECONDS:
                    status = "GAP" if gap > 0 else "OVERLAP"
                    discontinuity_log.append({
                        "agent": os.path.basename(fpath),
                        "type": status,
                        "seconds": gap,
                        "between_index": f"{i} and {i+1}",
                        "time_loc": end_t.isoformat()
                    })

    # --- PRINT RESULTS ---
    print("\n" + "="*50)
    print("SIMULATION BOUNDS REPORT")
    print("="*50)
    print(f"Earliest Start: {global_earliest}")
    print(f"Latest End:    {global_latest}")
    
    total_span = (global_latest - global_earliest).total_seconds() / 3600
    print(f"Total Fleet-wide Span: {total_span:.2f} hours")
    print("="*50)

    if not discontinuity_log:
        print("\n✅ SUCCESS: All agents have a perfectly continuous trip-stop-trip sequence.")
    else:
        print(f"\n❌ FOUND {len(discontinuity_log)} DISCONTINUITIES:")
        # Print the first 10 errors as examples
        for log in discontinuity_log[:50]:
            print(f"- {log['agent']}: {log['type']} of {log['seconds']}s at {log['time_loc']}")
        
        if len(discontinuity_log) > 50:
            print(f"... and {len(discontinuity_log) - 10} more.")

def main():
    scenario = scenario_from_cli("Check per-agent output for timeline discontinuities")
    output_dir = scenario.trips_time_dir
    print(output_dir)
    check_agent_integrity(output_dir)


if __name__ == "__main__":
    main()