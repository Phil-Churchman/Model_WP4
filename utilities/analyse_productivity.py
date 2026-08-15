import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import scenario_from_cli


def process_agent_data(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".geojson")]
    all_agent_records = []

    label_mapping = {
        "Stop - passenger": "Stop - passenger drop",
        "Stop - taxi": "Stop - taxi rank",
        "Stop - to_swap": "Stop - swap station",
        "Trip - hail": "Trip - to hail",
        "Trip - taxi": "Trip - to taxi rank",
        "Trip - to_swap": "Trip - to swap station",
        "Trip - passenger": "Trip - passenger"
    }

    for file in files:
        agent_id = file.split("_")[1] 
        with open(os.path.join(directory, file), "r") as f:
            data = json.load(f)
        
        sums = {}
        for feat in data.get("features", []):
            g_type = feat["geometry"]["type"]
            category = "Trip" if g_type == "LineString" else "Stop"
            trip_type = feat["properties"]["type"]
            duration_min = feat["properties"].get("duration_s", 0) / 60
            
            key = (category, trip_type)
            sums[key] = sums.get(key, 0) + duration_min
            
        for (cat, t_type), total in sums.items():
            original_label = f"{cat} - {t_type}"
            final_label = label_mapping.get(original_label, original_label)
            
            all_agent_records.append({
                "agent_id": agent_id,
                "label": final_label,
                "total_time": total
            })
            
    return pd.DataFrame(all_agent_records)

def main():
    scenario = scenario_from_cli("Agent productivity statistics and histograms")
    input_path = scenario.trips_time_dir
    output_path = scenario.output_dir

    df = process_agent_data(input_path)

    # Save Statistics CSV
    stats = df.groupby("label")["total_time"].agg(["max", "min", "mean", "std"]).reset_index()
    stats.columns = ["Activity Type", "Max Time (min)", "Min Time (min)", "Average Time (min)", "Std Dev (min)"]
    stats.to_csv(os.path.join(output_path, "agent_time_statistics.csv"), index=False)

    # Create Grid Histograms
    labels = sorted(df["label"].unique())
    cols = 2
    rows = (len(labels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.flatten()

    for i, label in enumerate(labels):
        subset = df[df["label"] == label]["total_time"]

        t_min = int(np.floor(subset.min()))
        t_max = int(np.ceil(subset.max()))
        t_range = t_max - t_min

        # Determine bin strategy
        if t_range <= 20:
            # One bin per integer, centered on the integer
            bins = np.arange(t_min, t_max + 2) - 0.5
        else:
            # Limit to 20 bins for large ranges
            # We still shift by 0.5 to encourage integer centering where possible
            bins = np.linspace(t_min, t_max + 1, 21) - 0.5

        # Plot histogram
        axes[i].hist(subset, bins=bins, color='teal', edgecolor='black', alpha=0.7, rwidth=0.8)

        # Centre the x-axis with padding
        axes[i].set_xlim(left=t_min - 1, right=t_max + 1)

        # Force integer ticks on the X axis
        axes[i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))

        axes[i].set_title(label, fontweight='bold')
        axes[i].set_xlabel("Total Time (min)")
        axes[i].set_ylabel("Frequency")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "agent_time_histograms.png"))
    print(f"Wrote agent_time_statistics.csv and agent_time_histograms.png to {output_path}")


if __name__ == "__main__":
    main()