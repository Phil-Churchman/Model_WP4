#!/usr/bin/env python3
"""
analyze_swap_station_arrivals_grid.py
"""

import os, datetime
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import scenario_from_cli

# ==============================
# SETTINGS
# ==============================
TIME_BIN_MINUTES = 15
NEAREST_TOLERANCE = 0.005  # ~500m


def main():
    scenario = scenario_from_cli("Swap station arrival profile")
    FOLDER_NAME = scenario.folder_name

    OUTPUT_DIR = scenario.output_dir
    INPUT_DIR = scenario.input_dir
    OUTPUT_PER_AGENT_TIME_DIR = scenario.trips_time_dir
    SWAP_STATIONS_FILE = os.path.join(INPUT_DIR, "swap_stations.geojson")
    CSV_OUTPUT = os.path.join(OUTPUT_DIR, "swap_station_arrivals.csv")
    FIG_OUTPUT = os.path.join(OUTPUT_DIR, "swap_station_arrivals_grid.png")

    d_start, d_end = scenario["start_time"], scenario["end_time"]
    START_TIME = datetime(d_start[0], d_start[1], d_start[2], d_start[3], d_start[4], d_start[5])
    END_TIME = datetime(d_end[0], d_end[1], d_end[2], d_end[3], d_end[4], d_end[5])

    # ==============================
    # LOAD SWAP STATIONS
    # ==============================
    with open(SWAP_STATIONS_FILE, "r") as f:
        data = json.load(f)

    swap_stations = []
    for feature in data.get("features", []):
        geom = feature.get("geometry")
        props = feature.get("properties", {})

        if geom and geom.get("type") == "Point":
            fid = props.get("facility_id", len(swap_stations))
            posts = props.get("posts", 2)
            swap_stations.append({
                "id": fid,
                "posts": posts,
                "coords": tuple(geom["coordinates"])
            })

    swap_stations = sorted(swap_stations, key=lambda s: s["id"])
    num_stations = len(swap_stations)

    # ==============================
    # CREATE TIME BINS
    # ==============================
    bins = pd.date_range(start=START_TIME, end=END_TIME, freq=f"{TIME_BIN_MINUTES}min")
    station_ids = [s["id"] for s in swap_stations]
    df = pd.DataFrame(0, index=bins[:-1], columns=station_ids)

    # ==============================
    # PROCESS PER-AGENT TIME GEOJSONS
    # ==============================
    if os.path.exists(OUTPUT_PER_AGENT_TIME_DIR):
        files = [f for f in os.listdir(OUTPUT_PER_AGENT_TIME_DIR) if f.endswith("_time.geojson")]
        for file in tqdm(files, desc="Processing agents"):
            with open(os.path.join(OUTPUT_PER_AGENT_TIME_DIR, file), "r") as f:
                data = json.load(f)
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                geom = feature.get("geometry", {})
                if geom.get("type") != "Point" or props.get("type") != "to_swap":
                    continue
                x, y = geom["coordinates"]
                nearest_id, min_dist_sq = None, float("inf")
                for s in swap_stations:
                    sx, sy = s["coords"]
                    dist_sq = (sx - x)**2 + (sy - y)**2
                    if dist_sq < min_dist_sq and dist_sq < NEAREST_TOLERANCE**2:
                        nearest_id, min_dist_sq = s["id"], dist_sq
                if nearest_id is not None:
                    t = pd.to_datetime(props["start_time"])
                    bin_idx = bins.searchsorted(t) - 1
                    if 0 <= bin_idx < len(df):
                        df.at[bins[bin_idx], nearest_id] += 1

    df.to_csv(CSV_OUTPUT)

    # ==============================
    # PLOT GRID
    # ==============================
    # DYNAMIC COLUMN LOGIC: If only one station, use 1 column. Otherwise 2.
    cols = 1 if num_stations == 1 else 2
    rows = (num_stations + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(10 if cols == 1 else 15, rows * 4),
                             sharex=True,
                             sharey=(False if num_stations == 1 else True),
                             squeeze=False) # squeeze=False ensures axes is always 2D
    flat_axes = axes.flatten()

    total_arrivals = df.sum()
    busiest_ids = total_arrivals.sort_values(ascending=False).index[:3]
    time_numeric = np.arange(len(df))

    for idx, s in enumerate(swap_stations):
        ax = flat_axes[idx]
        s_id = s["id"]
        color = 'orange' if s_id in busiest_ids else 'steelblue'

        ax.bar(time_numeric, df[s_id], width=0.8, color=color)
        ax.set_title(f"Station {s_id} ({s['posts']} Posts)", fontsize=14, fontweight='bold')

        # X-axis logic
        xticks = np.arange(0, len(df), 8)
        xticklabels = [(START_TIME + timedelta(minutes=i * TIME_BIN_MINUTES)).strftime("%I%p").lstrip("0") for i in range(0, len(df), 8)]

        # Only show x-labels on the bottom-most row
        if idx >= num_stations - cols:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
        else:
            ax.set_xticks(xticks)
            ax.set_xticklabels([])

    # Remove unused axes if more than one column and an odd number of stations
    for idx in range(num_stations, len(flat_axes)):
        fig.delaxes(flat_axes[idx])

    fig.suptitle(f"Total Taxi Arrivals per Station\n(Scenario: {FOLDER_NAME})", fontsize=16)
    fig.text(0.02, 0.5, 'Number of Taxi Arrivals', va='center', rotation='vertical', fontsize=12)

    plt.tight_layout(rect=[0.05, 0.03, 1, 0.93])
    plt.savefig(FIG_OUTPUT, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
