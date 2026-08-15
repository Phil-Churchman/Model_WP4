"""
The scripts the server knows how to run, and how dangerous each one is.

Nothing here reimplements a script -- a task is just a name, a path and a safety
class. The command built from it is the same one you would type.
"""

import os
from dataclasses import dataclass, field

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Safety classes, in increasing order of consequence:
#   read        writes nothing
#   derive      adds derived files alongside existing output; nothing is lost
#   destructive overwrites or deletes something that cannot be regenerated
#               cheaply -- a road network download, or clear_output_dir() wiping
#               a whole run
READ, DERIVE, DESTRUCTIVE = "read", "derive", "destructive"


@dataclass(frozen=True)
class Task:
    id: str
    label: str
    script: str                      # relative to MODEL_DIR
    safety: str
    description: str
    stage: str                       # pipeline stage, for grouping
    extra_args: list = field(default_factory=list)

    @property
    def path(self):
        return os.path.join(MODEL_DIR, self.script)

    @property
    def exists(self):
        return os.path.exists(self.path)

    def command(self, python, scenario_path):
        return [python, self.path, "--scenario", scenario_path, *self.extra_args]


TASKS = [
    Task("check_output", "Check output integrity",
         "utilities/check_output.py", READ,
         "Verifies every agent has a continuous trip-stop-trip timeline.",
         "analyse"),
    Task("productivity", "Agent productivity",
         "utilities/analyse_productivity.py", DERIVE,
         "Writes agent_time_statistics.csv and agent_time_histograms.png.",
         "analyse"),
    Task("station_visits", "Swap station visits",
         "utilities/extract_station_visits.py", DERIVE,
         "Exports swap_station_visits.xlsx from the per-agent tracks.",
         "analyse"),
    Task("station_utilisation", "Swap station utilisation",
         "utilities/swap_station_utilisation.py", DERIVE,
         "Arrival profile per station: CSV plus a grid of bar charts.",
         "analyse"),

    Task("simulate", "Run simulation",
         "Simulation/Simulation.py", DESTRUCTIVE,
         "Runs the fleet model. Clears the scenario's whole output folder first.",
         "model"),
    Task("extract_roads", "Download road network",
         "Road extraction/OSM_road_extractor.py", DESTRUCTIVE,
         "Downloads from OSM and overwrites roads.graphml and roads.geojson.",
         "inputs"),
    Task("area_from_trips", "Area from captured trips",
         "utilities/generate_area_from_trips.py", DESTRUCTIVE,
         "Rebuilds area.geojson from captured GPS (keeps one backup).",
         "inputs"),
    Task("captured_trips", "Captured trips to GeoJSON",
         "Simulation/captured_trips_to_geojson.py", DESTRUCTIVE,
         "Routes captured GPS trips and writes per-user tracks into output.",
         "model"),
]

TASKS_BY_ID = {t.id: t for t in TASKS}

# Phase 1 runs only what cannot destroy work. The destructive tasks are listed
# and described so the UI can show the whole pipeline, but the server refuses to
# launch them until run isolation exists -- a browser tab left open is a poor
# place to trigger clear_output_dir().
RUNNABLE_SAFETY = {READ, DERIVE}


def is_runnable(task):
    return task.safety in RUNNABLE_SAFETY and task.exists


# Artefacts that make up a scenario, for the pipeline view. Paths are relative
# to the scenario folder.
ARTEFACTS = [
    ("area.geojson", "geojson_files/area.geojson", "inputs"),
    ("roads.graphml", "geojson_files/roads.graphml", "inputs"),
    ("swap_stations.geojson", "geojson_files/swap_stations.geojson", "inputs"),
    ("taxi_ranks.geojson", "geojson_files/taxi_ranks.geojson", "inputs"),
    ("demand_points.geojson", "geojson_files/demand_points.geojson", "inputs"),
    ("agent tracks", "output/output_trips_time_queued", "model"),
    ("met_demand.json", "output/met_demand.json", "model"),
    ("swap_station_timesteps.xlsx", "output/swap_station_timesteps.xlsx", "model"),
    ("agent_time_statistics.csv", "output/agent_time_statistics.csv", "analyse"),
    ("swap_station_visits.xlsx", "output/swap_station_visits.xlsx", "analyse"),
]
