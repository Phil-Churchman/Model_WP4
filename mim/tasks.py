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
#   confirm     replaces the current results, but the previous state is
#               recoverable -- the simulation archives the run it replaces, and
#               the area builder keeps a backup. Allowed, on an explicit yes.
#   irreversible allowed too, but nothing is kept: re-downloading a road network
#               replaces the one your finished results were computed against,
#               with today's OSM data. Warned about in the strongest terms the
#               UI has, then run if you say so.
#   destructive not offered at all, because the damage is not just loss --
#               captured_trips writes the simulation's own filenames into the
#               simulation's own folder, leaving two datasets that look like one
READ, DERIVE, CONFIRM, IRREVERSIBLE, DESTRUCTIVE = (
    "read", "derive", "confirm", "irreversible", "destructive")


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
         "Simulation/Simulation.py", CONFIRM,
         "Runs the fleet model. The run it replaces is moved to output/runs/ "
         "first, unless the output is too large to be worth archiving.",
         "model"),
    Task("extract_roads", "Download road network",
         "Road extraction/OSM_road_extractor.py", IRREVERSIBLE,
         "Downloads the current OSM network for the scenario's area and "
         "overwrites roads.graphml and roads.geojson. No backup is kept.",
         "inputs"),
    Task("area_from_trips", "Area from captured trips",
         "utilities/generate_area_from_trips.py", CONFIRM,
         "Rebuilds area.geojson from captured GPS. Keeps one backup.",
         "inputs"),
    Task("captured_trips", "Captured trips to GeoJSON",
         "calibration/captured_trips_to_geojson.py", DESTRUCTIVE,
         "Routes captured GPS trips and writes per-user tracks into output. "
         "Writes the simulation's own filenames into the simulation's own "
         "folder, so it can silently merge two datasets.",
         "model"),

    # Everything in calibration/ reads trip_routing_analysis.csv and writes a
    # derived CSV or figure alongside it. Nothing is overwritten that cannot be
    # rebuilt by re-running, so these are all safe from the browser.
    Task("build_distributions", "Build trip distributions",
         "calibration/build_trip_distributions.py", DERIVE,
         "Derives the trip distance and fare wait distributions from the "
         "chained capture and writes trip_distributions.json into the scenario "
         "folder. Only distribution mode reads it.",
         "calibrate"),
    Task("compare_distributions", "Compare run vs distributions",
         "calibration/compare_trip_distributions.py", DERIVE,
         "Plots the trip distances and fare waits a distribution-mode run "
         "actually produced against the ones it was asked for. Writes a PNG "
         "and a CSV; changes nothing.",
         "calibrate"),
    Task("deviation_factor", "Measure deviation factor",
         "calibration/measure_deviation_factor.py", DERIVE,
         "Reports the road / straight-line distance ratio a run exhibits, "
         "overall and by trip type and distance band. The median is the "
         "value to put in deviation_factor.",
         "calibrate"),
    Task("calibrate_speeds", "Calibrate road speeds",
         "calibration/calibrate_road_speeds.py", DERIVE,
         "Fits road_speed_km-h to the recorded trip durations. Writes a "
         "recommendation; changes nothing in the scenario.",
         "calibrate"),
    Task("distance_discrepancy", "Distance discrepancy",
         "calibration/distance_discrepancy.py", DERIVE,
         "Distribution of routed minus recorded trip distance.",
         "calibrate"),
    Task("trip_gap_analysis", "Trip length by idle gap",
         "calibration/trip_gap_analysis.py", DERIVE,
         "Trip distance distributions split by how long the vehicle had waited.",
         "calibrate"),
    Task("trip_gap_correlation", "Consecutive idle gaps",
         "calibration/trip_gap_correlation.py", DERIVE,
         "Whether the gap before a trip predicts the gap before the next.",
         "calibrate"),
    Task("trip_gap_bin_search", "Idle gap bin search",
         "calibration/trip_gap_bin_search.py", DERIVE,
         "Sweeps gap bin counts to find the strongest association.",
         "calibrate"),
]

TASKS_BY_ID = {t.id: t for t in TASKS}

# What the browser may launch. `confirm` became launchable once the simulation
# started archiving the run it replaces; `irreversible` is launchable because
# the warning is explicit about what will not come back. Only `destructive`
# stays terminal-only, where the command has to be typed deliberately.
RUNNABLE_SAFETY = {READ, DERIVE, CONFIRM, IRREVERSIBLE}
NEEDS_CONFIRMATION = {CONFIRM, IRREVERSIBLE}


def is_runnable(task):
    return task.safety in RUNNABLE_SAFETY and task.exists


def needs_confirmation(task):
    return task.safety in NEEDS_CONFIRMATION


def is_irreversible(task):
    return task.safety == IRREVERSIBLE


# What each irreversible task destroys, so the warning can be specific rather
# than a generic "are you sure". Paths are relative to the scenario folder.
OVERWRITES = {
    "extract_roads": ["geojson_files/roads.graphml", "geojson_files/roads.geojson"],
}


@dataclass(frozen=True)
class Tool:
    """An HTML tool, served from the Model folder at its own path."""
    label: str
    path: str                        # relative to MODEL_DIR
    description: str
    group: str
    # True for the pages that load a scenario themselves. The panel appends
    # ?scenario=<file> so they follow its picker; opened directly they see no
    # parameter and fall back to scenario.json, as they always did.
    accepts_scenario: bool = False

    @property
    def exists(self):
        return os.path.exists(os.path.join(MODEL_DIR, self.path.replace("/", os.sep)))


TOOLS = [
    Tool("Fleet animation", "animation/animation.html",
         "Plays the simulated vehicle tracks on a map.", "view", True),
    Tool("Swap station animation", "animation/station_animation.html",
         "Queue and swap activity at each station over the day.", "view", True),
    Tool("Cleaned GPS animation", "utilities/clean_data_animation.html",
         "Replays captured GPS after cleaning.", "view"),
    Tool("View cleaned data", "utilities/view_cleaned_data.html",
         "Inspect cleaned GPS traces.", "view"),
    Tool("Business case dashboard", "utilities/EV_3Wheeler_Business_Case_Dashboard.html",
         "Three-wheeler battery-swap business case.", "view"),

    Tool("Draw scenario area", "utilities/set_area.html",
         "Draw or fetch an area boundary, then download area.geojson.", "edit"),
    Tool("Edit facilities", "utilities/edit_facility.html",
         "Place swap stations and taxi ranks.", "edit"),
    Tool("Edit demand points", "utilities/edit_demand points.html",
         "Place and weight demand points.", "edit"),
    Tool("Edit demand frequencies", "utilities/edit_demand_frequencies.html",
         "Hourly and weekly trip frequency profiles.", "edit"),
    Tool("GeoJSON editor", "utilities/geojson_editor.html",
         "General-purpose GeoJSON property editor.", "edit"),
]

TOOL_GROUPS = ["view", "edit"]


# When an artefact is needed. Which inputs are required is not fixed: it turns
# on the simulation mode, and each of the three reads a different set -- taxi
# ranks in hail_rank, demand points and frequencies in demand_model, the trip
# distributions in distribution.
ALWAYS = "always"
HAIL_RANK, DEMAND_MODEL_MODE, DISTRIBUTION = "hail_rank", "demand_model", "distribution"


def scenario_mode(scenario):
    """
    The mode a scenario selects, tolerating an unmigrated file.

    The board must render for every scenario on disk, including a malformed one,
    so an unknown mode falls back to hail_rank here rather than raising the way
    the simulation does -- the run itself will still refuse to start.
    """
    getter = getattr(scenario, "get", None)
    cfg = scenario if getter is None else scenario
    mode = cfg.get("simulation_mode")
    if mode is None:
        return DEMAND_MODEL_MODE if cfg.get("demand_model", False) else HAIL_RANK
    mode = str(mode).strip().lower()
    return mode if mode in (HAIL_RANK, DEMAND_MODEL_MODE, DISTRIBUTION) else HAIL_RANK


@dataclass(frozen=True)
class Artefact:
    key: str
    label: str
    path: str                        # relative to the scenario folder
    stage: str
    depends_on: tuple = ()           # keys of artefacts this is derived from
    produced_by: str = None          # task id that makes it
    required_when: str = ALWAYS       # ALWAYS, or the one mode that reads it

    def is_required(self, mode):
        return self.required_when in (ALWAYS, mode)


# The pipeline, as a dependency graph. An artefact older than something it was
# derived from is stale -- which is the question the board exists to answer, and
# the one that is impossible to hold in your head across eight scenario folders.
ARTEFACTS = [
    Artefact("area", "area.geojson", "geojson_files/area.geojson", "inputs",
             produced_by="area_from_trips"),
    Artefact("roads", "roads.graphml", "geojson_files/roads.graphml", "inputs",
             depends_on=("area",), produced_by="extract_roads"),
    Artefact("swap", "swap_stations.geojson", "geojson_files/swap_stations.geojson",
             "inputs", depends_on=("area",)),
    # Only hail_rank routes to a rank; the other two modes never load the file.
    Artefact("ranks", "taxi_ranks.geojson", "geojson_files/taxi_ranks.geojson",
             "inputs", depends_on=("area",), required_when=HAIL_RANK),
    # Trip distances and fare waits for distribution mode. Resolved from the
    # scenario folder first, so a per-scenario copy is what the board tracks.
    Artefact("distributions", "trip_distributions.json",
             "trip_distributions.json", "inputs",
             produced_by="build_distributions", required_when=DISTRIBUTION),
    # Both of these are read by trip_demand_generator, and it opens them
    # unguarded: either one missing is a crash, not a degraded run.
    Artefact("demand_points", "demand_points.geojson",
             "geojson_files/demand_points.geojson", "inputs",
             depends_on=("area",), required_when=DEMAND_MODEL_MODE),
    Artefact("demand_freqs", "demand_frequencies.json",
             "geojson_files/demand_frequencies.json", "inputs",
             required_when=DEMAND_MODEL_MODE),

    Artefact("tracks", "agent tracks", "output/output_trips_time_queued", "model",
             depends_on=("roads", "swap"), produced_by="simulate"),
    Artefact("trip_demand", "trip_demand.geojson", "output/trip_demand.geojson",
             "model", depends_on=("demand_points", "demand_freqs"),
             produced_by="simulate", required_when=DEMAND_MODEL_MODE),
    Artefact("met", "met_demand.json", "output/met_demand.json", "model",
             depends_on=("roads", "swap"), produced_by="simulate",
             required_when=DEMAND_MODEL_MODE),
    Artefact("timesteps", "swap_station_timesteps.xlsx",
             "output/swap_station_timesteps.xlsx", "model",
             depends_on=("roads", "swap"), produced_by="simulate"),

    Artefact("productivity", "agent_time_statistics.csv",
             "output/agent_time_statistics.csv", "analyse",
             depends_on=("tracks",), produced_by="productivity"),
    Artefact("visits", "swap_station_visits.xlsx", "output/swap_station_visits.xlsx",
             "analyse", depends_on=("tracks",), produced_by="station_visits"),
    Artefact("arrivals", "swap_station_arrivals.csv",
             "output/swap_station_arrivals.csv", "analyse",
             depends_on=("tracks",), produced_by="station_utilisation"),
]

ARTEFACTS_BY_KEY = {a.key: a for a in ARTEFACTS}

STAGES = ["inputs", "model", "analyse", "calibrate"]
