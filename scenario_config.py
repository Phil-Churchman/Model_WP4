"""
Scenario configuration: the one place that knows how a scenario.json maps onto
directories on disk.

Every script used to open scenario.json itself, at import time, from a fixed
path, and then build its own paths out of ``folder_name``. That meant a scenario
could only be switched by editing (or copying over) scenario.json, and it meant
importing a script ran it. Both are fixed by going through here instead.

Backward compatibility is deliberate: a scenario file that only has
``folder_name`` keeps resolving exactly as it always did, including when
folder_name is itself a relative path such as "..\\Model_data\\accra". Nothing
has to be migrated for the existing files to keep working.

    Old style (still supported)          New style
    ---------------------------          ---------
    {                                    {
      "folder_name":                       "data_root":   "../Model_data",
        "..\\Model_data\\accra"             "data_url":    "/Model_data",
    }                                       "folder_name": "accra"
                                         }

The new style exists because ``folder_name`` was doing two incompatible jobs at
once: a filesystem path for Python, and a URL path for the browser tools. Those
agreed while the data lived inside the Model folder and stopped agreeing when it
moved out. ``data_root`` is the filesystem half, ``data_url`` the URL half.

Typical use in a script::

    from scenario_config import load_scenario, add_scenario_argument

    parser = argparse.ArgumentParser()
    add_scenario_argument(parser)
    args = parser.parse_args()
    scenario = load_scenario(args.scenario)

    roads = os.path.join(scenario.input_dir, "roads.graphml")
"""

import argparse
import json
import os
import sys

# The Windows console defaults to cp1252, which cannot encode the arrows, dashes
# and tick marks several scripts print -- check_output.py crashed on its own
# "all agents continuous" success message, after the check had passed. Widening
# stdout here fixes every script at once, because they all import this module,
# and it can only ever allow output that previously raised.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass  # already wrapped, or not a real stream (pytest, pipes, spawn)

# This file sits in the Model directory, which is what every scenario path has
# always been resolved relative to.
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCENARIO_PATH = os.path.join(MODEL_DIR, "scenario.json")

# How the choice of scenario reaches a child process. On Windows,
# multiprocessing uses "spawn": each worker re-imports the parent module, but
# with its own sys.argv, so a --scenario passed on the command line is invisible
# to it and it would silently fall back to scenario.json. Environment variables
# are inherited, so exporting the resolved path is what keeps the pool workers
# and the parent reading the same scenario.
SCENARIO_ENV_VAR = "MIM_SCENARIO"


class Scenario:
    """A loaded scenario.json, plus the directories it resolves to."""

    __slots__ = ("cfg", "path")

    def __init__(self, cfg, path):
        self.cfg = cfg
        self.path = path

    # -- raw access ------------------------------------------------------
    def get(self, key, default=None):
        return self.cfg.get(key, default)

    def __getitem__(self, key):
        return self.cfg[key]

    def __contains__(self, key):
        return key in self.cfg

    # -- identity --------------------------------------------------------
    @property
    def folder_name(self):
        return self.cfg["folder_name"]

    @property
    def name(self):
        """Short label for logs and filenames: the last path component."""
        return os.path.basename(str(self.folder_name).replace("\\", "/").rstrip("/"))

    # -- filesystem ------------------------------------------------------
    @property
    def data_root(self):
        """Directory the scenario folders live in, absolute."""
        root = self.cfg.get("data_root")
        return os.path.normpath(os.path.join(MODEL_DIR, root)) if root else MODEL_DIR

    @property
    def folder(self):
        """This scenario's own folder, absolute."""
        return os.path.normpath(os.path.join(self.data_root, self.folder_name))

    @property
    def input_dir(self):
        return os.path.join(self.folder, "geojson_files")

    @property
    def output_dir(self):
        return os.path.join(self.folder, "output")

    @property
    def captured_dir(self):
        return os.path.join(self.folder, "captured_locations")

    @property
    def trips_time_dir(self):
        return os.path.join(self.output_dir, "output_trips_time_queued")

    @property
    def trips_dir(self):
        return os.path.join(self.output_dir, "output_trips")

    # -- browser ---------------------------------------------------------
    @property
    def data_url(self):
        """
        URL prefix the scenario folders are served under, without a trailing
        slash. Only meaningful to the HTML tools; Python never uses it.

        Defaults to matching data_root's own directory name, which is what the
        Live Server "mount" setting in .vscode/settings.json currently serves.
        """
        url = self.cfg.get("data_url")
        if url:
            return "/" + url.strip("/")
        if self.cfg.get("data_root"):
            return "/" + os.path.basename(self.data_root)
        return ""

    # -- diagnostics -----------------------------------------------------
    def describe(self):
        """One block of text saying what resolved to what. Cheap insurance
        against running a long job against the wrong scenario."""
        return (f"scenario : {self.path}\n"
                f"folder   : {self.folder}\n"
                f"inputs   : {self.input_dir}\n"
                f"outputs  : {self.output_dir}")

    def require_folder(self):
        """Fail early, and legibly, rather than deep inside a load."""
        if not os.path.isdir(self.folder):
            raise SystemExit(
                f"Scenario folder does not exist:\n  {self.folder}\n"
                f"Resolved from {self.path} "
                f"(folder_name={self.folder_name!r}"
                + (f", data_root={self.cfg['data_root']!r}" if self.cfg.get("data_root") else "")
                + ")")
        return self


def load_scenario(path=None, export=False):
    """
    Load a scenario file. ``path`` may be a full path, a bare filename looked up
    in the Model directory, or None -- in which case the inherited
    MIM_SCENARIO environment variable is used, and failing that scenario.json.

    Pass ``export=True`` in a script that starts a multiprocessing pool: it
    publishes the resolved path so re-imports in spawned workers resolve to the
    same scenario instead of falling back to the default.
    """
    path = path or os.environ.get(SCENARIO_ENV_VAR) or DEFAULT_SCENARIO_PATH
    if not os.path.isabs(path):
        # Accept both "scenario_accra.json" and a path relative to the cwd.
        candidate = os.path.join(MODEL_DIR, path)
        path = candidate if os.path.exists(candidate) else os.path.abspath(path)

    if not os.path.exists(path):
        raise SystemExit(f"Scenario file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "folder_name" not in cfg:
        raise SystemExit(f"{path} has no \"folder_name\".")

    if export:
        os.environ[SCENARIO_ENV_VAR] = path

    return Scenario(cfg, path)


def dumps_scenario(cfg, indent=2, inline_width=78):
    """
    Serialise a scenario dict the way the existing files are written by hand:
    two-space indent, but a short array of scalars stays on one line.

    json.dumps(indent=2) explodes "agents": [0,2,4,8,10,8,4,2] into ten lines,
    which turns every save into a large diff and buries the change that was
    actually made.

    The hand-written files are not internally consistent -- "agents" has no
    space after its commas, "start_time" does -- so no single rule reproduces
    all of them exactly. This follows the majority style; the first save of a
    file reformats its "agents" line by whitespace only, once, and is stable
    from then on.
    """
    def fmt(value, depth):
        pad, pad_in = " " * (indent * depth), " " * (indent * (depth + 1))

        if isinstance(value, dict):
            if not value:
                return "{}"
            items = [f'{pad_in}{json.dumps(k)}: {fmt(v, depth + 1)}'
                     for k, v in value.items()]
            return "{\n" + ",\n".join(items) + "\n" + pad + "}"

        if isinstance(value, list):
            if not value:
                return "[]"
            if all(isinstance(v, (int, float, bool)) or v is None for v in value):
                oneline = "[" + ", ".join(json.dumps(v) for v in value) + "]"
                if len(pad) + len(oneline) <= inline_width:
                    return oneline
            items = [pad_in + fmt(v, depth + 1) for v in value]
            return "[\n" + ",\n".join(items) + "\n" + pad + "]"

        return json.dumps(value)

    return fmt(cfg, 0)


def write_scenario(path, cfg, backup=True):
    """
    Write a scenario file without ever leaving a half-written one behind.

    Every script loads its config at import, so a truncated scenario.json breaks
    all of them at once. Writing to a temp file in the same directory and then
    replacing is atomic on Windows as well as POSIX, so a reader sees either the
    old file or the new one.
    """
    path = os.path.abspath(path)
    if backup and os.path.exists(path):
        previous = os.path.splitext(path)[0] + ".previous.json"
        if not os.path.exists(previous):
            import shutil
            shutil.copy2(path, previous)

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(dumps_scenario(cfg))
    os.replace(tmp, path)
    return path


def writable_path(path):
    """
    Return `path`, or a timestamped sibling if `path` cannot be written.

    On Windows a file open in Excel cannot be replaced, and the analysis scripts
    can spend minutes computing a result before they find that out. Throwing the
    work away because a spreadsheet is open is a worse outcome than writing it
    next door under a slightly different name and saying so.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8"):
            pass
        return path
    except OSError:
        pass

    stem, ext = os.path.splitext(path)
    from datetime import datetime
    alternative = f"{stem}_{datetime.now():%Y%m%d-%H%M%S}{ext}"
    print(f"\n  Could not write {path}\n"
          f"  (it is probably open in another program -- Excel is the usual one)\n"
          f"  Writing to {os.path.basename(alternative)} instead so the result "
          f"is not lost.")
    return alternative


ROAD_SPEEDS_FILENAME = "road_speeds.json"
ROAD_SPEEDS_ENV_VAR = "MIM_ROAD_SPEEDS"
ROAD_SPEEDS_KEY = "road_speed_km-h"


def data_file_candidates(scenario, filename):
    """
    The places a shared, calibrated data file is looked for, in order:
    the scenario folder first (a per-scenario override), then the Model folder
    (shared across scenarios over the same city), then Simulation/ (where these
    files sit today). Exposed so a script can say what it searched.
    """
    return [
        os.path.join(scenario.folder, filename),
        os.path.join(MODEL_DIR, filename),
        os.path.join(MODEL_DIR, "Simulation", filename),
    ]


def road_speeds_candidates(scenario):
    """The places load_road_speeds looks, in order."""
    return data_file_candidates(scenario, ROAD_SPEEDS_FILENAME)


def load_road_speeds(scenario, path=None, export=False):
    """
    Return ``(speeds_km_h, source)`` -- the highway-type speed table, and a
    human-readable note saying where it came from.

    Speeds used to live in the scenario file under "road_speed_km-h". They are
    calibrated output, not scenario description: the same city re-calibrated
    gives a different table, and every scenario over the same city should share
    it. So they now live in their own file, looked up in this order:

        1. ``path``, or the MIM_ROAD_SPEEDS environment variable
        2. <scenario folder>/road_speeds.json      -- per-scenario override
        3. <Model>/road_speeds.json                -- shared across scenarios
        4. <Model>/Simulation/road_speeds.json     -- where the file sits today
        5. the scenario file's own "road_speed_km-h"   (legacy fallback)

    Step 5 exists so scenarios that were never migrated keep running, but it is
    announced rather than silent: routing weights that differ between two
    scripts produce results that look plausible and are wrong, which is a much
    more expensive failure than a noisy line of output.

    The file may be either ``{"road_speed_km-h": {...}}`` (matching the scenario
    key it replaces) or a bare ``{highway: km_h}`` mapping.

    ``export=True`` publishes the resolved path in the environment, so workers
    spawned by multiprocessing resolve to the same table -- the same reason
    load_scenario takes the flag.
    """
    path = path or os.environ.get(ROAD_SPEEDS_ENV_VAR) or None
    if path:
        if not os.path.isabs(path):
            candidate = os.path.join(MODEL_DIR, path)
            path = candidate if os.path.exists(candidate) else os.path.abspath(path)
        if not os.path.exists(path):
            raise SystemExit(f"Road speeds file not found: {path}")
        found = path
    else:
        found = next((p for p in road_speeds_candidates(scenario)
                      if os.path.exists(p)), None)

    if found is None:
        speeds = scenario.cfg.get(ROAD_SPEEDS_KEY)
        if not speeds:
            looked = " | ".join(road_speeds_candidates(scenario))
            raise SystemExit(
                f"No road speeds found. Looked for {ROAD_SPEEDS_FILENAME} at: "
                f"{looked} -- and for the legacy key in {scenario.path}.")
        source = f"{scenario.path} (legacy key)"
        print(f"  road speeds: no {ROAD_SPEEDS_FILENAME} found; "
              f"falling back to {source}")
        return _validate_road_speeds(speeds, source), source

    with open(found, "r", encoding="utf-8") as f:
        data = json.load(f)
    speeds = data.get(ROAD_SPEEDS_KEY, data) if isinstance(data, dict) else data
    if not isinstance(speeds, dict) or not speeds:
        raise SystemExit(
            f"{found} holds no speed table. Expected either a "
            f"{ROAD_SPEEDS_KEY!r} object or a bare highway-to-km/h mapping.")

    if export:
        os.environ[ROAD_SPEEDS_ENV_VAR] = found
    return _validate_road_speeds(speeds, found), found


def _validate_road_speeds(speeds, source):
    """A non-numeric or negative speed becomes a nonsensical edge weight rather
    than an error, so it is caught here where the file name is still in hand."""
    bad = {k: v for k, v in speeds.items()
           if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0}
    if bad:
        raise SystemExit(f"{source} has invalid speeds (need a number >= 0): {bad}")
    return {str(k): float(v) for k, v in speeds.items()}


def road_speeds_ms(speeds_km_h):
    """km/h table -> m/s table, the form the routing code actually wants."""
    return {k: v / 3.6 for k, v in speeds_km_h.items()}


# ============================================================
# SIMULATION MODE
# ============================================================
# The simulation used to have two modes selected by a boolean, "demand_model".
# There are now three, so the mode is named rather than implied. The boolean is
# still honoured when "simulation_mode" is absent, so scenario files that were
# never migrated keep selecting the mode they always did.
HAIL_RANK, DEMAND_MODEL_MODE, DISTRIBUTION = "hail_rank", "demand_model", "distribution"
SIMULATION_MODES = (HAIL_RANK, DEMAND_MODEL_MODE, DISTRIBUTION)
SIMULATION_MODE_KEY = "simulation_mode"

SIMULATION_MODE_HELP = {
    HAIL_RANK: "Agents alternate between hailing on the road or waiting at a "
               "taxi rank, and carrying a passenger to a nearby destination.",
    DEMAND_MODEL_MODE: "Trips are generated from demand points and frequencies, "
                       "then allocated to the nearest idle agent.",
    DISTRIBUTION: "Agents work a fare-wait / pickup / passenger cycle, with trip "
                  "distances and fare waits drawn from measured distributions.",
}


def simulation_mode(scenario):
    """
    Which of SIMULATION_MODES a scenario selects.

    Accepts a Scenario or a plain config dict. An unrecognised value is an error
    rather than a silent fall back to the default: a typo in the mode name would
    otherwise run a completely different simulation without saying so.
    """
    cfg = getattr(scenario, "cfg", scenario)
    mode = cfg.get(SIMULATION_MODE_KEY)
    if mode is None:
        return DEMAND_MODEL_MODE if cfg.get("demand_model", False) else HAIL_RANK
    mode = str(mode).strip().lower()
    if mode not in SIMULATION_MODES:
        raise SystemExit(
            f"Unknown {SIMULATION_MODE_KEY} {mode!r}. "
            f"Expected one of: {', '.join(SIMULATION_MODES)}.")
    return mode


# ============================================================
# TRIP DISTRIBUTIONS  (distribution mode)
# ============================================================
TRIP_DISTRIBUTIONS_FILENAME = "trip_distributions.json"
TRIP_DISTRIBUTIONS_ENV_VAR = "MIM_TRIP_DISTRIBUTIONS"

# Multipliers into the base unit each distribution is normalised to: metres for
# distances, seconds for waits. Units are required in the file rather than
# assumed, because the plausible alternatives differ by a factor of 60 or 1000
# and a wrong guess produces a run that looks fine and is nonsense.
_DISTANCE_UNITS = {"m": 1.0, "metre": 1.0, "metres": 1.0, "meter": 1.0,
                   "meters": 1.0, "km": 1000.0, "kilometre": 1000.0,
                   "kilometres": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0}
_TIME_UNITS = {"s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
               "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
               "h": 3600.0, "hr": 3600.0, "hour": 3600.0, "hours": 3600.0}


class Bands(object):
    """
    A histogram to sample from: bin edges in base units, plus the cumulative
    weights a draw indexes into.

    Kept as parallel lists rather than a list of dicts because it is read inside
    the simulation's per-trip decision, which runs for every idle agent on every
    timestep in every pool worker.
    """
    __slots__ = ("lows", "highs", "cum", "units", "source")

    def __init__(self, lows, highs, cum, units, source):
        self.lows, self.highs, self.cum = lows, highs, cum
        self.units, self.source = units, source

    def __len__(self):
        return len(self.lows)

    def mean(self):
        """Weight-weighted mean of the bin midpoints, in base units."""
        prev, acc = 0.0, 0.0
        for lo, hi, c in zip(self.lows, self.highs, self.cum):
            acc += (c - prev) * (lo + hi) / 2.0
            prev = c
        return acc / self.cum[-1]

    def describe(self, scale=1.0, unit=""):
        return (f"{len(self.lows)} bins, "
                f"{self.lows[0] / scale:g}-{self.highs[-1] / scale:g}{unit}, "
                f"mean {self.mean() / scale:.2f}{unit}")


def _parse_bands(spec, name, unit_table, source):
    """Validate one distribution and normalise its bins into base units."""
    if not isinstance(spec, dict):
        raise SystemExit(f"{source}: {name!r} must be an object with "
                         f"'units' and 'bins'.")
    units = str(spec.get("units", "")).strip().lower()
    if units not in unit_table:
        raise SystemExit(
            f"{source}: {name!r} needs a 'units' field, one of: "
            f"{', '.join(sorted(unit_table))}. Got {spec.get('units')!r}. "
            f"It is required rather than assumed because the alternatives "
            f"differ by a large factor and the wrong one would run silently.")
    scale = unit_table[units]

    bins = spec.get("bins")
    if not isinstance(bins, list) or not bins:
        raise SystemExit(f"{source}: {name!r} has no 'bins' list.")

    lows, highs, cum, running = [], [], [], 0.0
    for i, b in enumerate(bins):
        if not isinstance(b, dict):
            raise SystemExit(f"{source}: {name!r} bin {i} is not an object.")
        try:
            lo, hi = float(b["min"]), float(b["max"])
        except (KeyError, TypeError, ValueError):
            raise SystemExit(f"{source}: {name!r} bin {i} needs numeric "
                             f"'min' and 'max'. Got {b!r}.")
        # "count" is accepted so a histogram can be pasted in unchanged.
        weight = b.get("weight", b.get("count", b.get("probability")))
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            raise SystemExit(f"{source}: {name!r} bin {i} needs a numeric "
                             f"'weight' (or 'count'/'probability'). Got {b!r}.")
        if lo < 0 or hi < lo:
            raise SystemExit(f"{source}: {name!r} bin {i} has min={lo}, "
                             f"max={hi}; need 0 <= min <= max.")
        if weight < 0:
            raise SystemExit(f"{source}: {name!r} bin {i} has a negative weight.")
        if weight == 0:
            continue                       # a never-drawn bin, harmlessly dropped
        running += weight
        lows.append(lo * scale)
        highs.append(hi * scale)
        cum.append(running)

    if not lows:
        raise SystemExit(f"{source}: every bin of {name!r} has zero weight, "
                         f"so nothing could ever be drawn from it.")
    return Bands(lows, highs, cum, units, source)


def load_trip_distributions(scenario, path=None, export=False):
    """
    Load the distributions distribution-mode trips are drawn from, returning
    ``(bands_by_name, source)`` where bands_by_name has keys "distance" and
    "wait", holding Bands normalised to metres and seconds respectively.

    Looked up the same way as road speeds -- an explicit path or
    MIM_TRIP_DISTRIBUTIONS, then the scenario folder, the Model folder,
    Simulation/ -- because it is the same kind of thing: measured output that
    several scenarios over one city should share, not a description of any one
    of them.

    Expected shape (see trip_distributions.json for a worked example):

        {"distance_distribution": {"units": "km",
                                   "bins": [{"min": 0, "max": 1, "weight": 12},
                                            ...]},
         "wait_distribution":     {"units": "min",
                                   "bins": [{"min": 0, "max": 2, "weight": 30},
                                            ...]}}
    """
    path = path or os.environ.get(TRIP_DISTRIBUTIONS_ENV_VAR) or None
    if path:
        if not os.path.isabs(path):
            candidate = os.path.join(MODEL_DIR, path)
            path = candidate if os.path.exists(candidate) else os.path.abspath(path)
        if not os.path.exists(path):
            raise SystemExit(f"Trip distributions file not found: {path}")
        found = path
    else:
        candidates = data_file_candidates(scenario, TRIP_DISTRIBUTIONS_FILENAME)
        found = next((c for c in candidates if os.path.exists(c)), None)
        if found is None:
            raise SystemExit(
                f"{SIMULATION_MODE_KEY} is {DISTRIBUTION!r}, which draws trip "
                f"distances and fare waits from {TRIP_DISTRIBUTIONS_FILENAME}, "
                f"but no such file was found. Looked at: "
                + " | ".join(candidates))

    with open(found, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"{found} is not a JSON object.")

    bands = {
        "distance": _parse_bands(data.get("distance_distribution"),
                                 "distance_distribution", _DISTANCE_UNITS, found),
        "wait": _parse_bands(data.get("wait_distribution"),
                             "wait_distribution", _TIME_UNITS, found),
    }
    if export:
        os.environ[TRIP_DISTRIBUTIONS_ENV_VAR] = found
    return bands, found


def add_scenario_argument(parser):
    """Give a script the standard --scenario option."""
    parser.add_argument(
        "--scenario", default=None, metavar="FILE",
        help="scenario file to use (default: scenario.json in the Model folder)")
    return parser


def scenario_from_cli(description=None):
    """
    For scripts that take no other arguments: parse just --scenario and return
    the loaded Scenario.
    """
    parser = argparse.ArgumentParser(description=description)
    add_scenario_argument(parser)
    return load_scenario(parser.parse_args().scenario)
