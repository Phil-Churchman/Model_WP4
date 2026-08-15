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
