"""
The FastAPI app.

Two jobs:
  1. Serve the existing HTML tools over the same URL space Live Server does, so
     animation.html and friends work unchanged under either server.
  2. Expose the scenarios, and run the scripts, over a small JSON API.

The static mounts are the important part for compatibility. Live Server's root
is the Model folder with /Model_data mounted onto the sibling data directory
(see .vscode/settings.json); this reproduces exactly that, deriving the mount
from where the scenarios actually resolve rather than hard-coding it.
"""

import glob
import json
import os
import re
import shutil
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import MODEL_DIR, load_scenario, write_scenario

from . import tasks as task_registry
from .jobs import JobManager

app = FastAPI(title="EV-TRACS model server", docs_url="/api/docs")
manager = JobManager(cwd=MODEL_DIR)


# ============================================================
# SCENARIOS
# ============================================================

def scenario_files():
    return sorted(glob.glob(os.path.join(MODEL_DIR, "scenario*.json")))


def scenario_summary(path):
    name = os.path.basename(path)
    try:
        s = load_scenario(path)
    except SystemExit as e:
        return {"file": name, "ok": False, "error": str(e)}
    return {
        "file": name,
        "ok": True,
        "active": name == "scenario.json",
        "name": s.name,
        "folder_name": s.folder_name,
        "folder": s.folder,
        "folder_exists": os.path.isdir(s.folder),
        "simulation_mode": task_registry.scenario_mode(s),
        # Kept so anything still reading the boolean sees the migrated value
        # rather than a missing key.
        "demand_model": task_registry.scenario_mode(s) == "demand_model",
        "data_url": s.data_url,
    }


def _newest_mtime(path):
    """
    Modification time of a file, or of the newest file inside a directory.

    A directory's own mtime only tracks entries being added or removed, so a run
    that rewrites the same 700 agent files in place would leave it unchanged and
    the artefact would look older than it is.
    """
    if os.path.isfile(path):
        return os.path.getmtime(path)
    newest, count = 0.0, 0
    with os.scandir(path) as entries:
        for entry in entries:
            count += 1
            if entry.is_file():
                newest = max(newest, entry.stat().st_mtime)
    return newest or os.path.getmtime(path)


def artefact_status(scenario):
    """
    Existence, size and staleness for every artefact in the pipeline.

    Stale means: it exists, but something it was derived from is newer. That is
    the whole point of the board -- an out-of-date result looks exactly like an
    up-to-date one on disk.
    """
    # Which inputs count depends on the mode: taxi ranks only in hail_rank,
    # demand points and frequencies only in demand_model, trip distributions
    # only in distribution. Deciding this per scenario is what lets the board say
    # "missing" about something that will actually break the run, rather than
    # shrugging at everything.
    mode = task_registry.scenario_mode(scenario)

    found = {}
    for art in task_registry.ARTEFACTS:
        path = os.path.join(scenario.folder, art.path.replace("/", os.sep))
        exists = os.path.exists(path)
        required = art.is_required(mode)
        entry = {
            "key": art.key, "label": art.label, "stage": art.stage,
            "path": art.path, "exists": exists, "optional": not required,
            "produced_by": art.produced_by,
            "depends_on": list(art.depends_on),
        }
        if exists:
            entry["modified"] = int(_newest_mtime(path))
            if os.path.isdir(path):
                entry["count"] = sum(1 for _ in os.scandir(path))
            else:
                entry["size"] = os.path.getsize(path)
        found[art.key] = entry

    for art in task_registry.ARTEFACTS:
        entry = found[art.key]
        if not art.is_required(mode):
            # Present or not, this mode does not read it. Saying "not used" is
            # more honest than "ok", which would imply it mattered.
            entry["status"] = "optional"
            entry["stale_because"] = []
            continue
        if not entry["exists"]:
            entry["status"] = "missing"
            continue
        newer = [found[d]["label"] for d in art.depends_on
                 if found.get(d, {}).get("exists")
                 and found[d]["modified"] > entry["modified"]]
        entry["status"] = "stale" if newer else "ok"
        entry["stale_because"] = newer

    return [found[a.key] for a in task_registry.ARTEFACTS]


@app.get("/api/scenarios")
def list_scenarios():
    return {"scenarios": [scenario_summary(p) for p in scenario_files()]}


@app.get("/api/scenarios/{name}")
def get_scenario(name: str):
    path = _scenario_path(name)
    s = load_scenario(path)
    return {
        **scenario_summary(path),
        "config": s.cfg,
        # The client sends this back on save. If the file changed underneath --
        # you have scenario.json open in VS Code -- the save is refused rather
        # than silently discarding whichever edit landed first.
        "mtime": int(os.path.getmtime(path)),
        "paths": {
            "input_dir": s.input_dir,
            "output_dir": s.output_dir,
            "captured_dir": s.captured_dir,
            "trips_time_dir": s.trips_time_dir,
        },
        "artefacts": artefact_status(s),
        "stages": task_registry.STAGES,
    }


@app.put("/api/scenarios/{name}")
def save_scenario(name: str, payload: dict):
    """
    Merge changes into a scenario file.

    Merged, not replaced. A form built from a known set of fields would silently
    drop any key it does not know about -- deviation_factor exists in only one
    of the four scenario files, and losing it would change how the simulation
    samples passenger destinations with nothing to show for it. Keys absent from
    the request keep their current value.
    """
    path = _scenario_path(name)
    changes = payload.get("config")
    if not isinstance(changes, dict):
        raise HTTPException(400, "config must be an object")

    current_mtime = int(os.path.getmtime(path))
    sent_mtime = payload.get("mtime")
    if sent_mtime is not None and int(sent_mtime) != current_mtime:
        raise HTTPException(
            409,
            "The file changed on disk since you loaded it (it may be open in "
            "your editor). Reload before saving so neither edit is lost.")

    with open(path, "r", encoding="utf-8") as f:
        merged = json.load(f)
    merged.update(changes)

    if "folder_name" not in merged or not str(merged["folder_name"]).strip():
        raise HTTPException(400, "folder_name cannot be empty")
    try:
        json.dumps(merged)
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"config is not serialisable: {e}")

    write_scenario(path, merged)
    s = load_scenario(path)
    return {
        "saved": name,
        "mtime": int(os.path.getmtime(path)),
        "config": s.cfg,
        "folder": s.folder,
        "folder_exists": os.path.isdir(s.folder),
        "artefacts": artefact_status(s),
    }


def _scenario_path(name):
    """Resolve a scenario filename, refusing anything outside the Model folder."""
    if os.path.basename(name) != name:
        raise HTTPException(400, "scenario must be a bare filename")
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(404, f"no such scenario: {name}")
    return path


# ============================================================
# TASKS & JOBS
# ============================================================

@app.get("/api/tasks")
def list_tasks():
    return {"tasks": [{
        "id": t.id, "label": t.label, "script": t.script, "stage": t.stage,
        "safety": t.safety, "description": t.description,
        "exists": t.exists, "runnable": task_registry.is_runnable(t),
        "needs_confirmation": task_registry.needs_confirmation(t),
        "irreversible": task_registry.is_irreversible(t),
    } for t in task_registry.TASKS]}


@app.get("/api/tasks/{task_id}/impact")
def task_impact(task_id: str, scenario: str = "scenario.json"):
    """
    What running this task would overwrite, with the files as they stand now.

    A warning that names the actual files and their sizes is worth far more
    than a generic "are you sure" -- it is the difference between reading the
    dialog and clicking through it.
    """
    task = task_registry.TASKS_BY_ID.get(task_id)
    if task is None:
        raise HTTPException(404, f"no such task: {task_id}")
    s = load_scenario(_scenario_path(scenario))

    files = []
    for rel in task_registry.OVERWRITES.get(task_id, []):
        path = os.path.join(s.folder, rel.replace("/", os.sep))
        exists = os.path.exists(path)
        files.append({
            "path": rel,
            "exists": exists,
            "size_mb": round(os.path.getsize(path) / 1e6, 1) if exists else None,
            "modified": int(os.path.getmtime(path)) if exists else None,
        })
    return {
        "task": task_id, "label": task.label, "safety": task.safety,
        "irreversible": task_registry.is_irreversible(task),
        "scenario": scenario, "folder": s.folder, "overwrites": files,
    }


# ============================================================
# COPY A SCENARIO
# ============================================================

# Copied by default: the inputs a scenario is defined by. Results are not --
# they are reproducible by running the model, and an accra output folder is
# 433 MB that would then sit in OneDrive twice.
COPY_ALWAYS = ["geojson_files", "captured_locations"]
COPY_ON_REQUEST = ["output"]


def _slug(name):
    """A scenario filename stem: lowercase, spaces and punctuation to _."""
    s = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return s or "copy"


def _dir_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


@app.post("/api/scenarios/{name}/copy")
def copy_scenario(name: str, payload: dict):
    """
    Duplicate a scenario: its folder, and a scenario file pointing at the copy.

    The new scenario file is a copy of the original with folder_name repointed,
    so every other setting -- fleet profile, road speeds, the keys this UI does
    not know about -- carries over unchanged.
    """
    src_path = _scenario_path(name)
    src = load_scenario(src_path)

    new_name = str(payload.get("new_name", "")).strip()
    if not new_name:
        raise HTTPException(400, "A name is required")
    if re.search(r'[\\/:*?"<>|]', new_name):
        raise HTTPException(400, 'A name cannot contain \\ / : * ? " < > |')

    include_output = bool(payload.get("include_output"))

    # Scenario file beside the original; folder beside the original's folder.
    scenario_file = payload.get("scenario_file") or f"scenario_{_slug(new_name)}.json"
    if os.path.basename(scenario_file) != scenario_file:
        raise HTTPException(400, "scenario_file must be a bare filename")
    if not scenario_file.endswith(".json"):
        scenario_file += ".json"
    dest_scenario = os.path.join(MODEL_DIR, scenario_file)

    dest_folder = os.path.join(os.path.dirname(src.folder), new_name)

    if os.path.exists(dest_scenario):
        raise HTTPException(409, f"{scenario_file} already exists")
    if os.path.exists(dest_folder):
        raise HTTPException(409, f"Folder already exists: {dest_folder}")
    if not os.path.isdir(src.folder):
        raise HTTPException(404, f"Source folder does not exist: {src.folder}")

    # Confined to the same root the folder browser uses.
    root = os.path.abspath(BROWSE_ROOT)
    if not os.path.normcase(os.path.abspath(dest_folder)).startswith(
            os.path.normcase(root) + os.sep):
        raise HTTPException(403, "Destination is outside the browsable root")

    wanted = COPY_ALWAYS + (COPY_ON_REQUEST if include_output else [])
    copied, skipped = [], []
    os.makedirs(dest_folder, exist_ok=False)
    try:
        for entry in os.scandir(src.folder):
            if entry.is_dir():
                if entry.name in wanted:
                    shutil.copytree(entry.path, os.path.join(dest_folder, entry.name))
                    copied.append(entry.name)
                else:
                    skipped.append(entry.name)
            else:
                # Loose files alongside the folders are cheap and usually notes.
                shutil.copy2(entry.path, os.path.join(dest_folder, entry.name))
                copied.append(entry.name)
    except OSError as e:
        # Do not leave a half-copied folder behind for someone to trip over.
        shutil.rmtree(dest_folder, ignore_errors=True)
        raise HTTPException(500, f"Copy failed, nothing kept: {e}")

    cfg = dict(src.cfg)
    cfg["folder_name"] = _as_relative(dest_folder)
    write_scenario(dest_scenario, cfg, backup=False)

    return {
        "scenario_file": scenario_file,
        "folder": dest_folder,
        "folder_name": cfg["folder_name"],
        "copied": copied,
        "skipped": skipped,
        "size_mb": round(_dir_size(dest_folder) / 1e6, 1),
    }


# ============================================================
# FOLDER BROWSER
# ============================================================
# A browser cannot hand a page a filesystem path -- neither <input
# webkitdirectory> nor showDirectoryPicker() exposes one, by design -- so the
# folder picker is served from here instead. Browsing is confined to
# BROWSE_ROOT: the server is loopback-only, but an endpoint that walks the whole
# disk is still not something to leave lying around.
BROWSE_ROOT = os.path.dirname(MODEL_DIR)

# Directories that are never a scenario folder and only add noise.
BROWSE_SKIP = {"venv", "__pycache__", "node_modules", ".git", ".vscode", "cache",
               "staticfiles", "site-packages"}


def _as_relative(abs_path):
    """Path as folder_name wants it: relative to Model/, forward slashes."""
    rel = os.path.relpath(abs_path, MODEL_DIR)
    return rel.replace(os.sep, "/")


def _resolve_browse(rel):
    """Absolute path for a browse request, refusing anything outside the root."""
    target = os.path.abspath(os.path.join(MODEL_DIR, rel or "."))
    root = os.path.abspath(BROWSE_ROOT)
    if os.path.normcase(target) != os.path.normcase(root) and \
       not os.path.normcase(target).startswith(os.path.normcase(root) + os.sep):
        raise HTTPException(403, "Outside the browsable root")
    if not os.path.isdir(target):
        raise HTTPException(404, f"Not a directory: {target}")
    return target


@app.get("/api/browse")
def browse(path: str = ""):
    """
    Directories under `path`, for the folder picker.

    Entries are flagged as scenario folders when they contain geojson_files,
    so the one you want is obvious rather than something to recognise by name.
    """
    target = _resolve_browse(path)
    root = os.path.abspath(BROWSE_ROOT)

    entries = []
    try:
        for entry in sorted(os.scandir(target), key=lambda e: e.name.lower()):
            if not entry.is_dir() or entry.name.startswith(".") \
               or entry.name in BROWSE_SKIP:
                continue
            has_inputs = os.path.isdir(os.path.join(entry.path, "geojson_files"))
            entries.append({
                "name": entry.name,
                "path": _as_relative(entry.path),
                "is_scenario": has_inputs,
                "has_output": os.path.isdir(os.path.join(entry.path, "output")),
            })
    except PermissionError:
        raise HTTPException(403, f"Cannot read {target}")

    at_root = os.path.normcase(target) == os.path.normcase(root)
    return {
        "path": _as_relative(target),
        "abs": target,
        "name": os.path.basename(target) or target,
        "parent": None if at_root else _as_relative(os.path.dirname(target)),
        "is_scenario": os.path.isdir(os.path.join(target, "geojson_files")),
        "entries": entries,
        "root": _as_relative(root),
    }


@app.get("/api/tools")
def list_tools():
    """
    The HTML tools, with URLs ready to link to.

    Built server-side so the path is quoted properly -- "edit_demand
    points.html" has a space in it -- and so a tool that has been moved or
    renamed shows up as missing instead of as a link that 404s.
    """
    return {
        "groups": task_registry.TOOL_GROUPS,
        "tools": [{
            "label": t.label,
            "url": "/" + quote(t.path),
            "description": t.description,
            "group": t.group,
            "accepts_scenario": t.accepts_scenario,
            "exists": t.exists,
        } for t in task_registry.TOOLS],
    }


@app.post("/api/jobs")
def create_job(payload: dict):
    task_id = payload.get("task")
    task = task_registry.TASKS_BY_ID.get(task_id)
    if task is None:
        raise HTTPException(404, f"no such task: {task_id}")
    if not task.exists:
        raise HTTPException(404, f"script not found: {task.script}")
    if not task_registry.is_runnable(task):
        # Refused rather than hidden, so the reason is visible.
        raise HTTPException(
            403,
            f"'{task.label}' is classed {task.safety} and cannot be launched "
            "from the browser. It overwrites work with no way back; run it from "
            "a terminal, where you have to type the command deliberately.")
    if task_registry.needs_confirmation(task) and not payload.get("confirm"):
        # The UI asks first; this is the backstop for anything calling the API
        # directly, so a confirmation cannot be skipped by bypassing the page.
        raise HTTPException(
            428,
            f"'{task.label}' replaces the current results. Resend with "
            '{"confirm": true} to go ahead.')

    path = _scenario_path(payload.get("scenario") or "scenario.json")
    job = manager.submit(task, path)
    return job.to_dict()


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": manager.list()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: int):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job.to_dict()


@app.get("/api/jobs/{job_id}/log")
def get_job_log(job_id: int, offset: int = 0):
    """Log lines from `offset` onwards, plus the new cursor to poll with."""
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    lines, total = job.log_from(offset)
    return {"lines": lines, "offset": total,
            "status": job.status, "finished": job.is_finished,
            "duration_s": job.duration_s, "returncode": job.returncode}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: int):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return {"cancelled": job.cancel(), "status": job.status}


@app.get("/api/health")
def health():
    return {"ok": True, "python": manager.python, "model_dir": MODEL_DIR,
            "running_jobs": len(manager.running)}


# ============================================================
# STATIC -- must be mounted last so /api takes precedence
# ============================================================

def data_mounts():
    """
    Every directory the scenarios actually live in, keyed by the URL prefix it
    should be served under. Derived rather than hard-coded, so it keeps matching
    Live Server's mount if the data folder is renamed or moved again.
    """
    mounts = {}
    for path in scenario_files():
        try:
            folder = load_scenario(path).folder
        except SystemExit:
            continue
        parent = os.path.dirname(os.path.normpath(folder))
        if os.path.normcase(parent) != os.path.normcase(MODEL_DIR) and os.path.isdir(parent):
            mounts["/" + os.path.basename(parent)] = parent
    return mounts


def mount_static(application):
    for route, directory in data_mounts().items():
        application.mount(route, StaticFiles(directory=directory), name=route.strip("/"))

    web_dir = os.path.join(MODEL_DIR, "web")
    if os.path.isdir(web_dir):
        application.mount("/app", StaticFiles(directory=web_dir, html=True), name="web")

    # The Model folder last: it answers everything not claimed above, which is
    # what makes animation.html and the utilities/ tools work unchanged.
    application.mount("/", StaticFiles(directory=MODEL_DIR, html=True), name="model")


mount_static(app)
