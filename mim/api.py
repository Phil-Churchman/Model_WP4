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

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import MODEL_DIR, load_scenario, write_scenario

from . import tasks as task_registry
from .jobs import JobManager

app = FastAPI(title="Moving Impact model server", docs_url="/api/docs")
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
        "demand_model": s.get("demand_model", False),
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
    found = {}
    for art in task_registry.ARTEFACTS:
        path = os.path.join(scenario.folder, art.path.replace("/", os.sep))
        exists = os.path.exists(path)
        entry = {
            "key": art.key, "label": art.label, "stage": art.stage,
            "path": art.path, "exists": exists, "optional": art.optional,
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
        if not entry["exists"]:
            entry["status"] = "optional" if art.optional else "missing"
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
    } for t in task_registry.TASKS]}


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
            "from the browser yet. It overwrites or deletes work that cannot be "
            "recovered; run it from a terminal, or wait for run isolation.")

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
