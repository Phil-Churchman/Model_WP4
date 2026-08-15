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
from scenario_config import MODEL_DIR, load_scenario

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


def artefact_status(scenario):
    """Existence and mtime for each pipeline artefact, for the pipeline view."""
    out = []
    for label, rel, stage in task_registry.ARTEFACTS:
        path = os.path.join(scenario.folder, rel.replace("/", os.sep))
        exists = os.path.exists(path)
        entry = {"label": label, "stage": stage, "path": rel, "exists": exists}
        if exists:
            entry["modified"] = int(os.path.getmtime(path))
            if os.path.isdir(path):
                entry["count"] = len(os.listdir(path))
            else:
                entry["size"] = os.path.getsize(path)
        out.append(entry)
    return out


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
        "paths": {
            "input_dir": s.input_dir,
            "output_dir": s.output_dir,
            "captured_dir": s.captured_dir,
            "trips_time_dir": s.trips_time_dir,
        },
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
