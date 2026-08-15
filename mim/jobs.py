"""
Run a script as a subprocess and collect its output so a browser can watch.

Threads rather than asyncio.subprocess: on Windows the proactor loop's pipe
handling is the fiddly part of an otherwise trivial problem, and a reader thread
per job is both simpler and more robust. The API polls by line index, so a
client that reconnects picks up exactly where it left off.
"""

import itertools
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

MAX_LOG_LINES = 5000

QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED = (
    "queued", "running", "succeeded", "failed", "cancelled")


class Job:
    def __init__(self, job_id, task, scenario_path, command):
        self.id = job_id
        self.task = task
        self.scenario_path = scenario_path
        self.command = command
        self.status = QUEUED
        self.returncode = None
        self.started_at = None
        self.finished_at = None
        self.lines = []
        self.error = None
        self._process = None
        self._lock = threading.Lock()

    # -- log -------------------------------------------------------------
    def append(self, text):
        with self._lock:
            self.lines.append(text)
            if len(self.lines) > MAX_LOG_LINES:
                # Keep the tail; a truncation marker keeps the log honest about
                # what it dropped rather than silently losing the start.
                dropped = len(self.lines) - MAX_LOG_LINES
                self.lines = ([f"... {dropped} earlier lines truncated ..."]
                              + self.lines[-MAX_LOG_LINES:])

    def log_from(self, index):
        with self._lock:
            index = max(0, min(index, len(self.lines)))
            return self.lines[index:], len(self.lines)

    # -- state -----------------------------------------------------------
    @property
    def duration_s(self):
        if not self.started_at:
            return None
        end = self.finished_at or time.time()
        return round(end - self.started_at, 1)

    @property
    def is_finished(self):
        return self.status in (SUCCEEDED, FAILED, CANCELLED)

    def to_dict(self, include_log_length=True):
        d = {
            "id": self.id,
            "task": self.task.id,
            "task_label": self.task.label,
            "scenario": os.path.basename(self.scenario_path),
            "status": self.status,
            "returncode": self.returncode,
            "duration_s": self.duration_s,
            "started_at": (datetime.fromtimestamp(self.started_at).isoformat(timespec="seconds")
                           if self.started_at else None),
            "command": " ".join(self.command),
            "error": self.error,
        }
        if include_log_length:
            with self._lock:
                d["log_length"] = len(self.lines)
        return d

    def cancel(self):
        proc = self._process
        if proc and proc.poll() is None:
            proc.terminate()
            self.status = CANCELLED
            self.append("--- cancelled ---")
            return True
        return False


def _stream_output(process, job):
    """
    Forward the child's output line by line.

    Split on \\r as well as \\n: tqdm redraws a progress bar by returning to the
    start of the line, so reading only on \\n would buffer an entire progress bar
    into one enormous line and show nothing until the job ended.
    """
    buffer = ""
    while True:
        chunk = process.stdout.read(1)
        if not chunk:
            break
        if chunk in ("\r", "\n"):
            if buffer.strip():
                job.append(buffer.rstrip())
            buffer = ""
        else:
            buffer += chunk
    if buffer.strip():
        job.append(buffer.rstrip())


class JobManager:
    def __init__(self, python=None, cwd=None):
        # sys.executable, so the subprocess runs in whatever interpreter the
        # server itself is running in -- the venv, if it was started from there.
        self.python = python or sys.executable
        self.cwd = cwd
        self.jobs = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def submit(self, task, scenario_path):
        with self._lock:
            job_id = next(self._ids)
        command = task.command(self.python, scenario_path)
        job = Job(job_id, task, scenario_path, command)
        self.jobs[job_id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def _run(self, job):
        job.status = RUNNING
        job.started_at = time.time()
        job.append(f"$ {' '.join(job.command)}")

        env = dict(os.environ)
        # Unbuffered so the log arrives while the job runs rather than at the
        # end, and UTF-8 so the scripts' arrows and tick marks survive the trip.
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        try:
            job._process = subprocess.Popen(
                job.command, cwd=self.cwd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=0)
        except OSError as e:
            job.status = FAILED
            job.error = str(e)
            job.append(f"failed to start: {e}")
            job.finished_at = time.time()
            return

        _stream_output(job._process, job)
        job._process.wait()
        job.returncode = job._process.returncode
        job.finished_at = time.time()

        if job.status != CANCELLED:
            job.status = SUCCEEDED if job.returncode == 0 else FAILED
        job.append(f"--- {job.status} (exit {job.returncode}) "
                   f"in {job.duration_s}s ---")

    def get(self, job_id):
        return self.jobs.get(job_id)

    def list(self):
        return [j.to_dict() for j in sorted(self.jobs.values(),
                                            key=lambda j: j.id, reverse=True)]

    @property
    def running(self):
        return [j for j in self.jobs.values() if j.status == RUNNING]
