"""
Geo-logger upload agent.

Replays the simulation trips in <folder_name>/output/output_trips_time_queued as if
each simulated vehicle were carrying a GPS logger that posts its track to the
Geo_logger API (loggerapp.views.receive_location, routed at /api/upload/).

Each simulated vehicle:
  * captures a position every POINT_SPACING_M metres of travel (and at least every
    MAX_POINT_INTERVAL_SEC seconds, plus every STATIONARY_INTERVAL_SEC while parked,
    hailing or queueing at a swap station),
  * buffers those points on the "handset" until it has at least
    MIN_POINTS_PER_UPLOAD of them and the mobile network is available,
  * suffers an unreliable network driven by NETWORK_UNRELIABILITY and
    MAX_NETWORK_AVAILABLE_SEC - points simply pile up in the buffer until the
    connection returns,
  * uses a device id built from a shared run element plus a per-device element,
    e.g. sim-20260810-1432-a0007.

Every API request and its response is recorded and dumped to a single CSV log,
rewritten (overwriting the previous file) every LOG_INTERVAL_SEC of wall-clock time.

Credentials are read from the environment (the API uses SimpleJWT):
    set GEOLOGGER_USERNAME=...
    set GEOLOGGER_PASSWORD=...
Run with --dry-run to exercise the whole pipeline without touching the server.
"""

import argparse
import csv
import json
import math
import os
import queue
import random
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import load_scenario, add_scenario_argument

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

API_BASE_URL = "https://geologger.pathplotter.net"
UPLOAD_URL = API_BASE_URL + "/api/upload/"
TOKEN_URL = API_BASE_URL + "/api/token/"
TOKEN_REFRESH_URL = API_BASE_URL + "/api/token/refresh/"

# --- point capture -------------------------------------------------------------------
POINT_SPACING_M = 25.0            # distance travelled between captured points
MAX_POINT_INTERVAL_SEC = 60.0     # also capture at least this often while moving
STATIONARY_INTERVAL_SEC = 30.0    # capture cadence while the vehicle is not moving
GPS_NOISE_M = 3.0                 # random positional error added to each point
GPS_ACCURACY_RANGE_M = (4.0, 15.0)  # reported "accuracy" value

# --- upload batching -----------------------------------------------------------------
MIN_POINTS_PER_UPLOAD = 20        # minimum number of points sent per API call
MAX_POINTS_PER_UPLOAD = 500       # cap on a single API call (buffer catch-up)
MAX_BUFFERED_POINTS = 20000       # handset storage limit; oldest points are dropped

# --- network reliability -------------------------------------------------------------
# NETWORK_UNRELIABILITY is the long-run fraction of time a device has no connection.
# Online windows are drawn uniformly from
# [MIN_NETWORK_AVAILABLE_SEC, MAX_NETWORK_AVAILABLE_SEC]; the mean outage length is
# derived from those so the offline fraction matches NETWORK_UNRELIABILITY.
NETWORK_UNRELIABILITY = 0.30
MIN_NETWORK_AVAILABLE_SEC = 60.0
MAX_NETWORK_AVAILABLE_SEC = 900.0

# --- run / device identity -----------------------------------------------------------
RUN_ID = None                     # None -> "sim-YYYYmmdd-HHMM" generated at start
DEVICE_ID_TEMPLATE = "{run_id}-a{agent:04d}"

# --- replay pacing -------------------------------------------------------------------
TIME_SCALE = 1                # simulated seconds per real second (1.0 = real time)
TICK_SEC = 10.0                   # simulated seconds per loop iteration
MAX_FLUSH_EXTRA_SEC = 7200.0      # keep running this long past the last point to drain
SIM_UTC_OFFSET_HOURS = 0.0        # simulation clock -> UTC (Accra is UTC+0)
SKIP_IDLE_GAPS = True             # jump the clock over spells when no vehicle is active
MIN_IDLE_GAP_SEC = 60.0           # only jump gaps longer than this

# --- devices -------------------------------------------------------------------------
MAX_DEVICES = 25                  # None / --devices all for every agent file
DEVICE_STRIDE = 1                 # take every Nth agent file when sub-sampling

# --- logging -------------------------------------------------------------------------
LOG_INTERVAL_SEC = 600.0          # wall-clock seconds between CSV rewrites (10 min)
LOG_FILE_NAME = "geologger_api_log.csv"
RESPONSE_BODY_CHARS = 300         # response text kept per row

# --- transport -----------------------------------------------------------------------
UPLOAD_WORKERS = 8                # concurrent in-flight HTTP requests
REQUEST_TIMEOUT_SEC = 30.0
RANDOM_SEED = 42

EARTH_RADIUS_M = 6371008.8
NEVER = datetime.max.replace(tzinfo=timezone.utc)  # connectivity state that never ends

CSV_COLUMNS = [
    "wall_time",
    "sim_time",
    "device_id",
    "agent_id",
    "batch_seq",
    "attempt",
    "network_state",
    "points_in_batch",
    "buffer_before",
    "buffer_after",
    "first_point_time",
    "last_point_time",
    "request_bytes",
    "url",
    "http_status",
    "latency_ms",
    "response_count",
    "response_body",
    "error",
    "outcome",
]


# --------------------------------------------------------------------------------------
# Geometry / track sampling
# --------------------------------------------------------------------------------------

def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres between two lon/lat pairs."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def jitter(lon, lat, rng):
    """Add GPS_NOISE_M of random positional error."""
    if GPS_NOISE_M <= 0:
        return lon, lat
    bearing = rng.uniform(0, 2 * math.pi)
    offset = rng.gauss(0, GPS_NOISE_M / 2.0)
    dlat = (offset * math.cos(bearing)) / 111320.0
    dlon = (offset * math.sin(bearing)) / (111320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return lon + dlon, lat + dlat


def parse_sim_time(value):
    return datetime.fromisoformat(value)


def _interp(x, xs, ys):
    """Linear interpolation on a non-decreasing xs (small, pure-python np.interp)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo]
    if span <= 0:
        return ys[lo]
    f = (x - xs[lo]) / span
    return ys[lo] + f * (ys[hi] - ys[lo])


def linestring_points(feature, rng):
    """Yield (time, lon, lat) samples along a travelled leg."""
    coords = feature["geometry"]["coordinates"]
    props = feature["properties"]
    start = parse_sim_time(props["start_time"])
    duration = float(props.get("duration_s") or 0.0)

    if len(coords) < 2:
        if coords:
            yield start, coords[0][0], coords[0][1]
        return

    seg_times = props.get("segment_times") or []
    n_segments = len(coords) - 1
    if len(seg_times) != n_segments:
        # Fall back to spreading the leg duration evenly over the segments.
        even = (duration / n_segments) if n_segments else 0.0
        seg_times = [even] * n_segments

    cum_d, cum_t = [0.0], [0.0]
    for i in range(n_segments):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        cum_d.append(cum_d[-1] + haversine_m(lon1, lat1, lon2, lat2))
        cum_t.append(cum_t[-1] + float(seg_times[i]))

    total_d, total_t = cum_d[-1], cum_t[-1]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    # Sample by distance travelled, and by elapsed time so slow crawls still report.
    offsets = set()
    d = 0.0
    while d < total_d:
        offsets.add(round(_interp(d, cum_d, cum_t), 3))
        d += POINT_SPACING_M
    t = 0.0
    while t < total_t and MAX_POINT_INTERVAL_SEC > 0:
        offsets.add(round(t, 3))
        t += MAX_POINT_INTERVAL_SEC
    offsets.add(round(total_t, 3))

    for offset in sorted(offsets):
        dist = _interp(offset, cum_t, cum_d)
        lon = _interp(dist, cum_d, lons)
        lat = _interp(dist, cum_d, lats)
        yield start + timedelta(seconds=offset), lon, lat


def stationary_points(feature, rng):
    """Yield (time, lon, lat) samples while the vehicle is stopped."""
    lon, lat = feature["geometry"]["coordinates"]
    props = feature["properties"]
    start = parse_sim_time(props["start_time"])
    duration = float(props.get("duration_s") or 0.0)

    step = max(STATIONARY_INTERVAL_SEC, 1.0)
    offset = 0.0
    while offset < duration:
        yield start + timedelta(seconds=offset), lon, lat
        offset += step
    yield start + timedelta(seconds=duration), lon, lat


def iter_captured_points(file_path, rng):
    """Yield (utc_datetime, lat, lon, accuracy) for one agent's whole day, in time order."""
    with open(file_path, "r") as fh:
        features = json.load(fh).get("features", [])

    features.sort(key=lambda f: f["properties"].get("start_time", ""))
    tz_shift = timedelta(hours=SIM_UTC_OFFSET_HOURS)
    last_time = None

    for feature in features:
        geom_type = feature.get("geometry", {}).get("type")
        if geom_type == "LineString":
            sampler = linestring_points(feature, rng)
        elif geom_type == "Point":
            sampler = stationary_points(feature, rng)
        else:
            continue

        for sim_time, lon, lat in sampler:
            # Legs share their endpoints with the following stop - keep one point only.
            if last_time is not None and (sim_time - last_time).total_seconds() < 1.0:
                continue
            last_time = sim_time
            noisy_lon, noisy_lat = jitter(lon, lat, rng)
            accuracy = round(rng.uniform(*GPS_ACCURACY_RANGE_M), 1)
            yield (sim_time - tz_shift).replace(tzinfo=timezone.utc), noisy_lat, noisy_lon, accuracy


# --------------------------------------------------------------------------------------
# Simulated device
# --------------------------------------------------------------------------------------

class SimulatedDevice:
    """One simulated vehicle: capture buffer, flaky network, upload state."""

    def __init__(self, agent_id, device_id, file_path, start_time, end_time, rng):
        self.agent_id = agent_id
        self.device_id = device_id
        self.file_path = file_path
        self.start_time = start_time
        self.end_time = end_time
        self.rng = rng

        self.points = None            # lazily opened generator, keeps memory flat
        self.pending = None           # next point not yet due
        self.exhausted = False
        self.buffer = deque()

        self.online = rng.random() >= NETWORK_UNRELIABILITY
        self.next_state_change = NEVER
        self._schedule_next_change(start_time)
        self.in_flight = False

        self.batch_seq = 0
        self.captured = 0
        self.uploaded = 0
        self.dropped = 0
        self.failed_calls = 0
        self.offline_sec = 0.0

    # -- network ------------------------------------------------------------------
    def _state_duration(self):
        """Length of the current connectivity state, in seconds."""
        if NETWORK_UNRELIABILITY <= 0:
            return float("inf") if self.online else 0.0
        if NETWORK_UNRELIABILITY >= 1:
            return 0.0 if self.online else float("inf")

        mean_up = (MIN_NETWORK_AVAILABLE_SEC + MAX_NETWORK_AVAILABLE_SEC) / 2.0
        if self.online:
            return self.rng.uniform(MIN_NETWORK_AVAILABLE_SEC, MAX_NETWORK_AVAILABLE_SEC)
        mean_down = mean_up * NETWORK_UNRELIABILITY / (1.0 - NETWORK_UNRELIABILITY)
        return max(TICK_SEC, self.rng.uniform(0.25 * mean_down, 1.75 * mean_down))

    def _schedule_next_change(self, from_time):
        """Set the end of the current connectivity state. Never at 0% / 100% reliability."""
        duration = self._state_duration()
        if duration == float("inf"):
            self.next_state_change = NEVER
        else:
            self.next_state_change = from_time + timedelta(seconds=max(duration, TICK_SEC))

    def update_network(self, sim_now, elapsed):
        if not self.online:
            self.offline_sec += elapsed
        while sim_now >= self.next_state_change:
            self.online = not self.online
            previous_change = self.next_state_change
            self._schedule_next_change(previous_change)
            if self.next_state_change is NEVER:
                break

    # -- capture ------------------------------------------------------------------
    def capture_due(self, sim_now):
        """Move every point whose timestamp has passed into the handset buffer."""
        if self.exhausted:
            return
        if self.points is None:
            if sim_now < self.start_time:
                return
            self.points = iter_captured_points(self.file_path, self.rng)
            self.pending = next(self.points, None)

        while self.pending is not None and self.pending[0] <= sim_now:
            self.buffer.append(self.pending)
            self.captured += 1
            if len(self.buffer) > MAX_BUFFERED_POINTS:
                self.buffer.popleft()
                self.dropped += 1
            self.pending = next(self.points, None)

        if self.pending is None and self.points is not None:
            self.exhausted = True

    # -- upload -------------------------------------------------------------------
    def take_batch(self):
        """Return the next batch to POST, or None if nothing should be sent yet."""
        if self.in_flight or not self.online or not self.buffer:
            return None
        if len(self.buffer) < MIN_POINTS_PER_UPLOAD and not self.exhausted:
            return None

        size = min(len(self.buffer), MAX_POINTS_PER_UPLOAD)
        batch = [self.buffer.popleft() for _ in range(size)]
        self.in_flight = True
        self.batch_seq += 1
        return batch

    def on_success(self, batch):
        self.in_flight = False
        self.uploaded += len(batch)

    def on_failure(self, batch):
        """Upload failed - the points stay on the handset for the next attempt."""
        self.in_flight = False
        self.failed_calls += 1
        self.buffer.extendleft(reversed(batch))
        while len(self.buffer) > MAX_BUFFERED_POINTS:
            self.buffer.pop()
            self.dropped += 1

    @property
    def finished(self):
        return self.exhausted and not self.buffer and not self.in_flight

    @property
    def started(self):
        return self.points is not None

    @property
    def idle(self):
        """True when the device needs no clock time: not on shift yet, or done and drained."""
        return (not self.started or self.exhausted) and not self.buffer and not self.in_flight


# --------------------------------------------------------------------------------------
# API client
# --------------------------------------------------------------------------------------

class GeoLoggerClient:
    """Thin SimpleJWT client for /api/upload/."""

    def __init__(self, username, password, dry_run=False):
        self.dry_run = dry_run
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.access = None
        self.refresh = None

    def authenticate(self):
        if self.dry_run:
            return
        response = self.session.post(
            TOKEN_URL,
            json={"username": self.username, "password": self.password},
            timeout=REQUEST_TIMEOUT_SEC,
        )
        response.raise_for_status()
        tokens = response.json()
        self.access = tokens["access"]
        self.refresh = tokens.get("refresh")

    def refresh_token(self):
        if self.dry_run or not self.refresh:
            return False
        try:
            response = self.session.post(
                TOKEN_REFRESH_URL,
                json={"refresh": self.refresh},
                timeout=REQUEST_TIMEOUT_SEC,
            )
            if response.status_code == 200:
                self.access = response.json()["access"]
                return True
        except requests.RequestException:
            pass
        try:
            self.authenticate()
            return True
        except Exception:
            return False

    def post_points(self, payload):
        """POST a batch; returns (status_code, latency_ms, body_text, error)."""
        if self.dry_run:
            time.sleep(0.02)
            return 200, 20.0, json.dumps({"status": "ok", "count": len(payload)}), ""

        started = time.perf_counter()
        try:
            response = self.session.post(
                UPLOAD_URL,
                json=payload,
                headers={"Authorization": f"Bearer {self.access}"},
                timeout=REQUEST_TIMEOUT_SEC,
            )
            latency = (time.perf_counter() - started) * 1000.0
            return response.status_code, latency, response.text, ""
        except requests.RequestException as exc:
            latency = (time.perf_counter() - started) * 1000.0
            return None, latency, "", f"{type(exc).__name__}: {exc}"


def build_payload(device_id, batch):
    return [
        {
            "device": device_id,
            "latitude": round(lat, 7),
            "longitude": round(lon, 7),
            "accuracy": accuracy,
            "timestamp": ts.isoformat().replace("+00:00", "Z"),
        }
        for ts, lat, lon, accuracy in batch
    ]


def upload_job(client, device, batch, meta):
    """Runs on a worker thread. Returns (device, batch, ok, [log rows]).

    Everything describing the device's state is snapshotted into `meta` by the main
    loop before submission, so this thread never reads mutating device attributes.
    """
    payload = build_payload(device.device_id, batch)
    body = json.dumps(payload)
    rows = []
    ok = False

    for attempt in (1, 2):
        status, latency, text, error = client.post_points(payload)
        response_count = ""
        if text:
            try:
                response_count = json.loads(text).get("count", "")
            except (ValueError, AttributeError):
                response_count = ""

        ok = status is not None and 200 <= status < 300
        if ok:
            outcome = "success"
        elif error:
            outcome = "network_error"
        elif status == 401:
            outcome = "unauthorised"
        else:
            outcome = "http_error"

        rows.append({
            "wall_time": datetime.now().isoformat(timespec="seconds"),
            "sim_time": meta["sim_time"],
            "device_id": device.device_id,
            "agent_id": device.agent_id,
            "batch_seq": meta["batch_seq"],
            "attempt": attempt,
            "network_state": meta["network_state"],
            "points_in_batch": len(batch),
            "buffer_before": meta["buffer_before"],
            "buffer_after": meta["buffer_after"],
            "first_point_time": batch[0][0].isoformat(),
            "last_point_time": batch[-1][0].isoformat(),
            "request_bytes": len(body),
            "url": UPLOAD_URL,
            "http_status": status if status is not None else "",
            "latency_ms": round(latency, 1),
            "response_count": response_count,
            "response_body": (text or "")[:RESPONSE_BODY_CHARS].replace("\n", " "),
            "error": error,
            "outcome": outcome,
        })

        if ok or status != 401 or attempt == 2:
            break
        if not client.refresh_token():
            break

    return device, batch, ok, rows


# --------------------------------------------------------------------------------------
# Log file
# --------------------------------------------------------------------------------------

def write_log(path, records):
    """Rewrite the whole API log, overwriting the previous file."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    os.replace(tmp_path, path)


# --------------------------------------------------------------------------------------
# Device discovery
# --------------------------------------------------------------------------------------

def index_devices(input_path, run_id, max_devices, stride, rng):
    """Scan the trip files for their time span; keep only the path in memory."""
    files = sorted(f for f in os.listdir(input_path) if f.endswith((".json", ".geojson")))
    files = files[::max(stride, 1)]
    if max_devices:
        files = files[:max_devices]

    devices = []
    for file_name in tqdm(files, desc="Indexing agent trips", unit="file"):
        file_path = os.path.join(input_path, file_name)
        with open(file_path, "r") as fh:
            features = json.load(fh).get("features", [])
        if not features:
            continue

        starts = [f["properties"]["start_time"] for f in features if f["properties"].get("start_time")]
        ends = [f["properties"]["end_time"] for f in features if f["properties"].get("end_time")]
        if not starts or not ends:
            continue

        agent_id = features[0]["properties"].get("agent")
        if agent_id is None:
            agent_id = len(devices)

        tz_shift = timedelta(hours=SIM_UTC_OFFSET_HOURS)
        start = (parse_sim_time(min(starts)) - tz_shift).replace(tzinfo=timezone.utc)
        end = (parse_sim_time(max(ends)) - tz_shift).replace(tzinfo=timezone.utc)

        devices.append(SimulatedDevice(
            agent_id=agent_id,
            device_id=DEVICE_ID_TEMPLATE.format(run_id=run_id, agent=int(agent_id)),
            file_path=file_path,
            start_time=start,
            end_time=end,
            rng=random.Random(rng.randrange(2 ** 31)),
        ))

    return devices


# --------------------------------------------------------------------------------------
# Main replay loop
# --------------------------------------------------------------------------------------

def main():
    global MAX_DEVICES, DEVICE_STRIDE, TIME_SCALE, NETWORK_UNRELIABILITY
    global MAX_NETWORK_AVAILABLE_SEC, POINT_SPACING_M, MIN_POINTS_PER_UPLOAD, LOG_INTERVAL_SEC

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devices", default=str(MAX_DEVICES),
                        help="number of simulated vehicles, or 'all'")
    parser.add_argument("--stride", type=int, default=DEVICE_STRIDE,
                        help="take every Nth agent file, to spread the sample over the fleet")
    parser.add_argument("--time-scale", type=float, default=TIME_SCALE,
                        help="simulated seconds per real second (1 = real time)")
    parser.add_argument("--spacing", type=float, default=POINT_SPACING_M,
                        help="metres between captured points")
    parser.add_argument("--min-points", type=int, default=MIN_POINTS_PER_UPLOAD,
                        help="minimum points per API call")
    parser.add_argument("--unreliability", type=float, default=NETWORK_UNRELIABILITY,
                        help="fraction of time a device has no network (0-1)")
    parser.add_argument("--max-network-sec", type=float, default=MAX_NETWORK_AVAILABLE_SEC,
                        help="longest continuous window the network stays available")
    parser.add_argument("--log-interval", type=float, default=LOG_INTERVAL_SEC,
                        help="wall-clock seconds between CSV rewrites")
    parser.add_argument("--run-id", default=RUN_ID, help="shared run element of the device ids")
    parser.add_argument("--dry-run", action="store_true",
                        help="simulate the uploads without calling the API")
    add_scenario_argument(parser)
    args = parser.parse_args()

    MAX_DEVICES = None if str(args.devices).lower() == "all" else int(args.devices)
    DEVICE_STRIDE = max(args.stride, 1)
    TIME_SCALE = max(args.time_scale, 1e-6)
    POINT_SPACING_M = args.spacing
    MIN_POINTS_PER_UPLOAD = args.min_points
    NETWORK_UNRELIABILITY = min(max(args.unreliability, 0.0), 1.0)
    MAX_NETWORK_AVAILABLE_SEC = args.max_network_sec
    LOG_INTERVAL_SEC = args.log_interval

    scenario = load_scenario(args.scenario)
    input_path = scenario.trips_time_dir
    output_path = scenario.output_dir
    log_path = os.path.join(output_path, LOG_FILE_NAME)

    if not os.path.isdir(input_path):
        sys.exit(f"Trip folder not found: {input_path}")

    run_id = args.run_id or "sim-" + datetime.now().strftime("%Y%m%d-%H%M")
    rng = random.Random(RANDOM_SEED)

    username = os.environ.get("GEOLOGGER_USERNAME", "")
    password = os.environ.get("GEOLOGGER_PASSWORD", "")
    if not args.dry_run and not (username and password):
        sys.exit("Set GEOLOGGER_USERNAME and GEOLOGGER_PASSWORD, or run with --dry-run.")

    devices = index_devices(input_path, run_id, MAX_DEVICES, DEVICE_STRIDE, rng)
    if not devices:
        sys.exit("No usable agent trip files found.")

    sim_start = min(d.start_time for d in devices)
    sim_end = max(d.end_time for d in devices)
    hard_stop = sim_end + timedelta(seconds=MAX_FLUSH_EXTRA_SEC)

    client = GeoLoggerClient(username, password, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"Authenticating as {username} at {TOKEN_URL} ...")
        client.authenticate()

    print(f"Run id            : {run_id}")
    print(f"Devices           : {len(devices)}  ({devices[0].device_id} ... {devices[-1].device_id})")
    print(f"Simulated window  : {sim_start.isoformat()} -> {sim_end.isoformat()}")
    print(f"Point spacing     : {POINT_SPACING_M:.0f} m (or every {MAX_POINT_INTERVAL_SEC:.0f} s)")
    print(f"Batch size        : >= {MIN_POINTS_PER_UPLOAD} points, <= {MAX_POINTS_PER_UPLOAD}")
    print(f"Network           : {NETWORK_UNRELIABILITY:.0%} offline, up to "
          f"{MAX_NETWORK_AVAILABLE_SEC:.0f} s available at a time")
    print(f"Time scale        : x{TIME_SCALE:.0f}")
    print(f"API log           : {log_path} (rewritten every {LOG_INTERVAL_SEC / 60:.0f} min)")
    print(f"Mode              : {'DRY RUN - no API calls' if args.dry_run else 'live uploads'}\n")

    records = []
    completions = queue.Queue()
    executor = ThreadPoolExecutor(max_workers=UPLOAD_WORKERS)

    sim_now = sim_start
    wall_start = time.time()
    last_log_write = wall_start
    in_flight = 0
    skipped_sec = 0.0

    progress = tqdm(total=int((sim_end - sim_start).total_seconds()),
                    desc="Replaying", unit="sim-s", unit_scale=True)

    try:
        while True:
            for device in devices:
                device.update_network(sim_now, TICK_SEC)
                device.capture_due(sim_now)

                batch = device.take_batch()
                if batch is not None:
                    meta = {
                        "sim_time": sim_now.isoformat(),
                        "batch_seq": device.batch_seq,
                        "network_state": "online" if device.online else "offline",
                        "buffer_after": len(device.buffer),
                        "buffer_before": len(device.buffer) + len(batch),
                    }
                    in_flight += 1
                    executor.submit(
                        lambda d=device, b=batch, m=meta:
                        completions.put(upload_job(client, d, b, m))
                    )

            while True:
                try:
                    device, batch, ok, rows = completions.get_nowait()
                except queue.Empty:
                    break
                in_flight -= 1
                records.extend(rows)
                if ok:
                    device.on_success(batch)
                else:
                    device.on_failure(batch)

            now = time.time()
            try:
                if now - last_log_write >= LOG_INTERVAL_SEC:
                    write_log(log_path, records)
                    last_log_write = now
                    progress.write(f"[{datetime.now():%H:%M:%S}] wrote {len(records)} API "
                                f"records to {LOG_FILE_NAME}")
            except:
                pass
            all_done = all(d.finished for d in devices) and in_flight == 0
            if all_done or sim_now >= hard_stop:
                break

            sim_now += timedelta(seconds=TICK_SEC)
            progress.update(TICK_SEC)

            # Nothing is moving, buffering or in flight: jump to the next vehicle's shift
            # rather than ticking through hours of empty simulated time.
            if SKIP_IDLE_GAPS and in_flight == 0 and all(d.idle for d in devices):
                upcoming = [d.start_time for d in devices if not d.started and d.start_time > sim_now]
                if upcoming:
                    gap = (min(upcoming) - sim_now).total_seconds()
                    if gap >= MIN_IDLE_GAP_SEC:
                        sim_now = min(upcoming)
                        skipped_sec += gap
                        progress.update(gap)
                        progress.write(f"No activity - skipped {gap / 60:.0f} min to "
                                       f"{sim_now.isoformat()}")

            # Pace the replay against the wall clock, ignoring any time we skipped.
            # Sleep in slices so a slow tick still gives up the whole of its wait while
            # staying responsive to Ctrl-C: one capped sleep would let the clock run fast.
            target = ((sim_now - sim_start).total_seconds() - skipped_sec) / TIME_SCALE
            while True:
                drift = target - (time.time() - wall_start)
                if drift <= 0:
                    break
                time.sleep(min(drift, 0.5))
    except KeyboardInterrupt:
        progress.write("Interrupted - draining in-flight requests and writing the log.")
    finally:
        progress.close()
        executor.shutdown(wait=True)
        while not completions.empty():
            device, batch, ok, rows = completions.get()
            records.extend(rows)
            if ok:
                device.on_success(batch)
            else:
                device.on_failure(batch)
        try:
            write_log(log_path, records)
        except:
            pass

        

    captured = sum(d.captured for d in devices)
    uploaded = sum(d.uploaded for d in devices)
    dropped = sum(d.dropped for d in devices)
    unsent = sum(len(d.buffer) for d in devices)
    failed = sum(d.failed_calls for d in devices)
    successful = sum(1 for r in records if r["outcome"] == "success")

    print(f"\nPoints captured    : {captured}")
    print(f"Points uploaded    : {uploaded}")
    print(f"Points still buffered when the run ended: {unsent}")
    print(f"Points dropped (buffer full)            : {dropped}")
    print(f"API calls          : {len(records)} ({successful} succeeded, {failed} failed)")
    print(f"API log written to : {log_path}")


if __name__ == "__main__":
    main()
