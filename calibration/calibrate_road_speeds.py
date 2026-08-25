"""
Calibrate road_speed_km-h against observed trip durations.

Reads the per-trip analysis written by Simulation/captured_trips_to_geojson.py
(trip_routing_analysis.csv) and finds the speed for each highway type that makes
the routed times agree as closely as possible with the durations actually
recorded. Writes one CSV of fitted speeds to the scenario's output folder.

    python utilities/calibrate_road_speeds.py --scenario scenario_nairobi.json


================================================================================
THE ALGORITHM
================================================================================

The model
---------
The routing engine computes a trip's time as the sum, over highway types, of the
distance travelled on that type divided by that type's speed:

        t_i(v)  =  sum_h  d_ih / v_h

where
        d_ih  = metres trip i spends on highway type h   (the highway_breakdown
                column, which is exactly this decomposition)
        v_h   = the speed for type h, in m/s -- what we are solving for
        T_i   = the duration actually recorded for trip i

Fitting v directly is awkward: t_i(v) is nonlinear in v, so the least-squares
problem is non-convex, needs a starting guess, and can settle in a local
optimum.

The substitution that makes it exact
------------------------------------
Work in SLOWNESS instead of speed -- u_h = 1 / v_h, seconds per metre:

        t_i(u)  =  sum_h  d_ih * u_h  =  (D u)_i

which is LINEAR in u. D is the (trips x types) matrix of per-type distances,
read straight out of the analysis file. Calibration is therefore an ordinary
linear regression, with all the properties that brings: convex, a unique
solution up to rank deficiency, and no iteration or heuristic anywhere.

The speeds come back as v_h = 1 / u_h at the end.

Mean or median? -- what --target selects
----------------------------------------
The two objectives below are not just different norms; they fit different
statistics, and which one you want depends on the question.

    --target mean    (least squares, L2)
        Minimises the sum of SQUARED residuals. The fitted speeds make the
        routing match the MEAN recorded duration. Squaring means one trip that
        took ten times too long counts a hundred times as much as one that took
        slightly too long, so a handful of stuck-in-traffic trips can drag every
        speed with them.

    --target median  (least absolute error, L1)
        Minimises the sum of ABSOLUTE residuals. This is median (quantile)
        regression: the fitted speeds make the routing match the MEDIAN recorded
        duration, so a typical trip is matched and freak ones are largely
        ignored. Every residual counts in proportion to its size, never its
        square.

For GPS trip data, --target median is usually the honest choice: durations
include arbitrary amounts of standing still, which produces a long right tail
that the mean chases and the median does not.

One thing neither option does: minimise the MEDIAN of the absolute errors
directly (least median of squares). That objective is non-convex, has no exact
solution, and needs a randomised search -- a genuinely different piece of
machinery. --target median minimises the SUM of absolute errors, which is what
makes the fit track the median duration. The report prints both the mean and the
median error either way, so the effect of the choice is visible rather than
asserted.

The objective
-------------
--target mean (--objective l2) minimises the sum of squared time discrepancies:

        minimise  || D u - T ||^2      subject to   u_lo <= u <= u_hi

The bounds are just the speed limits expressed as slowness --
u_hi = 1/v_min and u_lo = 1/v_max -- and they are what keeps every fitted speed
physically sensible. Without them a type that happens to correlate with short
trips can be handed a negative or absurd slowness, which is arithmetically
optimal and physically nonsense.

This is a bounded linear least-squares problem, solved exactly by
scipy.optimize.lsq_linear (trust-region reflective). For ~20 highway types the
normal equations are a 20x20 system: it is instantaneous regardless of how many
trips there are.

--target median (--objective l1) minimises the sum of ABSOLUTE discrepancies, far
less sensitive to the occasional wild duration (this dataset contains a
987-minute "trip"). L1 has no closed form, but it is still exactly solvable as a
linear program by splitting each residual into positive and negative parts:

        minimise    sum_i (e+_i + e-_i)
        subject to  D u - e+ + e-  =  T
                    e+ >= 0,  e- >= 0,  u_lo <= u <= u_hi

Still convex, still a global optimum, just slower: the LP has one constraint per
trip and two extra variables per trip, so 65,000 trips means a 65,000 x 130,000
sparse program. Minutes rather than milliseconds.

What the fit cannot do
----------------------
A type that appears in very little distance is barely constrained by the data,
and its "optimal" speed is mostly noise. Types below --min-distance-km are
therefore held at their current value rather than fitted, and every type's
distance share and trip count are reported so a thin fit is visible rather than
implied. Types that are perfectly collinear (always used together, in a fixed
ratio) cannot be separated at all -- the fit will split the difference between
them arbitrarily, and no amount of data fixes that.

A caveat that matters more than the algorithm
---------------------------------------------
Recorded durations include time stopped in traffic, at lights, and waiting.
Speeds fitted to them are therefore effective door-to-door speeds, not free-flow
limits, and will come out slower than a speed limit. That may be exactly what
you want for a simulation -- or not. --subtract-idle removes the idle_time_min
column from each duration first, fitting to moving time instead.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import load_scenario, add_scenario_argument, writable_path

DEFAULT_MIN_SPEED_KMH = 3.0     # slower than walking is not a road
DEFAULT_MAX_SPEED_KMH = 120.0
DEFAULT_MIN_DISTANCE_KM = 5.0   # below this a type is not really constrained
# Only trips the router essentially reproduced can inform a speed: see
# load_analysis for why a route disagreement cannot be read as a speed.
DEFAULT_MAX_DISTANCE_ERROR_PCT = 20.0


# ============================================================
# DATA
# ============================================================

def load_analysis(path, subtract_idle, max_duration_min, max_distance_error_pct):
    """
    Read the per-trip analysis and return the pieces the fit needs.

    Returns (design DataFrame indexed by trip with one column per highway type,
    observed durations in seconds, dict of drop counts).

    max_distance_error_pct is the important filter. Calibrating a speed means
    dividing a recorded DURATION by a routed DISTANCE, which is only meaningful
    if the router went roughly where the vehicle went. When the two distances
    disagree badly the trip says nothing about speed -- it says the driver took
    a different route -- and including it forces the fit to absorb that route
    difference as though it were a speed difference. Trips outside the tolerance
    are therefore excluded rather than down-weighted.
    """
    df = pd.read_csv(path)
    dropped = {}

    required = {"highway_breakdown", "duration_min", "routed"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing column(s): {', '.join(sorted(missing))}\n"
                         "Is it the trip_routing_analysis.csv written by "
                         "Simulation/captured_trips_to_geojson.py?")

    before = len(df)
    # Straight-line fallbacks have no per-type breakdown, so they carry no
    # information about any speed and would just add noise to the residuals.
    df = df[df["routed"] == True]                                    # noqa: E712
    dropped["not routed (straight-line fallback)"] = before - len(df)

    if max_distance_error_pct:
        have_distance = {"distance_km", "routed_distance_m"} <= set(df.columns)
        if not have_distance:
            raise SystemExit(
                "Cannot apply --max-distance-error-pct: this analysis file has "
                "no distance_km / routed_distance_m columns. Regenerate it with "
                "the current Simulation/captured_trips_to_geojson.py, or pass "
                "--max-distance-error-pct 0 to disable the filter.")
        recorded = pd.to_numeric(df["distance_km"], errors="coerce")
        routed_km = pd.to_numeric(df["routed_distance_m"], errors="coerce") / 1000.0
        rel_error = (routed_km - recorded).abs() / recorded.where(recorded > 0)

        before = len(df)
        df = df[rel_error <= max_distance_error_pct / 100.0]
        dropped[f"routed distance off by more than "
                f"{max_distance_error_pct:g}% of recorded"] = before - len(df)

    duration_s = pd.to_numeric(df["duration_min"], errors="coerce") * 60.0
    if subtract_idle and "idle_time_min" in df.columns:
        idle_s = pd.to_numeric(df["idle_time_min"], errors="coerce").fillna(0.0) * 60.0
        duration_s = duration_s - idle_s

    before = len(df)
    keep = duration_s > 0
    df, duration_s = df[keep], duration_s[keep]
    dropped["non-positive duration"] = before - len(df)

    if max_duration_min:
        before = len(df)
        keep = duration_s <= max_duration_min * 60.0
        df, duration_s = df[keep], duration_s[keep]
        dropped[f"duration over {max_duration_min:g} min"] = before - len(df)

    # Expand the JSON breakdown into one column of metres per highway type.
    rows = df["highway_breakdown"].apply(json.loads)
    types = sorted({h for r in rows for h in r})
    design = pd.DataFrame(
        [[r.get(h, {}).get("distance_m", 0.0) for h in types] for r in rows],
        columns=types, index=df.index, dtype=float)

    before = len(design)
    keep = design.sum(axis=1) > 0
    design, duration_s, df = design[keep], duration_s[keep], df[keep]
    dropped["no routed distance"] = before - len(design)

    return design, duration_s.to_numpy(dtype=float), df, dropped


# ============================================================
# THE FIT
# ============================================================

def fit_slowness(D, T, u_lo, u_hi, objective):
    """
    Solve for slowness u (seconds per metre) per highway type.

    Both objectives are convex and return a global optimum; see the module
    docstring for the formulations.
    """
    if objective == "l2":
        # Bounded linear least squares: minimise ||D u - T||^2, u in [u_lo, u_hi].
        result = lsq_linear(D, T, bounds=(u_lo, u_hi), method="trf")
        return result.x

    # L1: minimise sum |D u - T| as a linear program.
    from scipy.optimize import linprog
    from scipy.sparse import hstack, identity, csr_matrix

    n_trips, n_types = D.shape
    eye = identity(n_trips, format="csr")
    # variables: [u (n_types), e+ (n_trips), e- (n_trips)]
    A_eq = hstack([csr_matrix(D), -eye, eye], format="csr")
    c = np.concatenate([np.zeros(n_types), np.ones(2 * n_trips)])
    bounds = ([(lo, hi) for lo, hi in zip(u_lo, u_hi)]
              + [(0, None)] * (2 * n_trips))
    res = linprog(c, A_eq=A_eq, b_eq=T, bounds=bounds, method="highs")
    if not res.success:
        raise SystemExit(f"L1 solve failed: {res.message}")
    return res.x[:n_types]


def discrepancy(D, T, u):
    """Summary of how far routed times sit from recorded ones."""
    residual = D @ u - T
    return {
        "total_abs_error_h": float(np.abs(residual).sum() / 3600.0),
        "mean_abs_error_s": float(np.abs(residual).mean()),
        "median_abs_error_s": float(np.median(np.abs(residual))),
        "rmse_s": float(np.sqrt((residual ** 2).mean())),
        "bias_s": float(residual.mean()),          # +ve = routing too slow
    }


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scenario_argument(p)
    p.add_argument("--input", default=None,
                   help="trip_routing_analysis.csv (default: in the scenario's "
                        "output_trips_time_queued folder)")
    p.add_argument("--output", default=None,
                   help="CSV to write (default: road_speed_calibration.csv in "
                        "the scenario's output folder)")
    p.add_argument("--target", choices=("mean", "median"), default=None,
                   help="which discrepancy the fit should minimise: "
                        "'mean' matches the mean recorded duration (least "
                        "squares; fast, but chased by outliers), 'median' "
                        "matches the median (least absolute error; robust, "
                        "slower). Default: median")
    p.add_argument("--objective", choices=("l2", "l1"), default=None,
                   help="the same choice named after the norm: "
                        "l2 == --target mean, l1 == --target median")
    p.add_argument("--min-speed", type=float, default=DEFAULT_MIN_SPEED_KMH,
                   help=f"lower bound on any fitted speed, km/h "
                        f"(default {DEFAULT_MIN_SPEED_KMH:g})")
    p.add_argument("--max-speed", type=float, default=DEFAULT_MAX_SPEED_KMH,
                   help=f"upper bound on any fitted speed, km/h "
                        f"(default {DEFAULT_MAX_SPEED_KMH:g})")
    p.add_argument("--min-distance-km", type=float, default=DEFAULT_MIN_DISTANCE_KM,
                   help="hold a highway type at its current speed when the data "
                        "contains less than this much distance on it")
    p.add_argument("--subtract-idle", action="store_true",
                   help="fit to moving time by removing idle_time_min from each "
                        "recorded duration")
    p.add_argument("--max-duration-min", type=float, default=None,
                   help="ignore trips longer than this many minutes")
    p.add_argument("--max-distance-error-pct", type=float,
                   default=DEFAULT_MAX_DISTANCE_ERROR_PCT,
                   help="only calibrate on trips whose routed distance is "
                        "within this percent of the recorded distance "
                        f"(default {DEFAULT_MAX_DISTANCE_ERROR_PCT:g}); 0 "
                        "disables the filter. Trips where the router took a "
                        "materially different route say nothing about speed")
    return p.parse_args()


TARGET_TO_OBJECTIVE = {"mean": "l2", "median": "l1"}
OBJECTIVE_TO_TARGET = {v: k for k, v in TARGET_TO_OBJECTIVE.items()}


def resolve_objective(args):
    """--target and --objective are the same switch under two names."""
    if args.target and args.objective:
        if TARGET_TO_OBJECTIVE[args.target] != args.objective:
            raise SystemExit(
                f"--target {args.target} and --objective {args.objective} "
                "disagree; they are the same switch "
                "(mean == l2, median == l1). Pass one.")
    if args.target:
        return TARGET_TO_OBJECTIVE[args.target]
    if args.objective:
        return args.objective
    # Median by default: recorded durations include time spent stationary, and
    # that long right tail pulls a least-squares fit away from the typical trip.
    return "l1"


def main():
    args = parse_args()
    objective = resolve_objective(args)
    scenario = load_scenario(args.scenario)
    current = scenario["road_speed_km-h"]

    src = args.input or os.path.join(scenario.trips_time_dir,
                                     "trip_routing_analysis.csv")
    if not os.path.exists(src):
        raise SystemExit(
            f"No analysis file at {src}\nRun Simulation/captured_trips_to_geojson.py "
            "first -- it writes trip_routing_analysis.csv alongside the tracks.")
    out = args.output or os.path.join(scenario.output_dir,
                                      "road_speed_calibration.csv")

    print(f"Scenario : {scenario.name}")
    print(f"Analysis : {src}")
    print(f"Target   : {OBJECTIVE_TO_TARGET[objective]} discrepancy  "
          f"({'L2, least squares' if objective == 'l2' else 'L1, least absolute error'})")
    if args.subtract_idle:
        print("Fitting to moving time (idle_time_min removed from each duration)")
    if args.max_distance_error_pct:
        print(f"Using only trips the router reproduced to within "
              f"{args.max_distance_error_pct:g}% on distance")

    design, T, df, dropped = load_analysis(src, args.subtract_idle,
                                           args.max_duration_min,
                                           args.max_distance_error_pct)
    types = list(design.columns)
    print(f"\nTrips used: {len(design):,}   highway types: {len(types)}")
    for reason, n in dropped.items():
        if n:
            print(f"  dropped {n:,}: {reason}")

    dist_km = design.sum(axis=0) / 1000.0
    trips_using = (design > 0).sum(axis=0)

    # Types with too little distance are pinned to their current speed rather
    # than fitted: the data does not constrain them, and a confident-looking
    # number derived from 200 m of road would be worse than no number.
    thin = {h for h in types if dist_km[h] < args.min_distance_km}
    fitted_types = [h for h in types if h not in thin]
    if thin:
        print(f"  holding {len(thin)} thin type(s) at current speed: "
              + ", ".join(sorted(thin)))
    if not fitted_types:
        raise SystemExit("No highway type has enough distance to calibrate.")

    D_all = design.to_numpy()
    # Distance on pinned types still contributes time; it is moved to the
    # right-hand side at the current speed so it does not distort the fit.
    T_fit = T.copy()
    for h in thin:
        speed_ms = max(float(current.get(h, 0.1)), 0.1) / 3.6
        T_fit = T_fit - design[h].to_numpy() / speed_ms

    D = design[fitted_types].to_numpy()
    u_lo = np.full(len(fitted_types), 1.0 / (args.max_speed / 3.6))
    u_hi = np.full(len(fitted_types), 1.0 / (args.min_speed / 3.6))

    if objective == "l1":
        print(f"\nSolving an LP with {len(D):,} constraints -- this takes a while ...")
    u = fit_slowness(D, T_fit, u_lo, u_hi, objective)

    # Compare like with like: the "before" uses the scenario's current speeds.
    u_current = np.array([1.0 / (max(float(current.get(h, 0.1)), 0.1) / 3.6)
                          for h in fitted_types])
    before_stats = discrepancy(D, T_fit, u_current)
    after_stats = discrepancy(D, T_fit, u)

    speeds = {h: 3.6 / u_i for h, u_i in zip(fitted_types, u)}

    def evidence(share_pct, n_trips):
        """
        How much the data actually says about this type.

        A fitted speed is only as good as the distance behind it: a type that
        carries 0.1% of the mileage is barely constrained, and its "optimal"
        speed is mostly whatever makes the residuals of a few trips tidy. These
        thresholds are a reading aid, not a statistical test -- the raw distance
        and trip count sit next to them so the judgement stays yours.
        """
        if share_pct < 1.0 or n_trips < 100:
            return "thin"
        if share_pct >= 5.0 and n_trips >= 1000:
            return "strong"
        return "moderate"

    n_trips_total = len(design)
    rows = []
    for h in types:
        cur = current.get(h)
        is_fitted = h in speeds
        fit = speeds.get(h, cur)
        at_bound = ""
        if is_fitted:
            if abs(fit - args.min_speed) < 1e-6:
                at_bound = "min"
            elif abs(fit - args.max_speed) < 1e-6:
                at_bound = "max"
        share = float(100 * dist_km[h] / dist_km.sum())
        n_using = int(trips_using[h])
        rows.append({
            "highway": h,
            "current_speed_kmh": cur,
            "fitted_speed_kmh": round(float(fit), 2) if fit is not None else None,
            "change_kmh": (round(float(fit) - float(cur), 2)
                           if is_fitted and cur is not None else None),
            # Sample size, so a recommendation can be weighed rather than taken.
            "evidence": evidence(share, n_using) if is_fitted else "not fitted",
            "distance_km": round(float(dist_km[h]), 1),
            "distance_share_pct": round(share, 2),
            "trips_using": n_using,
            "trips_share_pct": round(100.0 * n_using / n_trips_total, 1),
            "mean_distance_per_trip_m": (round(float(1000 * dist_km[h] / n_using), 1)
                                         if n_using else 0.0),
            "in_scenario": cur is not None,
            "fitted": is_fitted,
            "at_bound": at_bound,
        })
    # Types configured in the scenario but never driven on: reported so the file
    # is a complete picture of road_speed_km-h, not just the part the data saw.
    for h, cur in sorted(current.items()):
        if h not in types:
            rows.append({"highway": h, "current_speed_kmh": cur,
                         "fitted_speed_kmh": cur, "change_kmh": None,
                         "evidence": "no data", "distance_km": 0.0,
                         "distance_share_pct": 0.0, "trips_using": 0,
                         "trips_share_pct": 0.0, "mean_distance_per_trip_m": 0.0,
                         "in_scenario": True, "fitted": False, "at_bound": ""})

    out_df = pd.DataFrame(rows).sort_values("distance_km", ascending=False)
    out = writable_path(out)
    out_df.to_csv(out, index=False)

    print(f"\n{'highway':22s} {'current':>8s} {'fitted':>8s} {'change':>8s} "
          f"{'dist km':>9s} {'share':>7s} {'trips':>8s} {'evidence':>9s}")
    for r in out_df.itertuples():
        if not r.fitted:
            continue
        print(f"  {r.highway:20s} {r.current_speed_kmh!s:>8s} "
              f"{r.fitted_speed_kmh:>8.1f} {r.change_kmh:>+8.1f} "
              f"{r.distance_km:>9,.0f} {r.distance_share_pct:>6.1f}% "
              f"{r.trips_using:>8,} {r.evidence:>9s}"
              + (f"  [at {r.at_bound}]" if r.at_bound else ""))

    thin_rows = [r for r in out_df.itertuples() if r.fitted and r.evidence == "thin"]
    if thin_rows:
        print(f"\n  {len(thin_rows)} type(s) marked thin: fitted from under 1% of "
              "the distance\n  or fewer than 100 trips. Treat those speeds as "
              "indicative only.")

    print(f"\nTime discrepancy over {len(D):,} trips")
    print(f"{'':22s} {'current':>12s} {'calibrated':>12s}")
    for key, label, scale in [("total_abs_error_h", "total |error|, h", 1),
                              ("mean_abs_error_s", "mean |error|, s", 1),
                              ("median_abs_error_s", "median |error|, s", 1),
                              ("rmse_s", "RMSE, s", 1),
                              ("bias_s", "bias, s", 1)]:
        print(f"  {label:20s} {before_stats[key]:>12,.1f} {after_stats[key]:>12,.1f}")
    print("  (bias > 0 means the routing is slower than what was recorded)")

    print(f"\nWrote {out}")
    print("Nothing has been changed in the scenario -- copy the fitted column "
          "into road_speed_km-h if you want to adopt it.")


if __name__ == "__main__":
    main()
