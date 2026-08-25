"""
Is the gap before a trip related to the gap before the one prior to it?

For every trip, this pairs two numbers:

    previous gap  = end of trip i-2  ->  start of trip i-1
    current gap   = end of trip i-1  ->  start of trip i

buckets both into the same bands used by trip_gap_analysis.py, and counts how
often each combination occurs. Writes the contingency table as CSV and a heatmap
into the scenario's output folder.

    python calibration/trip_gap_correlation.py --scenario scenario.json

What it answers
---------------
Whether idle time is "sticky". If a vehicle that has just waited an hour tends
to wait again, the two gaps are positively associated and the diagonal of the
table is heavy -- vehicles fall into busy and quiet spells. If the gaps are
independent, each pause is drawn fresh regardless of what came before, and a
simulation can sample idle time without carrying any state between trips. That
is a modelling decision, and this table is the evidence for it.

Which trips qualify
-------------------
Both gaps have to be real gaps between two journeys, so a trip is used only when
neither it NOR the trip before it is the first of that vehicle's day. The gap
before a first-of-day trip is overnight and says nothing about working rhythm.

In practice that means keeping the THIRD trip of each vehicle-day onwards:

    trip 0 of the day   current gap is overnight              -> excluded
    trip 1 of the day   current gap fine, previous is overnight -> excluded
    trip 2 onwards      both gaps are between real journeys   -> used

How association is measured
---------------------------
Three complementary numbers, because each can mislead alone:

  * chi-square tests whether the table differs from what independence predicts.
    With tens of thousands of trips it will report significance for differences
    far too small to matter, so it is reported but not leaned on.

  * Cramer's V rescales chi-square to 0..1 and does not grow with sample size,
    so it says how STRONG the association is rather than how detectable.
    Roughly: <0.1 negligible, 0.1-0.3 weak, 0.3-0.5 moderate, >0.5 strong.

  * Spearman's rho on the raw gap durations, which uses the ordering the bands
    throw away. The bands are ordered, so a monotonic relationship should show
    here even when the table looks flat.

The standardised residuals in the CSV show WHERE the table departs from
independence: positive means that combination happens more often than chance.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trip_gap_analysis import GAP_BANDS          # one definition of the bands

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import load_scenario, add_scenario_argument, writable_path

BAND_LABELS = [label for label, _lo, _hi in GAP_BANDS]


def load_pairs(path):
    """
    (previous gap, current gap) in minutes for every qualifying trip.

    Returns (DataFrame with prev_gap_min/gap_min/band indices, drop counts).
    """
    df = pd.read_csv(path)
    needed = {"user_id", "start_date", "start_time", "gap_from_previous_s"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(
            f"{path} is missing column(s): {', '.join(sorted(missing))}\n"
            "Regenerate it with the current calibration/captured_trips_to_geojson.py.")

    dropped = {}
    df = df.sort_values(["user_id", "start_date", "start_time"]).reset_index(drop=True)

    # Position within the vehicle-day. Both gaps are real journey-to-journey
    # gaps only from the third trip onwards; see the module docstring.
    day = df.groupby(["user_id", "start_date"], sort=False)
    df["position_in_day"] = day.cumcount()
    df["prev_gap_s"] = day["gap_from_previous_s"].shift(1)

    before = len(df)
    df = df[df["position_in_day"] >= 2]
    dropped["first or second trip of a vehicle-day"] = before - len(df)

    before = len(df)
    df = df[df["gap_from_previous_s"].notna() & df["prev_gap_s"].notna()]
    dropped["missing gap"] = before - len(df)

    before = len(df)
    df = df[(df["gap_from_previous_s"] >= 0) & (df["prev_gap_s"] >= 0)]
    dropped["negative gap (overlapping trips)"] = before - len(df)

    out = pd.DataFrame({
        "user_id": df["user_id"].to_numpy(),
        "prev_gap_min": df["prev_gap_s"].to_numpy() / 60.0,
        "gap_min": df["gap_from_previous_s"].to_numpy() / 60.0,
    })
    for column, source in (("prev_band", "prev_gap_min"), ("band", "gap_min")):
        out[column] = np.select(
            [(out[source] >= lo) & (out[source] < hi) for _l, lo, hi in GAP_BANDS],
            list(range(len(GAP_BANDS))), default=-1)

    before = len(out)
    out = out[(out["prev_band"] >= 0) & (out["band"] >= 0)]
    dropped["gap outside every band"] = before - len(out)
    return out, dropped


def analyse(pairs):
    """Contingency table plus the three association measures."""
    n_bands = len(GAP_BANDS)
    observed = np.zeros((n_bands, n_bands), dtype=int)
    for prev, cur in zip(pairs["prev_band"], pairs["band"]):
        observed[prev, cur] += 1

    # Drop all-zero rows/cols for the test only; chi2 cannot handle them, but
    # the reported table keeps every band so the shape stays comparable.
    mask_r = observed.sum(axis=1) > 0
    mask_c = observed.sum(axis=0) > 0
    compact = observed[np.ix_(mask_r, mask_c)]

    chi2, p_value, dof, expected_compact = stats.chi2_contingency(compact)
    n = compact.sum()
    k = min(compact.shape)
    cramers_v = float(np.sqrt(chi2 / (n * (k - 1)))) if n and k > 1 else float("nan")

    expected = np.full(observed.shape, np.nan)
    expected[np.ix_(mask_r, mask_c)] = expected_compact
    with np.errstate(invalid="ignore", divide="ignore"):
        residual = (observed - expected) / np.sqrt(expected)

    rho, rho_p = stats.spearmanr(pairs["prev_gap_min"], pairs["gap_min"])
    return {
        "observed": observed, "expected": expected, "residual": residual,
        "chi2": float(chi2), "p_value": float(p_value), "dof": int(dof),
        "cramers_v": cramers_v, "spearman_rho": float(rho),
        "spearman_p": float(rho_p), "n": int(observed.sum()),
    }


def strength(v):
    return ("negligible" if v < 0.1 else "weak" if v < 0.3
            else "moderate" if v < 0.5 else "strong")


def to_rows(result):
    """Long-format table: one row per (previous band, current band)."""
    observed, expected, residual = (result["observed"], result["expected"],
                                    result["residual"])
    row_tot = observed.sum(axis=1)
    col_tot = observed.sum(axis=0)
    total = observed.sum()
    rows = []
    for i, prev_label in enumerate(BAND_LABELS):
        for j, cur_label in enumerate(BAND_LABELS):
            rows.append({
                "previous_gap_band": prev_label,
                "current_gap_band": cur_label,
                "count": int(observed[i, j]),
                "expected_if_independent": (round(float(expected[i, j]), 1)
                                            if np.isfinite(expected[i, j]) else None),
                "std_residual": (round(float(residual[i, j]), 2)
                                 if np.isfinite(residual[i, j]) else None),
                "pct_of_all": round(100.0 * observed[i, j] / total, 3) if total else 0.0,
                "pct_of_previous_band": (round(100.0 * observed[i, j] / row_tot[i], 2)
                                         if row_tot[i] else 0.0),
                "pct_of_current_band": (round(100.0 * observed[i, j] / col_tot[j], 2)
                                        if col_tot[j] else 0.0),
            })
    return rows


def plot(result, title, out_path):
    observed, residual = result["observed"], result["residual"]
    row_tot = observed.sum(axis=1, keepdims=True)
    row_pct = np.divide(100.0 * observed, row_tot, where=row_tot > 0,
                        out=np.zeros_like(observed, dtype=float))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    im = axes[0].imshow(row_pct, cmap="Blues", vmin=0)
    axes[0].set_title("Where each previous-gap band leads\n(row %, sums to 100 across)",
                      fontsize=12, fontweight="bold")
    for i in range(len(BAND_LABELS)):
        for j in range(len(BAND_LABELS)):
            axes[0].text(j, i, f"{observed[i, j]:,}\n{row_pct[i, j]:.0f}%",
                         ha="center", va="center", fontsize=9,
                         color="white" if row_pct[i, j] > row_pct.max() * 0.6 else "#2c3e50")
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="% of the previous-gap band")

    limit = np.nanmax(np.abs(residual)) if np.isfinite(residual).any() else 1
    im2 = axes[1].imshow(residual, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[1].set_title("Departure from independence\n(standardised residual)",
                      fontsize=12, fontweight="bold")
    for i in range(len(BAND_LABELS)):
        for j in range(len(BAND_LABELS)):
            if np.isfinite(residual[i, j]):
                axes[1].text(j, i, f"{residual[i, j]:+.1f}", ha="center",
                             va="center", fontsize=10, color="#2c3e50")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, label="+ more often than chance")

    for ax in axes:
        ax.set_xticks(range(len(BAND_LABELS)))
        ax.set_yticks(range(len(BAND_LABELS)))
        ax.set_xticklabels(BAND_LABELS, rotation=20, ha="right", fontsize=9)
        ax.set_yticklabels(BAND_LABELS, fontsize=9)
        ax.set_xlabel("gap before CURRENT trip")
        ax.set_ylabel("gap before PREVIOUS trip")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01,
             f"n = {result['n']:,} trips   |   "
             f"Cramer's V = {result['cramers_v']:.3f} ({strength(result['cramers_v'])})   |   "
             f"Spearman rho = {result['spearman_rho']:+.3f}   |   "
             "first two trips of each vehicle-day excluded",
             ha="center", fontsize=9, color="#555555")
    plt.tight_layout(rect=[0, 0.035, 1, 0.94])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scenario_argument(p)
    p.add_argument("--input", default=None,
                   help="trip_routing_analysis.csv (default: in the scenario's "
                        "output_trips_time_queued folder)")
    p.add_argument("--out-csv", default=None,
                   help="contingency table (default: trip_gap_correlation.csv)")
    p.add_argument("--out-png", default=None,
                   help="heatmap (default: trip_gap_correlation.png)")
    return p.parse_args()


def main():
    args = parse_args()
    scenario = load_scenario(args.scenario)

    src = args.input or os.path.join(scenario.trips_time_dir,
                                     "trip_routing_analysis.csv")
    if not os.path.exists(src):
        raise SystemExit(
            f"No analysis file at {src}\nRun calibration/captured_trips_to_geojson.py "
            "first -- it writes trip_routing_analysis.csv alongside the tracks.")

    out_csv = args.out_csv or os.path.join(scenario.output_dir,
                                           "trip_gap_correlation.csv")
    out_png = args.out_png or os.path.join(scenario.output_dir,
                                           "trip_gap_correlation.png")

    print(f"Scenario : {scenario.name}")
    print(f"Analysis : {src}")

    pairs, dropped = load_pairs(src)
    if pairs.empty:
        raise SystemExit("No trip has both a previous and a current gap.")
    print(f"\nTrip pairs analysed: {len(pairs):,}")
    for reason, n in dropped.items():
        if n:
            print(f"  excluded {n:,}: {reason}")

    result = analyse(pairs)
    observed = result["observed"]

    width = max(len(l) for l in BAND_LABELS) + 2
    print(f"\nCounts: previous gap (down) vs current gap (across)")
    print(" " * width + "".join(f"{l:>15s}" for l in BAND_LABELS) + f"{'total':>12s}")
    for i, label in enumerate(BAND_LABELS):
        print(f"  {label:{width - 2}s}"
              + "".join(f"{observed[i, j]:>15,}" for j in range(len(BAND_LABELS)))
              + f"{observed[i].sum():>12,}")
    print("  " + "total".ljust(width - 2)
          + "".join(f"{observed[:, j].sum():>15,}" for j in range(len(BAND_LABELS)))
          + f"{observed.sum():>12,}")

    print("\nRow %: given the previous gap, where does the current gap fall")
    print(" " * width + "".join(f"{l:>15s}" for l in BAND_LABELS))
    for i, label in enumerate(BAND_LABELS):
        total_i = observed[i].sum()
        print(f"  {label:{width - 2}s}"
              + "".join(f"{100.0 * observed[i, j] / total_i:>14.1f}%" if total_i else
                        f"{'-':>15s}" for j in range(len(BAND_LABELS))))

    print("\nAssociation")
    print(f"  chi-square      {result['chi2']:>12,.1f}  (dof {result['dof']}, "
          f"p = {result['p_value']:.3g})")
    print(f"  Cramer's V      {result['cramers_v']:>12.3f}  "
          f"({strength(result['cramers_v'])})")
    print(f"  Spearman rho    {result['spearman_rho']:>12.3f}  "
          f"(p = {result['spearman_p']:.3g}, on the raw durations)")

    out_csv = writable_path(out_csv)
    pd.DataFrame(to_rows(result)).to_csv(out_csv, index=False)
    out_png = writable_path(out_png)
    plot(result, f"Consecutive idle gaps - {scenario.name}", out_png)

    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
