"""
Which gap bins reveal the strongest association between consecutive idle gaps?

Sweeps the number of bins used for the gap before the CURRENT trip against the
number used for the gap before the PREVIOUS trip, measures the association for
every combination, and writes two heatmaps:

  1. the search grid  -- current bins across, previous bins down, coloured by
     association strength, so the best binning is visible rather than assumed
  2. the contingency table for the winning combination -- current bin across,
     previous bin down, with the number of trips in every cell

    python calibration/trip_gap_bin_search.py --scenario scenario.json

Which trips qualify is unchanged from trip_gap_correlation.py: both gaps must be
between two real journeys, so the first two trips of each vehicle-day are
excluded.


================================================================================
WHY THE SEARCH NEEDS A BIAS CORRECTION
================================================================================

Plain Cramer's V is NOT comparable across tables of different sizes. It is

        V = sqrt( chi2 / (n * (k - 1)) ),   k = min(rows, cols)

and chi2 rises with the number of cells even when the two variables are
completely independent -- every extra cell is another chance for sampling noise
to accumulate. Ranking bin schemes by plain V therefore rewards fine binning for
its own sake, and the "best" scheme would reliably be the largest table offered.

This uses the bias-corrected form (Bergsma 2013), which subtracts the
association expected from noise alone before rescaling:

        phi2      = chi2 / n
        phi2_corr = max(0, phi2 - (r-1)(c-1)/(n-1))
        r_corr    = r - (r-1)^2/(n-1)
        c_corr    = c - (c-1)^2/(n-1)
        V_corr    = sqrt( phi2_corr / min(r_corr - 1, c_corr - 1) )

With that, a scheme only scores well if it finds structure a coarser one missed.
If the variables really are independent, V_corr sits near zero at every size,
which is the behaviour that makes the search trustworthy.

--strategy chooses where the bin edges go:

    quantile  equal numbers of trips per bin (default). Puts the cut points
              where the data actually is, so no bin is starved -- generally the
              most sensitive to real structure.
    log       equal-width in log time, which suits a quantity spanning seconds
              to hours far better than equal-width in minutes.
    linear    equal-width in minutes, capped at --linear-max-min with an
              open-ended top bin. Easy to describe, but at these gap
              distributions it puts almost everything in the first bin.

The Spearman correlation on the raw durations is printed alongside as a
reference: it uses no bins at all, so it is the association available before any
binning throws information away. If the best V_corr is far below it, the bins
are the limitation; if they agree, binning is not costing anything.
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
from trip_gap_correlation import load_pairs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario_config import load_scenario, add_scenario_argument, writable_path

DEFAULT_MIN_BINS = 2
DEFAULT_MAX_BINS = 8
DEFAULT_LINEAR_MAX_MIN = 60.0


# ============================================================
# BINNING
# ============================================================

def bin_edges(values, n_bins, strategy, linear_max_min):
    """
    Interior cut points, in minutes. The outer edges are implicit (-inf, +inf),
    so no trip is ever excluded by the choice of binning.
    """
    if n_bins < 2:
        return np.array([])

    if strategy == "quantile":
        qs = np.linspace(0, 100, n_bins + 1)[1:-1]
        edges = np.percentile(values, qs)
    elif strategy == "log":
        # log1p keeps zero-length gaps, which are common, on the scale.
        lo, hi = np.log1p(values.min()), np.log1p(np.percentile(values, 99.5))
        edges = np.expm1(np.linspace(lo, hi, n_bins + 1)[1:-1])
    elif strategy == "linear":
        edges = np.linspace(0, linear_max_min, n_bins)[1:]
    else:
        raise ValueError(strategy)

    # Ties collapse bins; keeping them would create empty columns that chi2
    # cannot use and that would misreport the effective table size.
    edges = np.unique(np.round(edges, 6))
    # An edge sitting on the minimum is worse than a tie: it defines a bin that
    # nothing can fall into (values below the minimum). Over a sixth of these
    # gaps are exactly zero, so quantile cuts land there routinely.
    return edges[edges > values.min()]


def assign(values, edges):
    return np.searchsorted(edges, values, side="right")


def label_bins(edges):
    labels = []
    lo = 0.0
    for e in edges:
        labels.append(f"{lo:g}-{e:g}")
        lo = e
    labels.append(f">={lo:g}")
    return labels


# ============================================================
# ASSOCIATION
# ============================================================

def contingency(prev_idx, cur_idx, n_prev, n_cur):
    table = np.zeros((n_prev, n_cur), dtype=int)
    for i, j in zip(prev_idx, cur_idx):
        table[i, j] += 1
    return table


def corrected_cramers_v(table):
    """
    Bias-corrected Cramer's V, so tables of different sizes are comparable.
    Returns (v_corrected, v_plain, chi2, p_value, dof) -- or NaNs when the table
    is degenerate (a single occupied row or column carries no association).
    """
    keep_r = table.sum(axis=1) > 0
    keep_c = table.sum(axis=0) > 0
    compact = table[np.ix_(keep_r, keep_c)]
    if min(compact.shape) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0

    chi2, p, dof, _expected = stats.chi2_contingency(compact)
    n = compact.sum()
    r, c = compact.shape

    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    c_corr = c - (c - 1) ** 2 / (n - 1)
    denominator = min(r_corr - 1, c_corr - 1)
    v_corr = float(np.sqrt(phi2_corr / denominator)) if denominator > 0 else float("nan")
    v_plain = float(np.sqrt(phi2 / min(r - 1, c - 1)))
    return v_corr, v_plain, float(chi2), float(p), int(dof)


def sweep(pairs, min_bins, max_bins, strategy, linear_max_min):
    """Every (previous bins, current bins) combination in the range."""
    prev_values = pairs["prev_gap_min"].to_numpy()
    cur_values = pairs["gap_min"].to_numpy()

    counts = list(range(min_bins, max_bins + 1))
    grid_v = np.full((len(counts), len(counts)), np.nan)
    grid_plain = np.full((len(counts), len(counts)), np.nan)
    grid_n = np.zeros((len(counts), len(counts)), dtype=int)
    rows = []

    for a, n_prev in enumerate(counts):
        prev_edges = bin_edges(prev_values, n_prev, strategy, linear_max_min)
        prev_idx = assign(prev_values, prev_edges)
        for b, n_cur in enumerate(counts):
            cur_edges = bin_edges(cur_values, n_cur, strategy, linear_max_min)
            cur_idx = assign(cur_values, cur_edges)
            table = contingency(prev_idx, cur_idx,
                                len(prev_edges) + 1, len(cur_edges) + 1)
            v_corr, v_plain, chi2, p, dof = corrected_cramers_v(table)
            grid_v[a, b] = v_corr
            grid_plain[a, b] = v_plain
            grid_n[a, b] = int(table.sum())
            rows.append({
                "previous_bins": n_prev, "current_bins": n_cur,
                "previous_bins_effective": len(prev_edges) + 1,
                "current_bins_effective": len(cur_edges) + 1,
                "trips": int(table.sum()),
                "cramers_v_corrected": (round(v_corr, 5)
                                        if np.isfinite(v_corr) else None),
                "cramers_v_plain": (round(v_plain, 5)
                                    if np.isfinite(v_plain) else None),
                "chi2": round(chi2, 2) if np.isfinite(chi2) else None,
                "p_value": p if np.isfinite(p) else None,
                "dof": dof,
                "previous_edges_min": ";".join(f"{e:g}" for e in prev_edges),
                "current_edges_min": ";".join(f"{e:g}" for e in cur_edges),
            })
    return counts, grid_v, grid_plain, grid_n, rows


# ============================================================
# PLOTS
# ============================================================

def plot_search(counts, grid_v, grid_n, strategy, spearman, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(grid_v, cmap="viridis", origin="upper")

    finite = grid_v[np.isfinite(grid_v)]
    mid = finite.min() + 0.55 * (finite.max() - finite.min()) if finite.size else 0
    for a in range(len(counts)):
        for b in range(len(counts)):
            v = grid_v[a, b]
            if not np.isfinite(v):
                ax.text(b, a, "-", ha="center", va="center", color="#888888")
                continue
            ax.text(b, a, f"V={v:.4f}\nn={grid_n[a, b]:,}", ha="center",
                    va="center", fontsize=8,
                    color="white" if v < mid else "#101010")

    best = np.unravel_index(np.nanargmax(grid_v), grid_v.shape)
    ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1,
                               fill=False, edgecolor="#e74c3c", lw=3))

    ax.set_xticks(range(len(counts)))
    ax.set_yticks(range(len(counts)))
    ax.set_xticklabels(counts)
    ax.set_yticklabels(counts)
    ax.set_xlabel("number of bins for the gap before the CURRENT trip", fontsize=11)
    ax.set_ylabel("number of bins for the gap before the PREVIOUS trip", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    fig.colorbar(im, ax=ax, fraction=0.046, label="bias-corrected Cramer's V")
    fig.text(0.5, 0.015,
             f"{strategy} bin edges   |   red box = strongest association   |   "
             f"Spearman rho on the raw durations = {spearman:+.3f} (no binning)",
             ha="center", fontsize=9, color="#555555")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_best_table(table, prev_labels, cur_labels, stats_line, title, out_path):
    fig, ax = plt.subplots(figsize=(1.6 * len(cur_labels) + 4,
                                    1.1 * len(prev_labels) + 3.5))
    row_tot = table.sum(axis=1, keepdims=True)
    row_pct = np.divide(100.0 * table, row_tot, where=row_tot > 0,
                        out=np.zeros(table.shape, dtype=float))
    im = ax.imshow(row_pct, cmap="Blues", vmin=0)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, f"{table[i, j]:,}\n{row_pct[i, j]:.0f}%",
                    ha="center", va="center", fontsize=9,
                    color="white" if row_pct[i, j] > row_pct.max() * 0.6 else "#2c3e50")

    ax.set_xticks(range(len(cur_labels)))
    ax.set_yticks(range(len(prev_labels)))
    ax.set_xticklabels(cur_labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels(prev_labels, fontsize=9)
    ax.set_xlabel("gap before CURRENT trip (minutes)", fontsize=11)
    ax.set_ylabel("gap before PREVIOUS trip (minutes)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    fig.colorbar(im, ax=ax, fraction=0.046, label="% of the previous-gap bin")
    fig.text(0.5, 0.015, stats_line, ha="center", fontsize=9, color="#555555")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scenario_argument(p)
    p.add_argument("--input", default=None)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--out-png", default=None, help="the search grid heatmap")
    p.add_argument("--out-best-png", default=None,
                   help="contingency heatmap for the winning bin combination")
    p.add_argument("--strategy", choices=("quantile", "log", "linear"),
                   default="quantile", help="where the bin edges go (default quantile)")
    p.add_argument("--min-bins", type=int, default=DEFAULT_MIN_BINS)
    p.add_argument("--max-bins", type=int, default=DEFAULT_MAX_BINS)
    p.add_argument("--linear-max-min", type=float, default=DEFAULT_LINEAR_MAX_MIN,
                   help="top edge for --strategy linear (default 60)")
    return p.parse_args()


def main():
    args = parse_args()
    scenario = load_scenario(args.scenario)

    src = args.input or os.path.join(scenario.trips_time_dir,
                                     "trip_routing_analysis.csv")
    if not os.path.exists(src):
        raise SystemExit(f"No analysis file at {src}")

    out_csv = args.out_csv or os.path.join(scenario.output_dir,
                                           "trip_gap_bin_search.csv")
    out_png = args.out_png or os.path.join(scenario.output_dir,
                                           "trip_gap_bin_search.png")
    out_best = args.out_best_png or os.path.join(scenario.output_dir,
                                                 "trip_gap_bin_best.png")

    print(f"Scenario : {scenario.name}")
    print(f"Strategy : {args.strategy} bin edges, {args.min_bins}-{args.max_bins} bins")

    pairs, dropped = load_pairs(src)
    print(f"\nTrip pairs analysed: {len(pairs):,}")
    for reason, n in dropped.items():
        if n:
            print(f"  excluded {n:,}: {reason}")

    rho, rho_p = stats.spearmanr(pairs["prev_gap_min"], pairs["gap_min"])
    counts, grid_v, grid_plain, grid_n, rows = sweep(
        pairs, args.min_bins, args.max_bins, args.strategy, args.linear_max_min)

    out_csv = writable_path(out_csv)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    best = np.unravel_index(np.nanargmax(grid_v), grid_v.shape)
    n_prev, n_cur = counts[best[0]], counts[best[1]]
    v_best = grid_v[best]

    print(f"\nbias-corrected Cramer's V   (rows = previous bins, cols = current bins)")
    print("        " + "".join(f"{c:>10d}" for c in counts))
    for a, rp in enumerate(counts):
        print(f"  {rp:>4d}  " + "".join(
            f"{grid_v[a, b]:>10.4f}" if np.isfinite(grid_v[a, b]) else f"{'-':>10s}"
            for b in range(len(counts))))

    print(f"\nStrongest: previous {n_prev} bins x current {n_cur} bins   "
          f"V_corrected = {v_best:.4f}  (plain V = {grid_plain[best]:.4f})")
    print(f"Range across the whole grid: {np.nanmin(grid_v):.4f} to {np.nanmax(grid_v):.4f}")
    print(f"Spearman rho on the raw durations: {rho:+.4f} (p = {rho_p:.3g}) "
          "-- the association available without binning")

    # Rebuild the winning table for the second figure.
    prev_values = pairs["prev_gap_min"].to_numpy()
    cur_values = pairs["gap_min"].to_numpy()
    prev_edges = bin_edges(prev_values, n_prev, args.strategy, args.linear_max_min)
    cur_edges = bin_edges(cur_values, n_cur, args.strategy, args.linear_max_min)
    table = contingency(assign(prev_values, prev_edges), assign(cur_values, cur_edges),
                        len(prev_edges) + 1, len(cur_edges) + 1)

    print(f"\nWinning bin edges, minutes")
    print(f"  previous: {label_bins(prev_edges)}")
    print(f"  current : {label_bins(cur_edges)}")

    plot_search(counts, grid_v, grid_n, args.strategy, rho,
                f"Association between consecutive gaps - {scenario.name}", out_png)
    stats_line = (f"n = {table.sum():,} trips   |   bias-corrected Cramer's V = "
                  f"{v_best:.4f}   |   {args.strategy} bin edges   |   "
                  "first two trips of each vehicle-day excluded")
    plot_best_table(table, label_bins(prev_edges), label_bins(cur_edges),
                    stats_line,
                    f"Strongest binning: {n_prev} x {n_cur} - {scenario.name}",
                    writable_path(out_best))

    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_best}")


if __name__ == "__main__":
    main()
