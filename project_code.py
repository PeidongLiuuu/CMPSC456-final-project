"""
CMPSC 456 – Numerical Interpolation Study on Housing Price Data
Peidong Liu

Experiments:
  1. Equispaced vs Chebyshev barycentric interpolation
  2. Runge-type oscillation analysis across polynomial degrees
  3. Cubic spline comparison (natural, clamped, not-a-knot)
  4. L2 vs L∞ error norm summary
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – saves plots as files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator, CubicSpline


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def normalize(v, lo, hi):
    return 2 * (v - lo) / (hi - lo) - 1


def errors(pred, ref):
    """Return (L∞, L2 RMS) errors between pred and ref arrays."""
    diff = pred - ref
    return np.linalg.norm(diff, np.inf), np.linalg.norm(diff) / np.sqrt(len(diff))


def chebyshev_nodes(n):
    """Return n sorted Chebyshev nodes of the first kind on [-1, 1]."""
    k = np.arange(1, n + 1)
    return np.sort(np.cos((2 * k - 1) * np.pi / (2 * n)))


def barycentric_interpolant(x_nodes, x_data, y_data):
    """Build a BarycentricInterpolator by sampling y_data at x_nodes."""
    y_nodes = np.interp(x_nodes, x_data, y_data)
    return BarycentricInterpolator(x_nodes, y_nodes)


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_data(csv_path="Housing_Price_Data.csv", n_samples=60):
    """
    Load and preprocess housing data.

    Returns x_data, y_data (both normalized to [-1, 1]) and a dense
    evaluation grid x_eval.
    """
    df = pd.read_csv(csv_path)

    binary_cols = ["mainroad", "guestroom", "basement",
                   "hotwaterheating", "airconditioning", "prefarea"]
    for col in binary_cols:
        df[col] = (df[col].str.strip().str.lower() == "yes").astype(int)

    furnish_map = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
    df["furnishingstatus"] = (
        df["furnishingstatus"].str.strip().str.lower().map(furnish_map)
    )

    df_sorted = df.sort_values("area").reset_index(drop=True)
    idx = np.linspace(0, len(df_sorted) - 1, n_samples, dtype=int)

    x_raw = df_sorted.loc[idx, "area"].values.astype(float)
    y_raw = df_sorted.loc[idx, "price"].values.astype(float)

    x_data = normalize(x_raw, x_raw.min(), x_raw.max())
    y_data = normalize(y_raw, y_raw.min(), y_raw.max())
    x_eval = np.linspace(-1, 1, 500)

    return x_data, y_data, x_eval


# ─────────────────────────────────────────────
# Experiment 1 – Equispaced vs Chebyshev (n=15)
# ─────────────────────────────────────────────

def experiment_equi_vs_cheb(x_data, y_data, x_eval, n_nodes=15):
    y_ref = np.interp(x_eval, x_data, y_data)

    poly_equi = barycentric_interpolant(np.linspace(-1, 1, n_nodes), x_data, y_data)
    poly_cheb = barycentric_interpolant(chebyshev_nodes(n_nodes), x_data, y_data)

    pred_equi = poly_equi(x_eval)
    pred_cheb = poly_cheb(x_eval)

    linf_e, l2_e = errors(pred_equi, y_ref)
    linf_c, l2_c = errors(pred_cheb, y_ref)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Equispaced vs Chebyshev Barycentric Interpolation (n={n_nodes} nodes)",
        fontsize=13,
    )

    ax = axes[0]
    ax.scatter(x_data, y_data, s=18, alpha=0.4, color="gray", label="Data")
    ax.plot(x_eval, pred_equi, "r-", lw=1.8, label=f"Equispaced  L∞={linf_e:.4f}")
    ax.plot(x_eval, pred_cheb, "b-", lw=1.8, label=f"Chebyshev   L∞={linf_c:.4f}")
    ax.set_xlabel("Area (normalized)")
    ax.set_ylabel("Price (normalized)")
    ax.set_title("Interpolation Curves")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(x_eval, np.abs(pred_equi - y_ref), "r-", lw=1.5, label="Equispaced |error|")
    ax.plot(x_eval, np.abs(pred_cheb - y_ref), "b-", lw=1.5, label="Chebyshev  |error|")
    ax.set_xlabel("Area (normalized)")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Pointwise Absolute Error")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot1_equi_vs_cheb.png", dpi=150)
    plt.close()

    print("=== Experiment 1: Equispaced vs Chebyshev ===")
    print(f"  Equispaced  L∞={linf_e:.6f}  L2={l2_e:.6f}")
    print(f"  Chebyshev   L∞={linf_c:.6f}  L2={l2_c:.6f}")

    return poly_equi, poly_cheb


# ─────────────────────────────────────────────
# Experiment 2 – Runge-type oscillation sweep
# ─────────────────────────────────────────────

def experiment_runge(x_data, y_data, x_eval, degrees=None):
    if degrees is None:
        degrees = [4, 6, 8, 10, 12, 15, 18, 20]

    y_ref = np.interp(x_eval, x_data, y_data)
    results = {"linf_equi": [], "linf_cheb": [], "l2_equi": [], "l2_cheb": []}

    for n in degrees:
        pe = barycentric_interpolant(np.linspace(-1, 1, n), x_data, y_data)
        pc = barycentric_interpolant(chebyshev_nodes(n), x_data, y_data)

        le, l2e = errors(pe(x_eval), y_ref)
        lc, l2c = errors(pc(x_eval), y_ref)

        results["linf_equi"].append(le)
        results["linf_cheb"].append(lc)
        results["l2_equi"].append(l2e)
        results["l2_cheb"].append(l2c)

    # Error-vs-degree plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Runge-Type Oscillation: Error vs Polynomial Degree", fontsize=13)

    ax = axes[0]
    ax.plot(degrees, results["linf_equi"], "r-o", label="Equispaced  L∞")
    ax.plot(degrees, results["linf_cheb"], "b-s", label="Chebyshev   L∞")
    ax.set_xlabel("Polynomial Degree (n nodes)")
    ax.set_ylabel("L∞ Error")
    ax.set_title("L∞ Error vs Degree")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(degrees, results["l2_equi"], "r-o", label="Equispaced  L2")
    ax.plot(degrees, results["l2_cheb"], "b-s", label="Chebyshev   L2")
    ax.set_xlabel("Polynomial Degree (n nodes)")
    ax.set_ylabel("L2 Error (RMS)")
    ax.set_title("L2 (RMS) Error vs Degree")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot2_runge_analysis.png", dpi=150)
    plt.close()

    # High-degree oscillation visual (n=20)
    n_hi = 20
    pe_hi = barycentric_interpolant(np.linspace(-1, 1, n_hi), x_data, y_data)
    pc_hi = barycentric_interpolant(chebyshev_nodes(n_hi), x_data, y_data)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(x_data, y_data, s=15, alpha=0.35, color="gray", label="Data")
    ax.plot(x_eval, y_ref, "k--", lw=1.2, label="True (interpolated reference)")
    ax.plot(x_eval, pe_hi(x_eval), "r-", lw=1.5, label=f"Equispaced n={n_hi}")
    ax.plot(x_eval, pc_hi(x_eval), "b-", lw=1.5, label=f"Chebyshev  n={n_hi}")
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel("Area (normalized)")
    ax.set_ylabel("Price (normalized)")
    ax.set_title(f"Runge Oscillation Visualization at Degree {n_hi - 1}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot3_runge_visual.png", dpi=150)
    plt.close()

    print("\n=== Experiment 2: Runge Oscillation Analysis ===")
    print(f"{'n':>4}  {'Equi L∞':>12}  {'Cheb L∞':>12}  {'Equi L2':>12}  {'Cheb L2':>12}")
    for i, n in enumerate(degrees):
        print(
            f"{n:>4}  {results['linf_equi'][i]:>12.6f}  {results['linf_cheb'][i]:>12.6f}"
            f"  {results['l2_equi'][i]:>12.6f}  {results['l2_cheb'][i]:>12.6f}"
        )


# ─────────────────────────────────────────────
# Experiment 3 – Cubic spline comparison
# ─────────────────────────────────────────────

def experiment_splines(x_data, y_data, x_eval):
    y_ref = np.interp(x_eval, x_data, y_data)

    # Deduplicate x for CubicSpline (requires strictly increasing x)
    df_sp = pd.DataFrame({"x": x_data, "y": y_data})
    df_sp = df_sp.groupby("x", as_index=False).mean().sort_values("x")
    x_sp, y_sp = df_sp["x"].values, df_sp["y"].values

    splines = [
        CubicSpline(x_sp, y_sp, bc_type="natural"),
        CubicSpline(x_sp, y_sp, bc_type=((1, 0.0), (1, 0.0))),
        CubicSpline(x_sp, y_sp, bc_type="not-a-knot"),
    ]
    sp_names = ["Natural", "Clamped (slope=0)", "Not-a-Knot"]
    colors   = ["green", "purple", "orange"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cubic Spline Comparison: Natural vs Clamped vs Not-a-Knot", fontsize=13)

    ax = axes[0]
    ax.scatter(x_data, y_data, s=18, alpha=0.4, color="gray", label="Data")
    for cs, name, c in zip(splines, sp_names, colors):
        ax.plot(x_eval, cs(x_eval), color=c, lw=1.8, label=name)
    ax.set_xlabel("Area (normalized)")
    ax.set_ylabel("Price (normalized)")
    ax.set_title("Spline Curves")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for cs, name, c in zip(splines, sp_names, colors):
        ax.plot(x_eval, np.abs(cs(x_eval) - y_ref), color=c, lw=1.5, label=name)
    ax.set_xlabel("Area (normalized)")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Pointwise Absolute Error")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot4_splines.png", dpi=150)
    plt.close()

    print("\n=== Experiment 3: Cubic Spline Comparison ===")
    print(f"{'Spline':<22}  {'L∞ Error':>12}  {'L2 Error':>12}")
    for cs, name in zip(splines, sp_names):
        linf, l2 = errors(cs(x_eval), y_ref)
        print(f"{name:<22}  {linf:>12.6f}  {l2:>12.6f}")

    return splines, sp_names


# ─────────────────────────────────────────────
# Experiment 4 – L2 vs L∞ summary
# ─────────────────────────────────────────────

def experiment_error_summary(x_eval, y_ref, poly_equi, poly_cheb, splines, sp_names):
    all_labels = ["Equispaced Bary (n=15)", "Chebyshev Bary (n=15)"] + sp_names
    all_preds  = [poly_equi(x_eval), poly_cheb(x_eval)] + [cs(x_eval) for cs in splines]

    linf_vals = [errors(p, y_ref)[0] for p in all_preds]
    l2_vals   = [errors(p, y_ref)[1] for p in all_preds]

    x_pos = np.arange(len(all_labels))
    width = 0.35

    _, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x_pos - width / 2, linf_vals, width, label="L∞ Error",      color="steelblue")
    ax.bar(x_pos + width / 2, l2_vals,   width, label="L2 Error (RMS)", color="coral")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Error")
    ax.set_title("L2 vs L∞ Error Comparison Across All Methods")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot5_error_comparison.png", dpi=150)
    plt.close()

    print("\n=== Experiment 4: L2 vs L∞ Summary ===")
    print(f"{'Method':<28}  {'L∞ Error':>12}  {'L2 Error':>12}")
    for label, linf, l2 in zip(all_labels, linf_vals, l2_vals):
        print(f"{label:<28}  {linf:>12.6f}  {l2:>12.6f}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    x_data, y_data, x_eval = load_data()
    y_ref = np.interp(x_eval, x_data, y_data)

    poly_equi, poly_cheb          = experiment_equi_vs_cheb(x_data, y_data, x_eval)
    experiment_runge(x_data, y_data, x_eval)
    splines, sp_names             = experiment_splines(x_data, y_data, x_eval)
    experiment_error_summary(x_eval, y_ref, poly_equi, poly_cheb, splines, sp_names)

    print(
        "\nDone. Plots saved: plot1_equi_vs_cheb.png, plot2_runge_analysis.png,\n"
        "                  plot3_runge_visual.png, plot4_splines.png, plot5_error_comparison.png"
    )


if __name__ == "__main__":
    main()
