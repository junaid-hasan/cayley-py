# results/heatmap_koltsov3_nL_topdf.py
# Writes a PDF heatmap for a *larger* n from results/koltsov3_fullgraph.csv
#
# Default heatmap value: diameter
# Axes: rows = d, cols = k
#
# This version automatically chooses an n near the high end of what exists in the CSV
# (you can override via N_TARGET).
#
# Run:
#   python results/heatmap_koltsov3_nL_topdf.py
#
# Output:
#   results/koltsov3_heatmap_n<chosen>.pdf

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = Path("results/koltsov3_fullgraph.csv")

# If you want to force a specific n, set it to an int (e.g., 12) instead of None.
N_TARGET: int | None = None

VALUE_COL = "diameter"  # change to "last_layer_size" or "total_states" if you want


def choose_large_n(df: pd.DataFrame) -> int:
    """Pick a large n that exists in the data (uses ~80th percentile to avoid sparse extreme tail)."""
    n_vals = sorted({int(x) for x in df["n"].dropna().unique()})
    if not n_vals:
        raise ValueError("No n values found in CSV.")
    if len(n_vals) == 1:
        return n_vals[0]
    idx = int(round(0.80 * (len(n_vals) - 1)))
    return n_vals[idx]


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH.resolve()}")

    df = pd.read_csv(CSV_PATH)

    # Ensure numeric types
    for col in ["d", "k", "n", "diameter", "last_layer_size", "total_states"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if VALUE_COL not in df.columns:
        raise ValueError(f"Column '{VALUE_COL}' not found in CSV. Available: {list(df.columns)}")

    # Optional filter: coset == FullGraph (if column exists)
    sub = df.copy()
    if "coset" in sub.columns:
        sub = sub[sub["coset"].astype(str) == "FullGraph"]

    # Pick a larger n automatically unless user overrides
    n_target = int(N_TARGET) if N_TARGET is not None else choose_large_n(sub)

    # Filter to n=n_target
    sub = sub[sub["n"] == n_target].copy()
    if sub.empty:
        raise ValueError(
            f"No rows found for n={n_target}"
            + (" and coset=FullGraph" if "coset" in df.columns else "")
            + "."
        )

    # Build (d x k) matrix of the chosen value
    # If duplicates exist, take the mean (you can change to 'max' or 'min')
    pivot = (
        sub.pivot_table(index="d", columns="k", values=VALUE_COL, aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )

    data = pivot.to_numpy(dtype=float)
    d_vals = pivot.index.to_list()
    k_vals = pivot.columns.to_list()

    # Output path includes chosen n
    out_pdf = Path(f"results/koltsov3_heatmap_n{n_target}.pdf")

    # Plot (bigger figure so pattern is clearer at larger grids)
    fig_w = max(8, 0.75 * len(k_vals))
    fig_h = max(6, 0.65 * len(d_vals))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, aspect="auto", interpolation="nearest")

    ax.set_title(f"Koltsov3 Heatmap (n={n_target}) — {VALUE_COL}  (rows=d, cols=k)")
    ax.set_xlabel("k")
    ax.set_ylabel("d")

    ax.set_xticks(np.arange(len(k_vals)))
    ax.set_xticklabels([str(int(x)) if pd.notna(x) else "" for x in k_vals], rotation=45, ha="right")

    ax.set_yticks(np.arange(len(d_vals)))
    ax.set_yticklabels([str(int(x)) if pd.notna(x) else "" for x in d_vals])

    # Annotate cells (optional): for large grids, this can get cluttered.
    # Set ANNOTATE=False if the PDF looks too busy.
    ANNOTATE = True
    if ANNOTATE:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isfinite(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(VALUE_COL)

    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_pdf.resolve()}")


if __name__ == "__main__":
    main()
