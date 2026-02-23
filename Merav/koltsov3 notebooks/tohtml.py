# results/viz_koltsov3_d2_by_k.py
# Interactive HTML visualization for results/koltsov3_fullgraph.csv
# Plots d=2 data, with one trace per k, over n (x-axis).
#
# Run:
#   python results/viz_koltsov3_d2_by_k.py
# Output:
#   results/koltsov3_d2_by_k.html

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


CSV_PATH = Path("results/koltsov3_fullgraph.csv")
OUT_HTML = Path("results/koltsov3_d2_by_k.html")


def _parse_growth_list(x):
    """Safely parse the 'growth' column that looks like '[1, 3, 5, ...]'."""
    if pd.isna(x):
        return None
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if not s:
        return None
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else None
    except Exception:
        return None


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH.resolve()}")

    df = pd.read_csv(CSV_PATH)

    # Basic cleanup + types
    for col in ["d", "k", "n", "diameter", "last_layer_size", "total_states"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "growth" in df.columns:
        df["growth_list"] = df["growth"].apply(_parse_growth_list)
        df["growth_len"] = df["growth_list"].apply(lambda v: len(v) if isinstance(v, list) else None)
        df["growth_max"] = df["growth_list"].apply(lambda v: max(v) if isinstance(v, list) and len(v) else None)
    else:
        df["growth_len"] = None
        df["growth_max"] = None

    if "central" not in df.columns:
        df["central"] = None

    # Filter to coset=FullGraph (if present) and d=2
    d2 = df.copy()
    if "coset" in d2.columns:
        d2 = d2[d2["coset"].astype(str) == "FullGraph"]
    d2 = d2[d2["d"] == 2].copy()

    if d2.empty:
        raise ValueError("No rows found for d=2 (and coset=FullGraph if that column exists).")

    # Sort so lines look nice
    d2 = d2.sort_values(["k", "n"], kind="stable")

    # Metrics to toggle in dropdown
    metrics = [
        ("Diameter", "diameter"),
        ("Last layer size", "last_layer_size"),
        ("Total states", "total_states"),
        ("Growth max", "growth_max"),
        ("Growth length", "growth_len"),
    ]
    # Keep only metrics that exist / are meaningful
    metrics = [(label, col) for (label, col) in metrics if col in d2.columns]

    # Build one set of traces per metric; we’ll toggle visibility with a dropdown
    fig = go.Figure()

    ks = sorted([int(k) for k in d2["k"].dropna().unique()])
    trace_indices_by_metric = {}

    for metric_i, (metric_label, ycol) in enumerate(metrics):
        trace_indices = []
        for k in ks:
            sub = d2[d2["k"] == k]
            if sub.empty:
                continue

            # Hover info (kept consistent regardless of metric)
            hover = (
                "<b>k=%{customdata[0]}</b><br>"
                "n=%{x}<br>"
                f"{metric_label}=%{{y}}<br>"
                "diameter=%{customdata[1]}<br>"
                "last_layer_size=%{customdata[2]}<br>"
                "total_states=%{customdata[3]}<br>"
                "growth_max=%{customdata[4]}<br>"
                "growth_len=%{customdata[5]}<br>"
                "central=%{customdata[6]}<br>"
                "<extra></extra>"
            )

            customdata = pd.DataFrame(
                {
                    "k": sub["k"].astype("Int64"),
                    "diameter": sub.get("diameter"),
                    "last_layer_size": sub.get("last_layer_size"),
                    "total_states": sub.get("total_states"),
                    "growth_max": sub.get("growth_max"),
                    "growth_len": sub.get("growth_len"),
                    "central": sub.get("central"),
                }
            ).to_numpy()

            fig.add_trace(
                go.Scatter(
                    x=sub["n"],
                    y=sub[ycol],
                    mode="lines+markers",
                    name=f"k={k}",
                    visible=(metric_i == 0),  # only first metric visible initially
                    customdata=customdata,
                    hovertemplate=hover,
                )
            )
            trace_indices.append(len(fig.data) - 1)

        trace_indices_by_metric[ycol] = trace_indices

    # Dropdown to toggle which metric is shown
    buttons = []
    total_traces = len(fig.data)
    for metric_label, ycol in metrics:
        visible = [False] * total_traces
        for idx in trace_indices_by_metric.get(ycol, []):
            visible[idx] = True

        buttons.append(
            dict(
                label=metric_label,
                method="update",
                args=[
                    {"visible": visible},
                    {
                        "yaxis": {"title": metric_label},
                        "title": f"Koltsov3 FullGraph (d=2): {metric_label} vs n, grouped by k",
                    },
                ],
            )
        )

    fig.update_layout(
        title=f"Koltsov3 FullGraph (d=2): {metrics[0][0]} vs n, grouped by k",
        xaxis_title="n",
        yaxis_title=metrics[0][0],
        hovermode="closest",
        legend_title="k",

        # NEW: move legend down a bit so it doesn't collide with the dropdown
        legend=dict(
            x=1.02,
            y=0.92,
            xanchor="left",
            yanchor="top",
        ),

        updatemenus=[
            dict(
                type="dropdown",
                x=1.02,
                y=1.08,          # NEW: move dropdown up
                xanchor="left",
                yanchor="top",
                buttons=buttons,
            )
        ],

        margin=dict(l=70, r=240, t=120, b=60),  # NEW: extra top margin for dropdown
    )


    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUT_HTML), include_plotlyjs="cdn", full_html=True)
    print(f"Wrote: {OUT_HTML.resolve()}")


if __name__ == "__main__":
    main()
