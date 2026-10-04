# Interactive topology plot of a neuron tree with its labels and clumpiness scores (step 8 output).
# Replaces `plot_neuron_topology` of the old notebooks - same leaf-weighted hierarchical layout, but drawn as a zoomable
# plotly figure instead of one huge matplotlib image, so large neurons no longer hit the image size limit.
import os
from typing import Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joblib import delayed

from scripts.helpers import run_parallel


SWC_COLUMNS = ["node_id", "swc_type", "x", "y", "z", "r", "parent"]

######################
### Colours        ###
SURFACE, INK, INK_2, MUTED, EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"

# Score -> one hue, light to dark
SCORE_SCALE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Label -> (colour, marker symbol). The symbol repeats the colour, so a label is never told by colour alone.
LABEL_STYLES = {"pre": ("#eb6834", "triangle-up"), "post": ("#1baf7a", "square"), "post,pre": ("#4a3aa7", "diamond")}
EXTRA_STYLES = [("#eda100", "cross"), ("#e87ba4", "x"), ("#008300", "star"), ("#e34948", "hexagon")]


######################
### Helpers        ###
def _read_tree(data: Union[str, pd.DataFrame]) -> tuple:
    """Loads a tree from a DataFrame, a raw `.swc` file or a pipeline csv. Returns (DataFrame, name from the file path)."""
    if isinstance(data, pd.DataFrame):
        return data.reset_index(drop=True).copy(), ""

    path = str(data)
    if path.endswith(".swc"):
        df = pd.read_csv(path, comment="#", header=None, sep=r"\s+", names=SWC_COLUMNS)
    else:
        df = pd.read_csv(path)
        # csv files written with the pandas index (steps 2-3) start with an unnamed column
        df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])

    folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
    name = os.path.splitext(os.path.basename(path))[0]
    return df, f"{name} - {folder} tree" if folder in ("simplified", "raw") else name


def _clean_label(value) -> Union[str, None]:
    """"Pre, post" / "post,pre" -> "post,pre"; empty / nan -> None."""
    if pd.isna(value) or str(value).strip().lower() in ("", "nan", "none", "null"):
        return None
    return ",".join(sorted(part.strip().lower() for part in str(value).split(",") if part.strip()))


def _tree_layout(node_ids: list, parents: list) -> dict:
    """
    Leaf-weighted hierarchical layout, iterative (no recursion limit on deep trees).
    Every leaf gets one slot on the x axis, a parent sits above the middle of its leaves, y is the depth below the root.
    Returns a dict of lists aligned with `node_ids`: x, depth, n_leaves, n_children, is_leaf, is_root.
    """
    known = set(node_ids)
    children, roots = {}, []
    for node, parent in zip(node_ids, parents):
        if parent == -1 or parent not in known:
            roots.append(node)
        else:
            children.setdefault(parent, []).append(node)

    depth, first, last = {}, {}, {}
    next_slot = 0
    for root in roots:
        order, stack = [], [(root, 0)]
        while stack:
            node, node_depth = stack.pop()
            depth[node] = node_depth
            order.append(node)
            stack.extend((child, node_depth + 1) for child in reversed(children.get(node, [])))

        for node in order:               # pre-order -> the leaves take their slots from left to right
            if node not in children:
                first[node] = last[node] = next_slot
                next_slot += 1
        for node in reversed(order):     # children are always visited before their parent
            if node in children:
                first[node] = min(first[child] for child in children[node])
                last[node] = max(last[child] for child in children[node])
        next_slot += 2                   # gap between disconnected trees

    if len(depth) != len(node_ids):
        raise ValueError(f"> {len(node_ids) - len(depth)} nodes are not reachable from a root (cycle in the parent column?).")

    return {"x": [(first[n] + last[n]) / 2 for n in node_ids],
            "depth": [depth[n] for n in node_ids],
            "n_leaves": [last[n] - first[n] + 1 for n in node_ids],
            "n_children": [len(children.get(n, [])) for n in node_ids],
            "is_leaf": [n not in children for n in node_ids],
            "is_root": [n in set(roots) for n in node_ids]}


######################
### Plot           ###
def plot_neuron_topology(data: Union[str, pd.DataFrame],
                         labels_df: pd.DataFrame = None,
                         score: str = None,
                         score_range: tuple = None,
                         draw_node_id: bool = False,
                         title: str = None,
                         show: bool = True,
                         save_html: str = None,
                         include_plotlyjs: Union[bool, str] = "cdn",
                         height: int = 720):
    """
    Plots the hierarchical topological tree of a neuron - zoom / pan with the mouse, hover a node for its details.
    Nodes with a clumpiness score (root + internal nodes) are filled by the score, labelled nodes carry the colour
    and the shape of their label: filled marker on a leaf, outline around an internal node.

    data: str | pd.DataFrame -> step 8 csv (`8-swc_clumpiness/<simplified | raw>/<neuron_id>.csv`), any other pipeline
                                csv, a raw `.swc` file, or a DataFrame with `node_id` and `parent` columns.
    labels_df: pd.DataFrame -> optional labels, column 0 is the node id and column 1 the label text.
                               Default: the `type` column of `data`, if it has one.
    score: str -> score column shown first, e.g. "post_pre". Default: the first score column. All score columns
                  can be switched in the figure's menu.
    score_range: tuple -> (min, max) of the colour scale. Default: the range of each score column in this neuron -
                          set it to compare neurons with each other.
    draw_node_id: bool -> writes the node id next to every node. Readable on small trees only (ids are always in the hover).
    title: str -> figure title. Default: taken from the file name.
    show: bool -> True: shows the figure and returns None. False: returns the plotly figure without showing it.
    save_html: str -> path of a stand-alone html file of the figure (e.g. to view a plot made on the server).
    include_plotlyjs: bool | str -> "cdn": small html file that needs internet when opened; True: plotly is
                                    embedded (works offline, +5 MB per file).
    height: int -> figure height in pixels.
    """

    # ---------------------------------------------------------
    # 1. Tree, layout, labels and score columns
    # ---------------------------------------------------------
    df, name = _read_tree(data)
    if not {"node_id", "parent"}.issubset(df.columns):
        raise ValueError("> `data` needs the columns `node_id` and `parent`.")
    if not df["node_id"].is_unique:
        raise ValueError("> `data` has duplicated node ids.")
    if df.empty:
        raise ValueError("> `data` has no nodes.")

    node_ids = df["node_id"].astype(int).tolist()
    layout = _tree_layout(node_ids, df["parent"].astype(int).tolist())
    x = np.array(layout["x"])
    y = -np.array(layout["depth"])
    is_leaf, is_root = np.array(layout["is_leaf"]), np.array(layout["is_root"])

    if labels_df is not None and not labels_df.empty:
        label_map = dict(zip(labels_df.iloc[:, 0].astype(int), labels_df.iloc[:, 1]))
        labels = [_clean_label(label_map.get(n)) for n in node_ids]
    elif "type" in df.columns:
        labels = [_clean_label(v) for v in df["type"]]
    else:
        labels = [None] * len(df)
    labels = np.array(labels, dtype=object)

    score_cols = [c for c in df.columns if c not in SWC_COLUMNS + ["type"] and pd.api.types.is_numeric_dtype(df[c])]
    if score is not None and score not in score_cols:
        raise ValueError(f"> Score column `{score}` not found - available: {score_cols}.")
    score = score or (score_cols[0] if score_cols else None)

    # Label styles - fixed for pre / post / both, extra labels take the next free style
    styles = dict(LABEL_STYLES)
    for extra, style in zip(sorted({l for l in labels if l and l not in styles}), EXTRA_STYLES):
        styles[extra] = style

    # ---------------------------------------------------------
    # 2. Hover text of every node
    # ---------------------------------------------------------
    hover = []
    for i, node in enumerate(node_ids):
        lines = [f"<b>node {node}</b>" + (" · root" if is_root[i] else " · leaf" if is_leaf[i] else "")]
        if labels[i]:
            lines.append(f"label: {labels[i].replace(',', ' + ')}")
        lines.append(f"depth {layout['depth'][i]}" + ("" if is_leaf[i] else f" · {layout['n_leaves'][i]} leaves below"))
        if not is_leaf[i]:
            lines += [f"{c}: " + ("–" if pd.isna(df[c].iat[i]) else f"{df[c].iat[i]:.3f}") for c in score_cols]
        hover.append("<br>".join(lines))
    hover = np.array(hover, dtype=object)

    # ---------------------------------------------------------
    # 3. Traces - (trace, score column it belongs to | None = always shown)
    # ---------------------------------------------------------
    traces = []

    def add_markers(mask, name, marker, group=None, **kwargs):
        if mask.any():
            traces.append((go.Scattergl(x=x[mask], y=y[mask], mode="markers", name=name, marker=marker,
                                        hoverinfo="skip", **kwargs), group))

    # Edges first, so they sit under the nodes
    position = {n: i for i, n in enumerate(node_ids)}
    edge_x, edge_y = [], []
    for i, parent in enumerate(df["parent"].astype(int).tolist()):
        if parent in position:
            edge_x += [x[position[parent]], x[i], None]
            edge_y += [y[position[parent]], y[i], None]
    traces.append((go.Scattergl(x=edge_x, y=edge_y, mode="lines", line=dict(color=EDGE, width=1),
                                hoverinfo="skip", showlegend=False), None))

    has_label = np.array([l is not None for l in labels])
    add_markers(is_leaf & ~has_label, "leaf, no label", dict(size=4, color=MUTED))

    # Nodes without a score in the selected column. Unlabelled one-child nodes (the long chains of a raw tree) are only
    # a point on the line - drawn as a dot of the edge colour, so thousands of them do not bury the scored nodes.
    chain = (np.array(layout["n_children"]) == 1) & ~has_label & ~is_root
    hollow = dict(size=5, symbol="circle-open", color=MUTED, line=dict(width=1))
    for c in score_cols:
        no_score = ~is_leaf & df[c].isna().to_numpy()
        add_markers(no_score & chain, "chain node", dict(size=2, color=EDGE), group=c, showlegend=False)
        add_markers(no_score & ~chain, "no score", hollow, group=c)
    if not score_cols:
        add_markers(~is_leaf & chain, "chain node", dict(size=2, color=EDGE), showlegend=False)
        add_markers(~is_leaf & ~chain, "internal node", hollow)

    # Labels of internal nodes -> outline around the node
    for label, (color, symbol) in styles.items():
        add_markers(~is_leaf & (labels == label), f"{label.replace(',', ' + ')} (internal)", legendgroup=label,
                    marker=dict(size=12, symbol=f"{symbol}-open", color=color, line=dict(width=1.5)))

    # Scores -> fill of the root and the internal nodes
    for c in score_cols:
        values = df[c].to_numpy(dtype=float)
        scored = ~np.isnan(values)
        if not scored.any():
            continue
        low, high = score_range if score_range is not None else (float(values[scored].min()), float(values[scored].max()))
        if high <= low:
            high = low + 1.0
        add_markers(scored, f"{c} score", group=c, showlegend=False,
                    marker=dict(size=7, color=values[scored], colorscale=SCORE_SCALE, cmin=low, cmax=high,
                                line=dict(width=0.5, color=INK_2), showscale=True,
                                colorbar=dict(title=dict(text=f"{c} clumpiness", side="right"), thickness=12, len=0.7,
                                              outlinewidth=0, tickfont=dict(color=INK_2))))

    # Labels of leaves -> filled marker
    for label, (color, symbol) in styles.items():
        add_markers(is_leaf & (labels == label), f"{label.replace(',', ' + ')} (leaf)", legendgroup=label,
                    marker=dict(size=9, symbol=symbol, color=color, line=dict(width=0.5, color=SURFACE)))

    if draw_node_id:
        traces.append((go.Scattergl(x=x, y=y, mode="text", text=[str(n) for n in node_ids], textposition="middle right",
                                    textfont=dict(size=9, color=INK_2), hoverinfo="skip", showlegend=False), None))

    # Hover layer - one transparent marker per node, on top of everything. The hover text is stored once per node
    # instead of once per trace, which keeps the saved html small.
    traces.append((go.Scattergl(x=x, y=y, mode="markers", marker=dict(size=10, color="rgba(0,0,0,0)"),
                                text=hover, hoverinfo="text", showlegend=False), None))

    # ---------------------------------------------------------
    # 4. Figure
    # ---------------------------------------------------------
    fig = go.Figure()
    for trace, group in traces:
        trace.visible = group is None or group == score
        fig.add_trace(trace)

    n_leaves = int(is_leaf.sum())
    subtitle = f"{len(df):,} nodes · {n_leaves:,} leaves · depth {int(max(layout['depth']))}"
    subtitle += " · fill = clumpiness score, shape and colour = label" if score_cols else " · shape and colour = label"
    fig.update_layout(
        title=dict(text=f"<b>{title or ('Neuron ' + name if name else 'Neuron topology')}</b><br><sup>{subtitle}</sup>", x=0.01),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK, size=12),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=height, margin=dict(l=20, r=20, t=80, b=20),
        hovermode="closest", dragmode="pan",
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=EDGE, font=dict(color=INK, size=12)),
        legend=dict(orientation="h", yanchor="top", y=-0.01, x=0, font=dict(color=INK_2)),
    )
    for i in np.flatnonzero(is_root):
        fig.add_annotation(x=x[i], y=y[i], text="root", showarrow=False, yshift=14, font=dict(size=11, color=INK_2))

    # Menu -> one entry per score column, swaps the fill and the colour bar
    if len(score_cols) > 1:
        buttons = [dict(label=f"score: {c}", method="update",
                        args=[{"visible": [group is None or group == c for _, group in traces]}]) for c in score_cols]
        fig.update_layout(updatemenus=[dict(type="dropdown", buttons=buttons, active=score_cols.index(score), showactive=True,
                                            x=1.0, xanchor="right", y=1.0, yanchor="bottom",
                                            bgcolor=SURFACE, bordercolor=EDGE, font=dict(color=INK))])

    config = {"scrollZoom": True, "displaylogo": False}

    if save_html:
        os.makedirs(os.path.dirname(os.path.abspath(save_html)), exist_ok=True)
        fig.write_html(save_html, include_plotlyjs=include_plotlyjs, config=config)

    if show:
        try:
            fig.show(config=config)
        except ValueError:
            # plotly needs the `nbformat` package to draw inside a notebook -> without it the figure opens in the browser
            fig.show(renderer="browser", config=config)
        return None

    return fig


######################
### All neurons    ###
def _plot_worker(neuron_id: str,
                 scored_dir: str,
                 labels_dir: str,
                 output_dir: str,
                 trees: tuple,
                 overwrite: bool,
                 include_plotlyjs: Union[bool, str]) -> tuple:
    """joblib worker of `plot_scored_neurons` - never raises, returns (neuron_id, plots written, error | None)."""
    n_written = 0
    try:
        for tree in trees:
            source = os.path.join(scored_dir, tree, f"{neuron_id}.csv")
            target = os.path.join(output_dir, tree, f"{neuron_id}.html")
            if (not os.path.exists(source)) or (os.path.exists(target) and overwrite is False):
                continue

            # The raw tree file carries no labels -> taken from the labelled simplified file, joined by node id
            labels_df = None
            labels_path = os.path.join(labels_dir, f"{neuron_id}.csv")
            if tree == "raw" and os.path.exists(labels_path):
                labels_df = pd.read_csv(labels_path, index_col=0)[["node_id", "type"]]

            plot_neuron_topology(source, labels_df=labels_df, show=False, save_html=target, include_plotlyjs=include_plotlyjs)
            n_written += 1

        return neuron_id, n_written, None

    except Exception as err:
        return neuron_id, n_written, f"{type(err).__name__}: {err}"


def plot_scored_neurons(scored_dir: str,
                        labels_dir: str,
                        output_dir: str,
                        trees: tuple = ("simplified", "raw"),
                        overwrite: bool = False,
                        n_jobs: int = 1,
                        include_plotlyjs: Union[bool, str] = "cdn") -> dict:
    """
    Saves the topology plot of every scored neuron as a stand-alone html file (see `plot_neuron_topology`).

    scored_dir: str -> folder of the scored SWC files, `<scored_dir>/<simplified | raw>/<neuron_id>.csv`.
    labels_dir: str -> folder of the labelled simplified SWC csv files, source of the labels of the raw tree.
    output_dir: str -> output folder, the plots are saved as `<output_dir>/<simplified | raw>/<neuron_id>.html`.
    trees: tuple -> which trees to plot, "simplified" and / or "raw".
    overwrite: bool -> if False, existing html files are kept.
    n_jobs: int -> number of neurons plotted in parallel.
    include_plotlyjs: bool | str -> "cdn": small files that need internet when opened; True: plotly is embedded
                                    in every file (works offline, +5 MB per file).
    Returns a dict with the number of neurons, written plots and the failed neurons.
    """
    neuron_ids = sorted({f[:-4]
                         for tree in trees if os.path.isdir(os.path.join(scored_dir, tree))
                         for f in os.listdir(os.path.join(scored_dir, tree)) if f.endswith(".csv")})

    tasks = (delayed(_plot_worker)(neuron_id, scored_dir, labels_dir, output_dir, trees, overwrite, include_plotlyjs)
             for neuron_id in neuron_ids)
    results = run_parallel(tasks, total=len(neuron_ids), n_jobs=n_jobs, desc="Plotting neurons", unit=" neurons")

    failed = {neuron_id: error for neuron_id, _, error in results if error is not None}
    if failed:
        print(f"> {len(failed)} neurons failed in the plot creation (see the returned `failed`).")

    return {"neurons": len(results),
            "plots_written": int(sum(n_written for _, n_written, _ in results)),
            "failed": failed}
