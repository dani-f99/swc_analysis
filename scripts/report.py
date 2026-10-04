# HTML report for the notebook test run - turns `df_test` / `tracebacks` (last cell of `dev_notebook.ipynb`)
# into one self-contained html file (no external assets) in the run's report folder.
import os
import html
import time
import numbers
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd


######################
### Labels         ###
STEP_LABELS = {"step01": "Step 1 - labels file",
               "step02": "Step 2 - simplify SWC",
               "step03": "Step 3 - join labels",
               "step04": "Step 4 - create JSON",
               "step05": "Step 5 - divide sub-trees",
               "step06": "Step 6 - clumpiness scores",
               "step07": "Step 7 - unified clumpiness file",
               "step08": "Step 8 - scores on the SWC nodes",
               "step09": "Step 9 - topology plots"}

CHECK_LABELS = {"ok_single_root":  ("Single root",  "Step 2 - simplified tree has exactly one root"),
                "ok_unique_nodes": ("Unique nodes", "Step 3 - one row per node"),
                "ok_labels":       ("Labels match", "Step 3 - every node carries exactly its own labels"),
                "ok_json_nodes":   ("JSON nodes",   "Step 4 - every csv node is in the json tree"),
                "ok_json_labels":  ("JSON labels",  "Step 4 - json labels equal the step 3 csv labels"),
                "ok_subtrees":     ("Sub-trees",    "Step 5 - one sub-tree per internal node, root excluded"),
                "ok_clumpiness":   ("Clumpiness",   "Step 6 - one clumpiness csv per json tree (main tree + every sub-tree)")}

COLUMN_LABELS = {"n_nodes_simplified":     "Nodes (simplified)",
                 "labeled_nodes_raw":      "Labeled raw",
                 "labeled_nodes_expected": "Labeled expected",
                 "labeled_nodes_found":    "Labeled found",
                 "label_mismatches":       "Label mismatches"}

# verdict -> (status colour, label); the icon comes with the colour, so state is never colour alone
VERDICTS = {"passed":       ("good",     "Passed"),
            "check_failed": ("serious",  "Check failed"),
            "failed":       ("critical", "Step failed"),
            "check_error":  ("warning",  "Check error")}

RUN_STATUS = {"ok":      ("good",     "Done"),
              "issues":  ("serious",  "Done with failures"),
              "failed":  ("critical", "Failed"),
              "not run": ("warning",  "Not run")}

ICONS = {"good": "&#10003;", "warning": "!", "serious": "&#10005;", "critical": "&#10005;"}


######################
### Style / script ###
_CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --s7: #4a3aa7; --s8: #e34948;
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 32px 16px 48px; }
h1 { font-size: 26px; line-height: 1.2; margin: 2px 0 8px; }
h2 { font-size: 15px; margin: 0 0 2px; }
.eyebrow { margin: 0; font-size: 12px; color: var(--ink-2); letter-spacing: .04em; text-transform: uppercase; }
.sub { margin: 0 0 14px; font-size: 12px; color: var(--ink-2); }
.muted { color: var(--muted); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 0; }
.chip { font-size: 12px; padding: 2px 8px; border: 1px solid var(--border); border-radius: 999px; color: var(--ink); }
.chip span { color: var(--ink-2); margin-right: 4px; }
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 2px 9px 2px 7px; border-radius: 999px; font-size: 12px;
        font-weight: 500; white-space: nowrap; color: var(--ink); background: color-mix(in srgb, var(--c) 16%, transparent); }
.pill b { color: var(--c); font-weight: 700; }
.pill.big { font-size: 14px; padding: 5px 14px 5px 11px; }
.good { --c: var(--good); } .warning { --c: var(--warning); } .serious { --c: var(--serious); } .critical { --c: var(--critical); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; margin-top: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 20px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.tile .k { font-size: 12px; color: var(--ink-2); }
.tile .v { font-size: 28px; font-weight: 600; line-height: 1.25; }
.tile .d { font-size: 12px; color: var(--muted); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th { font-size: 12px; font-weight: 500; color: var(--ink-2); text-align: left; white-space: nowrap; padding: 6px 10px; border-bottom: 1px solid var(--grid); }
td { padding: 7px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
tr:last-child td { border-bottom: 0; }
th.num, td.num { text-align: right; }
th.mid, td.mid { text-align: center; }
td.wrap { white-space: normal; color: var(--ink-2); }
td .bad { color: var(--critical); font-weight: 700; }
td .ok { color: var(--ink-2); }
td small { color: var(--muted); margin-left: 6px; }
.meter { height: 6px; min-width: 110px; border-radius: 3px; background: var(--grid); overflow: hidden; }
.meter i { display: block; height: 100%; background: var(--s1); border-radius: 3px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 0 0 12px; font-size: 12px; color: var(--ink-2); }
.legend i, #tip i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }
.rt-row { display: grid; grid-template-columns: 20ch 1fr; gap: 12px; align-items: center; padding: 4px 6px; margin: 0 -6px; border-radius: 6px; outline: 0; }
.rt-row:hover, .rt-row:focus-visible { background: color-mix(in srgb, var(--ink) 6%, transparent); }
.rt-label { font-size: 12px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.rt-track { display: flex; align-items: center; gap: 8px; min-width: 0; }
.rt-bar { display: flex; gap: 2px; height: 14px; flex: none; }
.rt-bar i { flex: 0 1 0; min-width: 1px; height: 100%; }
.rt-bar i:last-child { border-radius: 0 4px 4px 0; }
.rt-total { font-size: 12px; color: var(--ink-2); font-variant-numeric: tabular-nums; white-space: nowrap; }
#tip { position: fixed; z-index: 10; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--border);
       border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 6px 20px rgba(0,0,0,.16); }
#tip[hidden] { display: none; }
#tip table { width: auto; margin-top: 4px; }
#tip td { padding: 1px 0 1px 12px; border: 0; }
#tip td:first-child { padding-left: 0; color: var(--ink-2); }
#tip tr.sum td { border-top: 1px solid var(--grid); padding-top: 4px; font-weight: 600; color: var(--ink); }
details { border-top: 1px solid var(--grid); padding: 8px 0; }
details:first-of-type { border-top: 0; }
summary { cursor: pointer; }
pre { margin: 8px 0 0; padding: 12px; overflow-x: auto; font-size: 12px; border-radius: 6px; background: color-mix(in srgb, var(--ink) 5%, transparent); }
footer { margin-top: 20px; font-size: 12px; color: var(--muted); }
a { color: inherit; text-decoration-color: var(--muted); text-underline-offset: 2px; }
a:hover { text-decoration-color: currentColor; }
"""

_JS = """
(function () {
  var tip = document.getElementById("tip");
  function show(el, x, y) {
    tip.innerHTML = el.dataset.tip;
    tip.hidden = false;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.max(8, Math.min(x + 14, window.innerWidth - w - 8)) + "px";
    tip.style.top = Math.max(8, y + 18 + h > window.innerHeight ? y - h - 12 : y + 18) + "px";
  }
  document.querySelectorAll("[data-tip]").forEach(function (el) {
    el.addEventListener("mousemove", function (e) { show(el, e.clientX, e.clientY); });
    el.addEventListener("focus", function () { var r = el.getBoundingClientRect(); show(el, r.left + 170, r.top); });
    el.addEventListener("mouseleave", function () { tip.hidden = true; });
    el.addEventListener("blur", function () { tip.hidden = true; });
  });
})();
"""


######################
### Helpers        ###
def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _fmt_sec(sec: float) -> str:
    return f"{sec:.2f} s" if sec < 60 else f"{sec / 60:.1f} min"


def _fmt_value(value) -> str:
    """Table cell text - whole numbers without decimals, missing values as a dash."""
    if isinstance(value, bool):
        return str(value)
    if pd.isna(value):
        return "&ndash;"
    if isinstance(value, numbers.Real):
        return f"{int(value):,}" if float(value).is_integer() else f"{value:,.2f}"
    return _esc(value)


def _pill(kind: str, text: str, big: bool = False) -> str:
    return f'<span class="pill {kind}{" big" if big else ""}"><b aria-hidden="true">{ICONS[kind]}</b>{_esc(text)}</span>'


def _step_label(col: str, short: bool = False) -> str:
    """`step02_sec` -> "Step 2 - simplify SWC" (short -> "Step 2")."""
    label = STEP_LABELS.get(col[:-4], col[:-4])
    return label.split(" - ")[0] if short else label


def _verdict(row: pd.Series, check_cols: list) -> str:
    if row["status"] == "failed":
        return "failed"
    if row["status"] != "ok":
        return "check_error"
    if any(pd.notna(row[c]) and not bool(row[c]) for c in check_cols):
        return "check_failed"
    return "passed"


def _link(path: str, out_dir: str, text: str) -> str:
    """Link from the report to another output of the run - relative, so the run folder can be moved as a whole."""
    try:
        href = os.path.relpath(path, out_dir).replace(os.sep, "/")
    except ValueError:   # other drive (Windows) -> absolute link
        href = Path(path).resolve().as_uri()
    return f'<a href="{_esc(href)}">{_esc(text)}</a>'


def stats_summary(stats: dict) -> str:
    """Statistics of a step as one line, e.g. `neurons: 10 | rows: 6103`. Long lists / dicts are shown as their size."""
    parts = []
    for key, value in (stats or {}).items():
        if isinstance(value, (list, tuple, set, dict)):
            short = (not isinstance(value, dict)) and 0 < len(value) <= 6 and all(isinstance(v, str) and len(v) <= 20 for v in value)
            value = ", ".join(value) if short else len(value)
        parts.append(f"{str(key).replace('_', ' ')}: {value}")
    return " | ".join(parts)


######################
### Run-level steps ###
def time_run_steps(steps: list,
                   verbose: bool = True) -> list:
    """
    Runs the steps that work on the whole run (7-9) one after the other and records each of them for the report.
    A step that raises stops the chain - the later steps depend on it and are recorded as "not run".

    steps: list -> (step name, function without arguments, output path) tuples,
                   e.g. ("step08", test.step08_score_assignment, test.dict_paths["8-swc_clumpiness"]).
    verbose: bool -> prints a message when a step starts and its result when it ends.
    Returns one record per step: step, status ("ok" | "issues" | "failed" | "not run"), seconds, stats, output, error, traceback.
    "issues" -> the step finished, but its stats list failed neurons / files.
    """
    records, stopped = [], False
    for step_name, step_fn, output in steps:
        record = {"step": step_name, "status": "not run", "seconds": None, "stats": None, "output": output, "error": None, "traceback": None}

        if not stopped:
            if verbose:
                print(f"\n> {STEP_LABELS.get(step_name, step_name)}", flush=True)
            t0 = time.time()
            try:
                record["stats"] = step_fn()
                has_failures = any((record["stats"] or {}).get(key) for key in ("failed", "failed_files"))
                record["status"] = "issues" if has_failures else "ok"
            except Exception as err:
                record.update(status="failed", error=f"{type(err).__name__}: {err}", traceback=traceback.format_exc())
                stopped = True
            record["seconds"] = round(time.time() - t0, 2)
            if verbose:
                print(f"  {record['status']} in {record['seconds']} s | {record['error'] or stats_summary(record['stats'])}", flush=True)
                if record["traceback"]:
                    print(record["traceback"], flush=True)

        records.append(record)

    return records


######################
### Report         ###
def write_test_report(df_test: pd.DataFrame,
                      tracebacks: dict = None,
                      out_dir: str = ".",
                      run_name: str = "",
                      settings: dict = None,
                      run_steps: list = None,
                      plots_dir: str = None) -> str:
    """
    Writes the test run summary as a self-contained html report and returns its path.
    df_test: pd.DataFrame -> one row per neuron (`neuron`, `status`, `failed_step`, `error`, `*_sec`, `ok_*`, other stats).
    tracebacks: dict -> {neuron_id: traceback string} of the neurons that raised.
    out_dir: str -> folder for the report, e.g. `test.dict_paths["0-run_log"]`.
    run_name: str -> pipeline run name, used in the title.
    settings: dict -> test settings shown in the header, e.g. {"selection": "first"}.
    run_steps: list -> records of the run-level steps 7-9, as returned by `time_run_steps`.
    plots_dir: str -> step 9 folder - every neuron with a plot there gets a link to it, e.g. `test.dict_paths["9-swc_clumpiness_plots"]`.
    """
    df = df_test.reset_index(drop=True)
    tracebacks = tracebacks or {}
    generated = datetime.now()

    check_cols = [c for c in df.columns if c.startswith("ok_")]
    step_cols = [c for c in df.columns if c.endswith("_sec")]
    other_cols = [c for c in df.columns if c not in check_cols + step_cols + ["neuron", "status", "failed_step", "error"]]
    step_color = {c: f"var(--s{min(i, 7) + 1})" for i, c in enumerate(step_cols)}

    verdicts = [_verdict(row, check_cols) for _, row in df.iterrows()]
    totals = df[step_cols].sum(axis=1)
    n, n_passed = len(df), verdicts.count("passed")

    run_steps = run_steps or []
    n_run_bad = sum(record["status"] in ("failed", "issues") for record in run_steps)
    run_note = f"{n_run_bad} of the run-level steps had failures" if n_run_bad else ""

    # Header - verdict banner + settings
    if n == 0:
        banner = _pill("warning", "No neurons were tested", big=True)
    elif n_passed == n and not n_run_bad:
        banner = _pill("good", f"All {n} neurons passed" + (" every check" if check_cols else " (checks not run)"), big=True)
    elif n_passed == n:
        banner = _pill("critical", f"All {n} neurons passed, but {run_note}", big=True)
    else:
        banner = _pill("critical", f"{n - n_passed} of {n} neurons did not pass" + (f", and {run_note}" if run_note else ""), big=True)
    chips = "".join(f'<span class="chip"><span>{_esc(k)}</span>{_esc(v)}</span>' for k, v in (settings or {}).items())

    # Stat tiles
    failed_parts = [f"{verdicts.count(v)} {VERDICTS[v][1].lower()}" for v in ("failed", "check_failed", "check_error") if verdicts.count(v)]
    tiles = [("Neurons tested", f"{n}", ""),
             ("Passed", f"{n_passed}", f"{n_passed / n:.0%} of tested" if n else ""),
             ("Not passed", f"{n - n_passed}", " &middot; ".join(failed_parts) or "no failures")]
    step_sums = {c: float(df[c].sum()) for c in step_cols}
    step_sums.update({f"{record['step']}_sec": record["seconds"] for record in run_steps if record["seconds"] is not None})
    if step_sums and n:
        slowest, total_sec = max(step_sums, key=step_sums.get), sum(step_sums.values())
        tiles += [("Total runtime", _fmt_sec(total_sec), f"{_step_label(slowest, short=True)} takes {step_sums[slowest] / max(total_sec, 1e-9):.0%}"),
                  ("Median per neuron", _fmt_sec(totals.median()), f"slowest {_fmt_sec(totals.max())}")]
    html_tiles = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="d">{d}</div></div>' for k, v, d in tiles)

    # Checks - share of neurons passing each check
    if check_cols:
        rows = []
        for col in check_cols:
            values = df[col].dropna().astype(bool)
            n_ok, n_all = int(values.sum()), len(values)
            label, desc = CHECK_LABELS.get(col, (col[3:].replace("_", " ").capitalize(), ""))
            pill = _pill("good" if n_ok == n_all else "critical", f"{n_ok} / {n_all}") if n_all else '<span class="muted">not run</span>'
            rows.append(f'<tr><td>{_esc(label)}</td><td class="wrap">{_esc(desc)}</td><td class="num">{pill}</td>'
                        f'<td><div class="meter"><i style="width:{n_ok / max(n_all, 1):.1%}"></i></div></td></tr>')
        html_checks = ('<div class="scroll"><table><thead><tr><th>Check</th><th>What it verifies</th><th class="num">Neurons passing</th><th>Share</th></tr></thead>'
                       f'<tbody>{"".join(rows)}</tbody></table></div>')
    else:
        html_checks = '<p class="muted">Checks were not run (RUN_CHECKS = False, or no neuron finished all steps).</p>'

    # Run-level steps - steps 7-9 run once on the whole run, not per neuron
    html_run = '<p class="muted">The run-level steps (7-9) were not part of this run.</p>'
    if run_steps:
        rows = []
        for record in run_steps:
            output = record.get("output")
            link = _link(output, out_dir, os.path.basename(os.path.normpath(output))) if output and os.path.exists(output) else "&ndash;"
            summary = _esc(record["error"] or stats_summary(record["stats"])) or "&ndash;"
            seconds = _fmt_sec(record["seconds"]) if record["seconds"] is not None else "&ndash;"
            rows.append(f'<tr><td>{_esc(STEP_LABELS.get(record["step"], record["step"]))}</td><td>{_pill(*RUN_STATUS[record["status"]])}</td>'
                        f'<td class="wrap">{summary}</td><td>{link}</td><td class="num">{seconds}</td></tr>')
        html_run = ('<div class="scroll"><table><thead><tr><th>Step</th><th>Result</th><th>Summary</th><th>Output</th><th class="num">Time</th></tr></thead>'
                    f'<tbody>{"".join(rows)}</tbody></table></div>')

    # Runtime - one stacked bar per neuron, hover / focus shows the per-step breakdown
    html_runtime = '<p class="muted">No step timings recorded.</p>'
    if step_cols and n:
        max_total = max(float(totals.max()), 1e-9)
        legend = "".join(f'<span><i style="background:{step_color[c]}"></i>{_esc(_step_label(c))}</span>' for c in step_cols)
        rows = []
        for i, row in df.iterrows():
            secs = [(c, float(row[c])) for c in step_cols if pd.notna(row[c])]
            if not secs:
                continue
            # flex-grow in ms -> the grow factors always sum to >= 1, so the segments fill the bar
            segments = "".join(f'<i style="flex-grow:{max(round(sec * 1000), 1)};background:{step_color[c]}"></i>' for c, sec in secs)
            tip = (f'<b>{_esc(row["neuron"])}</b><table>'
                   + "".join(f'<tr><td><i style="background:{step_color[c]}"></i>{_esc(_step_label(c))}</td><td class="num">{_fmt_sec(sec)}</td></tr>' for c, sec in secs)
                   + f'<tr class="sum"><td>Total</td><td class="num">{_fmt_sec(totals[i])}</td></tr></table>')
            rows.append(f'<div class="rt-row" tabindex="0" data-tip="{_esc(tip)}"><span class="rt-label">{_esc(row["neuron"])}</span>'
                        f'<span class="rt-track"><span class="rt-bar" style="width:calc((100% - 64px) * {totals[i] / max_total:.4f})">{segments}</span>'
                        f'<span class="rt-total">{_fmt_sec(totals[i])}</span></span></div>')
        html_runtime = f'<div class="legend">{legend}</div>{"".join(rows)}'

    # Neurons - full table of the run
    head = ['<th>Neuron</th>', '<th>Result</th>'] + (['<th>Plots</th>'] if plots_dir else [])
    head += [f'<th class="num">{_esc(COLUMN_LABELS.get(c, c.replace("_", " ").capitalize()))}</th>' for c in other_cols]
    head += [f'<th class="mid" title="{_esc(CHECK_LABELS.get(c, ("", ""))[1])}">{_esc(CHECK_LABELS.get(c, (c[3:], ""))[0])}</th>' for c in check_cols]
    head += [f'<th class="num" title="{_esc(_step_label(c))}">{_esc(_step_label(c, short=True))} (s)</th>' for c in step_cols]
    head += ['<th class="num">Total (s)</th>'] if step_cols else []
    rows = []
    for (i, row), verdict in zip(df.iterrows(), verdicts):
        failed_step = f'<small>{_esc(row["failed_step"])}</small>' if pd.notna(row.get("failed_step")) else ""
        cells = [f'<td>{_esc(row["neuron"])}</td>', f'<td>{_pill(*VERDICTS[verdict])}{failed_step}</td>']
        if plots_dir:
            plots = [(tree, os.path.join(plots_dir, tree, f'{row["neuron"]}.html')) for tree in ("simplified", "raw")]
            links = [_link(plot_path, out_dir, tree) for tree, plot_path in plots if os.path.exists(plot_path)]
            cells.append(f'<td>{" &middot; ".join(links) or "&ndash;"}</td>')
        cells += [f'<td class="num">{_fmt_value(row[c])}</td>' for c in other_cols]
        for c in check_cols:
            mark = "&ndash;" if pd.isna(row[c]) else '<span class="ok" title="passed">&#10003;</span>' if row[c] else '<span class="bad" title="failed">&#10005;</span>'
            cells.append(f'<td class="mid">{mark}</td>')
        cells += [f'<td class="num">{"&ndash;" if pd.isna(row[c]) else f"{row[c]:.2f}"}</td>' for c in step_cols]
        cells += [f'<td class="num">{totals[i]:.2f}</td>'] if step_cols else []
        rows.append(f'<tr>{"".join(cells)}</tr>')
    html_neurons = f'<div class="scroll"><table><thead><tr>{"".join(head)}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'

    # Errors - message + full traceback per neuron
    errors = df[df["error"].notna()] if "error" in df.columns else df.iloc[0:0]
    html_errors = "".join(f'<details><summary>{_esc(row["neuron"])} <span class="muted">&middot; {_esc(row["failed_step"] if pd.notna(row["failed_step"]) else "checks")}</span>'
                          f' &middot; {_esc(row["error"])}</summary><pre>{_esc(tracebacks.get(row["neuron"], "No traceback recorded."))}</pre></details>'
                          for _, row in errors.iterrows())
    for record in run_steps:
        label = STEP_LABELS.get(record["step"], record["step"])
        if record["status"] == "failed":
            html_errors += (f'<details><summary>{_esc(label)} &middot; {_esc(record["error"])}</summary>'
                            f'<pre>{_esc(record["traceback"] or "No traceback recorded.")}</pre></details>')
        elif record["status"] == "issues":
            failed = record["stats"].get("failed") or record["stats"].get("failed_files")
            lines = [f"{key} | {value}" for key, value in failed.items()] if isinstance(failed, dict) else [str(value) for value in failed]
            html_errors += (f'<details><summary>{_esc(label)} &middot; {len(lines)} failed</summary>'
                            f'<pre>{_esc(chr(10).join(lines[:200]))}</pre></details>')
    html_errors = html_errors or '<p class="muted">No errors or tracebacks recorded.</p>'

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SWC test report - {_esc(run_name)}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
  <header>
    <p class="eyebrow">SWC pipeline &middot; test run</p>
    <h1>{_esc(run_name) or "Test report"}</h1>
    {banner}
    <div class="chips">{chips}</div>
  </header>
  <section class="tiles">{html_tiles}</section>
  <section class="card"><h2>Checks</h2><p class="sub">Independent validation of the files written by each step.</p>{html_checks}</section>
  <section class="card"><h2>Run-level steps</h2><p class="sub">Steps 7-9 run once, on every neuron processed so far - not only on the neurons tested here.</p>{html_run}</section>
  <section class="card"><h2>Runtime per neuron</h2><p class="sub">Seconds spent in each step - hover a row for the breakdown.</p>{html_runtime}</section>
  <section class="card"><h2>Neurons</h2><p class="sub">Every tested neuron with its plots, stats, check results and step timings.</p>{html_neurons}</section>
  <section class="card"><h2>Errors</h2><p class="sub">Neurons that raised during a step or a check, and failures of the run-level steps.</p>{html_errors}</section>
  <footer>Generated {generated:%Y-%m-%d %H:%M:%S}</footer>
</main>
<div id="tip" hidden></div>
<script>{_JS}</script>
</body>
</html>
"""

    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, f"test_report_{generated:%Y%m%d_%H%M%S}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(page)

    return report_path
