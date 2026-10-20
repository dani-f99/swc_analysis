# Results viewer - a small local website to review the results of a pipeline run: the neurons of every run in
# `output/`, their status in the run reports, the files of every step and the topology plots of step 9.
# Only reads files - nothing in `output/` is changed.
#
#   python viewer.py                   -> http://127.0.0.1:8000, opens the browser
#   python viewer.py --port 8080 --no-browser
#   python viewer.py --host 0.0.0.0    -> reachable from other machines (or from your PC: `ssh -L 8000:localhost:8000 <server>`)
#
# The page itself is `scripts/viewer.html`. Stop the server with Ctrl+C.
import argparse
import csv
import json
import math
import mimetypes
import os
import re
import sys
import webbrowser
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import pandas as pd
import polars as pl

REPO_DIR = Path(__file__).resolve().parent
PAGE_PATH = REPO_DIR / "scripts" / "viewer.html"

# Per-neuron outputs of a run: key -> (folder inside the run, file extension, label)
NEURON_FILES = {"step02":            ("2-swc_simplified", ".csv", "Step 2 - simplified SWC"),
                "step03":            ("3-swc_simplified_labeled", ".csv", "Step 3 - simplified SWC + labels"),
                "step04":            ("4-json", ".json", "Step 4 - JSON tree"),
                "step06":            ("6-clumpiness_scores", ".csv", "Step 6 - clumpiness scores"),
                "step08_simplified": ("8-swc_clumpiness/simplified", ".csv", "Step 8 - scored SWC (simplified)"),
                "step08_raw":        ("8-swc_clumpiness/raw", ".csv", "Step 8 - scored SWC (raw)"),
                "step09_simplified": ("9-swc_clumpiness_plots/simplified", ".html", "Step 9 - plot (simplified)"),
                "step09_raw":        ("9-swc_clumpiness_plots/raw", ".html", "Step 9 - plot (raw)")}
SUBTREES_DIR = "5_json_divided"
UNIFIED_PATH = "7-swc_clumpiness_result/unified_clumpiness.csv"
RUN_MARKERS = {"0-run_log", "2-swc_simplified", "4-json", "6-clumpiness_scores"}

# Columns that are not clumpiness scores - every other column of a table is treated as a score column
NON_SCORE_COLUMNS = {"", "neuron", "neuron_id", "node_id", "swc_type", "x", "y", "z", "r", "parent", "type"}
SWC_COLUMNS = ["node_id", "swc_type", "x", "y", "z", "r", "parent"]

MAX_TEXT_BYTES = 2_000_000      # JSON / text files are cut after this size in the viewer
PRETTY_JSON_BYTES = 1_000_000   # JSON files below this size are indented


######################
### Helpers        ###
def _clean(value):
    """NaN / numpy scalars -> plain JSON values."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _stems(folder: Path, ext: str) -> set:
    """File names (without `ext`) of the files in `folder` that end with `ext` - empty if the folder does not exist."""
    if not folder.is_dir():
        return set()
    with os.scandir(folder) as entries:
        return {e.name[:-len(ext)] for e in entries if e.name.endswith(ext) and e.is_file()}


def _inside(root: Path, rel: str) -> Path:
    """`root / rel`, refused when the result is outside `root` (e.g. `../`)."""
    root = root.resolve()
    path = (root / rel).resolve()
    if path != root and root not in path.parents:
        raise PermissionError(f"path outside of `{root}`")
    return path


class _FileCache:
    """Small cache of parsed files, invalidated when the file changes."""
    def __init__(self, size: int = 16):
        self.size, self.items = size, OrderedDict()

    def get(self, path: Path, loader):
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        value = loader(path)
        self.items[key] = value
        if len(self.items) > self.size:
            self.items.popitem(last=False)
        return value


######################
### Readers        ###
def read_table(path: Path) -> tuple:
    """csv / raw swc file -> (columns, rows of strings). A leading unnamed pandas index column is dropped."""
    if path.suffix == ".swc":
        rows = []
        with open(path) as f:
            for line in f:
                if line.strip() and not line.lstrip().startswith("#"):
                    rows.append(line.split()[:len(SWC_COLUMNS)])
        return list(SWC_COLUMNS), rows

    with open(path, newline="") as f:
        reader = csv.reader(f)
        columns = next(reader, [])
        rows = [row for row in reader if row]

    if columns and columns[0] == "":
        columns, rows = columns[1:], [row[1:] for row in rows]
    return columns, rows


def read_reports(log_dir: Path) -> dict:
    """Run report csv files (0-run_log) -> {neuron_id: record of its most recent report}."""
    records = {}
    for report in sorted(log_dir.glob("test_report_*.csv")):   # time stamp in the name -> oldest first, newer overwrite
        try:
            df = pd.read_csv(report, dtype={"neuron": str})
        except Exception:
            continue
        df = df.astype(object).where(pd.notna(df), None)
        for rec in df.to_dict("records"):
            rec = {k: _clean(v) for k, v in rec.items()}
            rec["report"] = report.with_suffix(".html").name
            records[str(rec["neuron"])] = rec
    return records


def read_unified(path: Path) -> dict:
    """Unified clumpiness csv (step 7) -> main tree scores and number of scored sub-trees of every neuron."""
    with open(path, newline="") as f:
        columns = next(csv.reader(f), [])
    label_cols = [c for c in columns if c not in NON_SCORE_COLUMNS]

    # Every column as text -> the 18 digit neuron ids are never rounded
    lf = pl.scan_csv(path, infer_schema_length=0)
    main = lf.filter(pl.col("node_id") == "0").collect()
    scored = {}
    if label_cols:
        scored_df = (lf.filter((pl.col("node_id") != "0") & pl.any_horizontal([pl.col(c).is_not_null() for c in label_cols]))
                       .group_by("neuron_id").agg(pl.col("node_id").count().alias("n"))
                       .collect())
        scored = dict(zip(scored_df["neuron_id"].to_list(), scored_df["n"].to_list()))

    scores = {}
    for row in main.to_dicts():
        scores[row["neuron_id"]] = {c: (float(row[c]) if row[c] not in (None, "") else None) for c in label_cols}

    return {"label_columns": label_cols, "main": scores, "scored_subtrees": scored}


def neuron_stats(simplified_path: Path, raw_path: Path | None) -> dict:
    """Small statistics of one neuron from its step 3 csv (and the raw swc for the raw node count)."""
    stats = {}
    if simplified_path.exists():
        df = pd.read_csv(simplified_path, index_col=0)
        labels = df["type"].dropna().astype(str).str.split(",") if "type" in df else pd.Series(dtype=object)
        flat = labels.explode().str.strip().str.lower()
        stats.update(n_nodes_simplified=len(df),
                     n_internal=int((df["node_id"].isin(df["parent"]) & (df["parent"] != -1)).sum()),
                     n_leaves=int((~df["node_id"].isin(df["parent"])).sum()),
                     n_labeled=int(len(labels)),
                     n_pre=int((flat == "pre").sum()),
                     n_post=int((flat == "post").sum()))
    if raw_path is not None and raw_path.exists():
        with open(raw_path) as f:
            stats["n_nodes_raw"] = sum(1 for line in f if line.strip() and not line.lstrip().startswith("#"))
    return stats


######################
### Viewer         ###
class Viewer:
    def __init__(self, output_dir: Path, swc_dir: Path | None, default_run: str | None):
        self.output_dir = output_dir.resolve()
        self.swc_dir = swc_dir.resolve() if swc_dir is not None and swc_dir.is_dir() else None
        self.default_run = default_run
        self.cache = _FileCache()

    def run_dir(self, run: str) -> Path:
        if not run or "/" in run or "\\" in run or run in (".", ".."):
            raise PermissionError("bad run name")
        path = self.output_dir / run
        if not path.is_dir():
            raise FileNotFoundError(f"run `{run}` not found")
        return path

    def root_dir(self, root: str, run: str) -> Path:
        """`output` -> the run folder, `swc` -> the raw SWC input folder."""
        if root == "swc":
            if self.swc_dir is None:
                raise FileNotFoundError("raw SWC folder not found (config `swc_path`)")
            return self.swc_dir
        return self.run_dir(run)

    # GET /api/runs
    def runs(self) -> dict:
        runs = []
        if self.output_dir.is_dir():
            for entry in os.scandir(self.output_dir):
                if entry.is_dir() and RUN_MARKERS & set(os.listdir(entry.path)):
                    runs.append({"name": entry.name, "modified": entry.stat().st_mtime})
        runs.sort(key=lambda r: -r["modified"])
        names = [r["name"] for r in runs]
        default = self.default_run if self.default_run in names else (names[0] if names else None)
        return {"runs": runs, "default": default, "output_dir": str(self.output_dir)}

    # GET /api/run?run=
    def run(self, run: str) -> dict:
        run_dir = self.run_dir(run)

        files = {key: _stems(run_dir / folder, ext) for key, (folder, ext, _) in NEURON_FILES.items()}
        subtrees_dir = run_dir / SUBTREES_DIR
        subtrees = {e.name for e in os.scandir(subtrees_dir) if e.is_dir()} if subtrees_dir.is_dir() else set()

        log_dir = run_dir / "0-run_log"
        records = read_reports(log_dir) if log_dir.is_dir() else {}
        reports = sorted((p.name for p in log_dir.glob("test_report_*.html")), reverse=True) if log_dir.is_dir() else []

        unified_path = run_dir / UNIFIED_PATH
        unified = self.cache.get(unified_path, read_unified) if unified_path.exists() else {"label_columns": [], "main": {}, "scored_subtrees": {}}

        ids = set().union(*files.values(), subtrees, records, unified["main"])
        neurons = []
        for neuron_id in sorted(ids):
            rec = records.get(neuron_id)
            neurons.append({"id": neuron_id,
                            "files": [key for key, stems in files.items() if neuron_id in stems] + (["step05"] if neuron_id in subtrees else []),
                            "verdict": _verdict(rec),
                            "record": rec,
                            "main_scores": unified["main"].get(neuron_id),
                            "scored_subtrees": unified["scored_subtrees"].get(neuron_id)})

        counts = {key: len(stems) for key, stems in files.items()}
        counts["step05"] = len(subtrees)
        return {"run": run,
                "neurons": neurons,
                "counts": counts,
                "file_labels": {key: label for key, (_, _, label) in NEURON_FILES.items()},
                "label_columns": unified["label_columns"],
                "has_unified": unified_path.exists(),
                "unified_path": UNIFIED_PATH,
                "reports": reports}

    # GET /api/neuron?run=&id=
    def neuron(self, run: str, neuron_id: str) -> dict:
        run_dir = self.run_dir(run)
        if not re.fullmatch(r"[\w.-]+", neuron_id or ""):
            raise PermissionError("bad neuron id")

        files = []
        if self.swc_dir is not None and (self.swc_dir / f"{neuron_id}.swc").exists():
            path = self.swc_dir / f"{neuron_id}.swc"
            files.append({"key": "input", "label": "Input - raw SWC", "root": "swc", "path": path.name,
                          "size": path.stat().st_size, "kind": "table"})
        for key, (folder, ext, label) in NEURON_FILES.items():
            path = run_dir / folder / f"{neuron_id}{ext}"
            if path.exists():
                files.append({"key": key, "label": label, "root": "output", "path": f"{folder}/{neuron_id}{ext}",
                              "size": path.stat().st_size, "kind": {".csv": "table", ".json": "text", ".html": "html"}[ext]})

        subtree_dir = run_dir / SUBTREES_DIR / neuron_id
        subtrees = []
        if subtree_dir.is_dir():
            subtrees = sorted((e.name for e in os.scandir(subtree_dir) if e.name.endswith(".json")),
                              key=lambda name: (len(name), name))

        raw_path = self.swc_dir / f"{neuron_id}.swc" if self.swc_dir is not None else None
        stats = neuron_stats(run_dir / NEURON_FILES["step03"][0] / f"{neuron_id}.csv", raw_path)

        return {"id": neuron_id, "files": files, "subtrees_dir": f"{SUBTREES_DIR}/{neuron_id}", "subtrees": subtrees, "stats": stats}

    # GET /api/table?root=&run=&path=&offset=&limit=&sort=&desc=&q=&scored=
    def table(self, root: str, run: str, rel: str, offset: int = 0, limit: int = 100,
              sort: str = "", desc: bool = False, q: str = "", scored: bool = False) -> dict:
        path = _inside(self.root_dir(root, run), rel)
        columns, rows = self.cache.get(path, read_table)
        score_idx = [i for i, c in enumerate(columns) if c not in NON_SCORE_COLUMNS]

        if scored and score_idx:
            rows = [row for row in rows if any(i < len(row) and row[i] != "" for i in score_idx)]
        if q:
            needle = q.lower()
            rows = [row for row in rows if any(needle in cell.lower() for cell in row)]
        if sort in columns:
            i = columns.index(sort)

            def key(row):
                cell = row[i] if i < len(row) else ""
                try:
                    return (0, float(cell), "")
                except ValueError:
                    return (0 if cell else 1, math.inf, cell)
            # empty cells always last, in both directions
            filled = [row for row in rows if i < len(row) and row[i] != ""]
            empty = [row for row in rows if not (i < len(row) and row[i] != "")]
            rows = sorted(filled, key=key, reverse=desc) + empty

        return {"columns": columns,
                "score_columns": [columns[i] for i in score_idx],
                "total": len(rows),
                "offset": offset,
                "rows": rows[offset:offset + limit]}

    # GET /api/text?root=&run=&path=
    def text(self, root: str, run: str, rel: str) -> dict:
        path = _inside(self.root_dir(root, run), rel)
        size = path.stat().st_size
        with open(path, "rb") as f:
            data = f.read(MAX_TEXT_BYTES)
        text = data.decode("utf-8", errors="replace")
        pretty = False
        if path.suffix == ".json" and size <= PRETTY_JSON_BYTES:
            try:
                text, pretty = json.dumps(json.loads(text), indent=1), True
            except (ValueError, RecursionError):
                pass
        return {"text": text, "size": size, "truncated": size > MAX_TEXT_BYTES, "pretty": pretty}


def _verdict(rec: dict | None) -> str:
    """Report record -> passed / check_failed / failed / check_error / not_tested (same verdicts as the run report)."""
    if not rec:
        return "not_tested"
    if rec.get("status") != "ok":
        return rec.get("status") or "failed"
    checks = [v for k, v in rec.items() if k.startswith("ok_") and v is not None]
    return "passed" if all(bool(v) for v in checks) else "check_failed"


######################
### Http server    ###
def make_handler(viewer: Viewer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):   # quiet - only errors are printed
            pass

        def send_json(self, data, status=HTTPStatus.OK):
            body = json.dumps(data, default=_clean).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path: Path, download: bool = False):
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix in (".csv", ".swc", ".json", ".txt"):
                ctype = "text/plain; charset=utf-8"   # shown in the browser instead of downloaded
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "no-cache")   # outputs change between runs -> always revalidate
            if download:
                self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(path.name)}")
            self.end_headers()
            with open(path, "rb") as f:
                while chunk := f.read(1 << 20):
                    self.wfile.write(chunk)

        def do_GET(self):
            url = urlparse(self.path)
            qs = {k: v[0] for k, v in parse_qs(url.query).items()}
            route = url.path
            try:
                if route in ("/", "/index.html"):
                    return self.send_file(PAGE_PATH)
                if route == "/api/runs":
                    return self.send_json(viewer.runs())
                if route == "/api/run":
                    return self.send_json(viewer.run(qs.get("run", "")))
                if route == "/api/neuron":
                    return self.send_json(viewer.neuron(qs.get("run", ""), qs.get("id", "")))
                if route == "/api/table":
                    return self.send_json(viewer.table(qs.get("root", "output"), qs.get("run", ""), qs.get("path", ""),
                                                       offset=max(int(qs.get("offset", 0)), 0),
                                                       limit=min(max(int(qs.get("limit", 100)), 1), 1000),
                                                       sort=qs.get("sort", ""), desc=qs.get("desc") == "1",
                                                       q=qs.get("q", ""), scored=qs.get("scored") == "1"))
                if route == "/api/text":
                    return self.send_json(viewer.text(qs.get("root", "output"), qs.get("run", ""), qs.get("path", "")))

                # /files/output/<run>/<path> and /files/swc/<file> -> the file itself (plots, reports, downloads)
                parts = unquote(route).split("/", 3)   # ["", "files", root, rest]
                if len(parts) == 4 and parts[1] == "files" and parts[2] in ("output", "swc"):
                    if parts[2] == "output":
                        run, _, rel = parts[3].partition("/")
                    else:
                        run, rel = "", parts[3]
                    path = _inside(viewer.root_dir(parts[2], run), rel)
                    if not path.is_file():
                        raise FileNotFoundError(rel)
                    return self.send_file(path, download=qs.get("download") == "1")

                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

            except PermissionError as err:
                self.send_json({"error": str(err)}, HTTPStatus.FORBIDDEN)
            except FileNotFoundError as err:
                self.send_json({"error": f"not found: {err}"}, HTTPStatus.NOT_FOUND)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as err:
                print(f"> {route}: {type(err).__name__}: {err}", file=sys.stderr)
                self.send_json({"error": f"{type(err).__name__}: {err}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Local website to review the results of the SWC pipeline runs.")
    parser.add_argument("-c", "--config", default="config.json", help="config file, relative to the repo folder (default: config.json)")
    parser.add_argument("--output-dir", default=None, help="folder of the runs (default: `output` next to the config)")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1 - this computer only)")
    parser.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = parser.parse_args()

    # Same path rules as `ProcessSWC` - relative paths in the config are resolved against the config's folder
    config_path = REPO_DIR / args.config
    swc_dir, default_run, base_dir = None, None, REPO_DIR
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        base_dir = config_path.resolve().parent
        default_run = config.get("run_name")
        if config.get("swc_path"):
            swc_dir = base_dir.joinpath(*config["swc_path"].split(","))
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "output"

    viewer = Viewer(output_dir, swc_dir, default_run)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(viewer))
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}/"
    print(f"> Results viewer of `{viewer.output_dir}` on {url}  (Ctrl+C to stop)")
    if viewer.swc_dir is None:
        print("> Raw SWC folder not found - the input SWC files are not shown.")
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n> Stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
