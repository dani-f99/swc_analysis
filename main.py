# Test run of the pipeline from the command line (server use) - same run and checks as the test cell of `dev_notebook.ipynb`.
# Processes a selection of neurons step by step (every step prints a start message and a progress bar), records failures,
# validates each step's output and writes
# an html report + a csv of the results to `output/<run_name>/0-run_log`.
#
#   python run_test.py                      -> first 10 neurons, steps 1-9, overwrite and n_jobs from the config
#   python run_test.py -n 0                 -> all neurons
#   python run_test.py -n 50 --selection random --n-jobs 1 --overwrite true
#   python run_test.py --last-step 5        -> stop before the clumpiness calculation
#   python run_test.py --neuron-ids 720575940596125868 720575940597856265
#
# Exit code: 0 -> every neuron and every run-level step (7-9) passed, 1 -> a neuron failed a step or a check, or a
#            run-level step had failures, 2 -> nothing was run (setup problem).
import argparse
import glob
import json
import os
import random
import shutil
import sys
import time
import traceback

# The scripts read `config.json` and create folders relative to the working directory -> always run from the repo folder
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

import pandas as pd
import polars as pl
from joblib import delayed

from scripts.process_swc import ProcessSWC
from scripts.helpers import run_parallel
from scripts.report import write_test_report, time_run_steps, STEP_LABELS


######################
### Per-step checks ###
def check_neuron(test: ProcessSWC,
                 neuron_id: str,
                 last_step: int = 6) -> dict:
    """
    Independent checks on the files written by steps 2-6 (does not use the pipeline functions).
    test: ProcessSWC -> initiated pipeline object (paths).
    neuron_id: str -> neuron id to check.
    last_step: int -> last step that was run, the step 6 check is skipped below 6.
    """
    checks = {}

    # Step 2 - simplified tree has exactly one root
    swc_simp = pd.read_csv(os.path.join(test.dict_paths["2-swc_simplified"], f"{neuron_id}.csv"), index_col=0)
    checks["n_nodes_simplified"] = len(swc_simp)
    checks["ok_single_root"] = int((swc_simp["parent"] == -1).sum()) == 1

    # Step 3 - one row per node, and every node carries exactly its own labels
    swc_lab = pd.read_csv(os.path.join(test.dict_paths["3-swc_simplified_labeled"], f"{neuron_id}.csv"), index_col=0)
    checks["ok_unique_nodes"] = swc_lab["node_id"].is_unique

    labels = pl.scan_parquet(test.labels_path).filter(pl.col("neuron").cast(pl.Utf8) == neuron_id).select(["node_id", "type"]).collect().to_pandas()
    truth = labels.dropna().groupby("node_id")["type"].agg(lambda s: set(s.astype(str)))
    expected = swc_lab["node_id"].map(truth).apply(lambda v: v if isinstance(v, set) else set())
    found = swc_lab["type"].apply(lambda v: set(str(v).split(",")) if pd.notna(v) else set())
    checks["labeled_nodes_raw"] = labels["node_id"].nunique()
    checks["labeled_nodes_expected"] = int((expected.apply(len) > 0).sum())
    checks["labeled_nodes_found"] = int((found.apply(len) > 0).sum())
    checks["label_mismatches"] = int((expected != found).sum())
    checks["ok_labels"] = checks["label_mismatches"] == 0

    # Step 4 - every csv node is in the json tree, with the same labels as the step 3 csv
    with open(os.path.join(test.dict_paths["4-json"], f"{neuron_id}.json")) as f:
        tree = json.load(f)
    json_labels, stack = {}, [tree]
    while stack:
        node = stack.pop()
        json_labels[node[0]["nodeID"]] = set(node[0]["nodeLabels"])
        stack.extend(node[1])
    csv_labels = {str(int(n)): {x.strip().lower() for x in labs} for n, labs in zip(swc_lab["node_id"], found)}
    checks["ok_json_nodes"] = len(json_labels) == len(swc_lab)
    checks["ok_json_labels"] = json_labels == csv_labels

    # Step 5 - one sub-tree per internal node, root excluded
    n_internal = int(swc_lab["node_id"].isin(swc_lab["parent"]).sum())
    n_subtrees = len(glob.glob(os.path.join(test.dict_paths["5_json_divided"], neuron_id, "*.json")))
    checks["ok_subtrees"] = n_subtrees == n_internal - 1

    # Step 6 - one clumpiness csv per json tree (main tree + every sub-tree)
    if last_step >= 6:
        n_csv = len(glob.glob(os.path.join(test.dict_paths["6-clumpiness_scores"], neuron_id, "*.csv")))
        checks["ok_clumpiness"] = n_csv == n_subtrees + 1

    return checks


######################
### Single steps   ###
STEP_METHODS = {"step02": "step02_simplify_swc",
                "step03": "step03_label_joining",
                "step04": "step04_json_creation",
                "step05": "step05_internal_node_div",
                "step06": "step06_clumpiness_calculation"}


def run_step(test: ProcessSWC,
             step_name: str,
             neuron_id: str) -> tuple:
    """Runs one per-neuron step on one neuron. Never raises - returns (seconds | None, error | None, traceback | None)."""
    t0 = time.time()
    try:
        getattr(test, STEP_METHODS[step_name])(neuron_id)
    except Exception as err:
        return None, f"{type(err).__name__}: {err}", traceback.format_exc()
    return round(time.time() - t0, 2), None, None


def run_check(test: ProcessSWC,
              neuron_id: str,
              last_step: int = 6) -> tuple:
    """Checks the outputs of one neuron. Never raises - returns (checks | None, error | None, traceback | None)."""
    try:
        return check_neuron(test, neuron_id, last_step), None, None
    except Exception as err:
        return None, f"{type(err).__name__}: {err}", traceback.format_exc()


######################
### Run            ###
def main():
    parser = argparse.ArgumentParser(description="Test run of the SWC pipeline (steps 1-9) with per-step checks and an html report.")
    parser.add_argument("-c", "--config", default="config.json", help="config file, relative to the repo folder (default: config.json)")
    parser.add_argument("-n", "--n-neurons", type=int, default=10, help="number of neurons to test, 0 = all (default: 10)")
    parser.add_argument("--selection", choices=["first", "random"], default="first", help="first = sorted ids (default), random = seeded sample")
    parser.add_argument("--seed", type=int, default=0, help="seed for --selection random (default: 0)")
    parser.add_argument("--neuron-ids", nargs="+", default=[], help="explicit neuron ids -> overrides -n / --selection")
    parser.add_argument("--overwrite", choices=["config", "true", "false"], default="config",
                        help="overwrite rule for steps 2+ only, the labels parquet is untouched (default: config `overwrite_info`)")
    parser.add_argument("--last-step", type=int, choices=[2, 3, 4, 5, 6, 7, 8, 9], default=9, help="last step to run (default: 9)")
    parser.add_argument("--n-jobs", type=int, default=None, help="neurons processed in parallel, 1 = sequential (default: config `n_jobs`)")
    parser.add_argument("--no-checks", action="store_true", help="skip the validation of the output files")
    parser.add_argument("--out-dir", default=None, help="folder for the html report + csv (default: the run's `0-run_log`)")
    args = parser.parse_args()

    # Step 6 needs the find-clumpiness program -> fail before the long part, not on the first neuron
    if args.last_step >= 6 and shutil.which("find-clumpiness") is None:
        print("> `find-clumpiness` was not found on PATH - install it, or run with `--last-step 5`.")
        return 2

    test = ProcessSWC(config_file=args.config)

    # Step 1 - labels parquet (needed by every neuron) -> returns the neurons with a raw SWC file AND labels
    print(f"> {STEP_LABELS['step01']}", flush=True)
    candidates = test.step01_create_labels()

    # Steps 2+ overwrite override (after step 1, so the labels parquet is not rebuilt)
    if args.overwrite != "config":
        test.dict_config["overwrite_info"] = args.overwrite == "true"

    if args.neuron_ids:
        selected = [str(i) for i in args.neuron_ids]
    elif args.n_neurons <= 0:
        selected = candidates
    elif args.selection == "random":
        selected = random.Random(args.seed).sample(candidates, min(args.n_neurons, len(candidates)))
    else:
        selected = candidates[:args.n_neurons]

    if not selected:
        print("> No neurons to test - no SWC file with labels was found.")
        return 2

    # The checks read the outputs of steps 2-5
    run_checks = (not args.no_checks) and args.last_step >= 5
    if (not args.no_checks) and (not run_checks):
        print("> Checks need steps 2-5 - skipped for this run.")

    n_jobs = args.n_jobs if args.n_jobs is not None else test.dict_config["n_jobs"]
    print(f"> {len(candidates)} neurons with labels, testing {len(selected)} "
          f"(steps 2-{args.last_step}, overwrite={test.dict_config['overwrite_info']}, n_jobs={n_jobs}).")

    records = {neuron_id: {"neuron": neuron_id, "status": "ok", "failed_step": None, "error": None} for neuron_id in selected}
    tracebacks = {}
    t0 = time.time()

    # Steps 2-6 - one step at a time over all neurons, so every step announces itself and has its own progress bar.
    # A neuron that failed a step is left out of the next steps.
    for step_name in list(STEP_METHODS)[:args.last_step - 1]:
        active = [neuron_id for neuron_id in selected if records[neuron_id]["status"] == "ok"]
        print(f"\n> {STEP_LABELS[step_name]} ({len(active)} neurons)", flush=True)
        results = run_parallel((delayed(run_step)(test, step_name, neuron_id) for neuron_id in active),
                               total=len(active), n_jobs=n_jobs, desc=STEP_LABELS[step_name].split(" - ")[0], unit=" neurons")
        for neuron_id, (seconds, error, trace) in zip(active, results):
            if error is None:
                records[neuron_id][f"{step_name}_sec"] = seconds
            else:
                records[neuron_id].update(status="failed", failed_step=step_name, error=error)
                tracebacks[neuron_id] = trace

    if run_checks:
        active = [neuron_id for neuron_id in selected if records[neuron_id]["status"] == "ok"]
        print(f"\n> Checks - validating the output files ({len(active)} neurons)", flush=True)
        results = run_parallel((delayed(run_check)(test, neuron_id, args.last_step) for neuron_id in active),
                               total=len(active), n_jobs=n_jobs, desc="Checks", unit=" neurons")
        for neuron_id, (checks, error, trace) in zip(active, results):
            if error is None:
                records[neuron_id].update(checks)
            else:
                records[neuron_id].update(status="check_error", error=error)
                tracebacks[neuron_id] = trace

    wall_sec = time.time() - t0
    df_test = pd.DataFrame(list(records.values()))
    print()


    ######################
    ### Summary        ###
    print(df_test["status"].value_counts().to_string())
    check_cols = [c for c in df_test.columns if c.startswith("ok_")]
    if check_cols:
        print("\n> Share of neurons passing each check:")
        # astype(float) -> a failed neuron leaves empty checks, which turns the columns into object dtype
        print(df_test[check_cols].astype(float).mean().round(3).to_string())

    errors = df_test[df_test["error"].notna()]
    if len(errors):
        print(f"\n> {len(errors)} neurons with errors (first 20, full traces in the report):")
        for _, row in errors.head(20).iterrows():
            print(f"  {row['neuron']} | {row['failed_step'] or 'checks'} | {row['error']}")

    # Steps 7-9 - run once, on every neuron processed so far (not only this selection):
    # unified clumpiness csv -> scores on the SWC nodes -> html topology plots
    final_steps = [("step07", test.step07_result_annealing, os.path.join(test.dict_paths["7-swc_clumpiness_result"], "unified_clumpiness.csv")),
                   ("step08", test.step08_score_assignment, test.dict_paths["8-swc_clumpiness"]),
                   ("step09", test.step09_plot_creation, test.dict_paths["9-swc_clumpiness_plots"])]
    run_steps = time_run_steps(final_steps[:max(args.last_step - 6, 0)])

    # Report + results table
    selection = "explicit ids" if args.neuron_ids else "all" if args.n_neurons <= 0 else args.selection
    report_path = write_test_report(df_test, tracebacks,
                                    out_dir  = args.out_dir or test.dict_paths["0-run_log"],
                                    run_name = test.dict_config["run_name"],
                                    settings = {"with labels": len(candidates), "tested": len(selected), "selection": selection,
                                                "seed": args.seed, "steps": f"2-{args.last_step}", "run checks": run_checks,
                                                "overwrite": test.dict_config["overwrite_info"], "n jobs": n_jobs,
                                                "wall time": f"{wall_sec / 60:.1f} min"},
                                    run_steps = run_steps,
                                    plots_dir = test.dict_paths["9-swc_clumpiness_plots"])
    csv_path = report_path.replace(".html", ".csv")
    df_test.to_csv(csv_path, index=False)
    print(f"\n> Report saved to {report_path}")
    print(f"> Results saved to {csv_path}")

    # A neuron passes when every step ran and every check is True
    passed = (df_test["status"] == "ok") & df_test[check_cols].astype(float).fillna(1.0).astype(bool).all(axis=1)
    print(f"> {int(passed.sum())} / {len(df_test)} neurons passed in {wall_sec / 60:.1f} min.")

    return 0 if (passed.all() and all(record["status"] == "ok" for record in run_steps)) else 1


if __name__ == "__main__":
    sys.exit(main())
