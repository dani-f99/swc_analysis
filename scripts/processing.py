#########
# Fixed version of the original `_archive/scripts/processing.py` - see `_archive/BUGFIX_REPORT.md` for the full change log.
# Imports
from scripts.helpers import  chunked_iterable

from joblib import Parallel, delayed
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pandas as pd
import polars as pl
import subprocess
import uuid
import json
import csv
import os


####################################################
####################################################
def generate_internal_subtrees(input_json_path: str,
                               neuron_number: str,
                               output_dir: str,
                               overwrite: bool = True,
                               ) -> int | None:
    """
    Parses a nested JSON tree of a neuron and exports all internal sub-trees
    (excluding the main root and leaves) into individual JSON files, inside `output_dir/<neuron_number>/`.

    input_json_path: str -> File path of the json file to be processed.
    neuron_number: str -> String which represents the neuron id.
    overwrite: bool -> If True, the neuron's sub-tree folder is emptied and re-written. If False and the folder
                       already holds sub-trees, nothing is done (returns None).
    output_dir: str -> String path of the output folder to which the processed file will be saved to.
    Returns the number of sub-tree files written, or None if skipped.
    """

    # Paths indiations
    # [FIX 4b] `Path(output_dir, <int>)` raised TypeError - always use the string id.
    neuron_number = str(neuron_number)
    input_path = Path(input_json_path)
    out_dir = Path(output_dir, neuron_number)

    # [FIX 4a] Old code called `.exists()` on a str (AttributeError whenever overwrite=False), and with
    # overwrite=True stale sub-trees from an older tree version stayed next to the new ones.
    existing = list(out_dir.glob("*.json")) if out_dir.exists() else []
    if existing and (overwrite is False):
        return None

    for stale_file in existing:
        stale_file.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the original JSON tree
    with open(input_path, 'r') as f:
        tree_data = json.load(f)

    n_written = 0

    def traverse(current_node, is_main_root=False):
        nonlocal n_written
        # Validate node structure: [{"nodeID": "...", "nodeLabels": [...]}, [children]]
        if not current_node or len(current_node) != 2:
            return

        node_info, children = current_node
        node_id = node_info.get("nodeID")

        # A node is internal if the children array is not empty
        if children:
            # Export the sub-tree if it is NOT the absolute root
            if not is_main_root:
                output_filepath = out_dir / f"{neuron_number}_{node_id}.json"
                with open(output_filepath, 'w') as out_f:
                    json.dump(current_node, out_f)
                n_written += 1

            # ALWAYS recursively process all children
            for child in children:
                traverse(child, is_main_root=False)

    # Initiate traversal, explicitly flagging the first node as the main root
    traverse(tree_data, is_main_root=True)

    return n_written


#####################################
#####################################
def _process_single_neuron(filepath:str):
    """
    Worker function executed in parallel. (Unchanged copy - not used by the pipeline.)
    Prints removed to prevent terminal output corruption.

    filepath: str -> Path to the JSON neuron clumpiness tree, on which the clumpiness calculation will be made.
    """
    neuron_id = filepath.stem
    temp_filename = f"temp_clump_{uuid.uuid4().hex}.csv"

    try:
        result = subprocess.run(
            ["find-clumpiness", "-e", "AllExclusive", "-i", str(filepath), "-f", "JSON"],
            capture_output=True,
            text=True,
            check=True
        )

        lines = result.stdout.strip().splitlines()

        if len(lines) <= 1:
            return None

        with open(temp_filename, 'w', newline='') as temp_out:
            for line in lines[1:]:
                parts = line.split(',')
                if len(parts) >= 3:
                    col1 = parts[0].strip()
                    col2 = parts[1].strip()
                    col3 = parts[2].strip()

                    formatted_result = f'"{neuron_id}",{col1}-{col2},{col3}\n'
                    temp_out.write(formatted_result)

        return temp_filename

    except subprocess.CalledProcessError:
        # Silently fail or log to a file instead of printing
        return None
    except FileNotFoundError:
        # Command not found
        return None


######################################################
######################################################
def process_single_clumpiness(filepath: str,
                              output_dir: str,
                              overwrite: bool = False):
    """
    Takes a JSON file, runs find-clumpiness, and saves the exact output
    to a CSV file with the same name as the input, overwriting if it exists.

    filepath: str -> String path to the JSON file containing the neuron tree.
    output_dir: str -> String path to which the clumpiness results will be save to.
    overwrite: bool -> If argument 'overwrite' is True, will overwrite data if already exists (Defualt is False).
    """
    filepath = Path(filepath)
    output_csv = Path(output_dir) / f"{filepath.stem}.csv"
    # Written to a temp file first -> a failed run (e.g. find-clumpiness missing) never destroys an existing result
    temp_csv = output_csv.with_name(output_csv.name + ".tmp")

    if (overwrite is False) and (os.path.exists(output_csv)):
        return None

    try:
        # Opening in 'w' mode automatically overwrites the file if it already exists
        with open(temp_csv, 'w') as f_out:
            subprocess.run(
                ["find-clumpiness", "-e", "AllExclusive", "-i", str(filepath), "-f", "JSON"],
                stdout=f_out,             # Dumps output straight to the file
                stderr=subprocess.PIPE,   # Catches errors so they don't print to terminal
                text=True,
                check=True
            )
        os.replace(temp_csv, output_csv)
        return True

    # [FIX 6c] The reason of the failure (find-clumpiness stderr / missing executable) used to be discarded.
    except subprocess.CalledProcessError as err:
        if temp_csv.exists():
            temp_csv.unlink()
        raise Exception(f"> find-clumpiness failed on `{filepath}`: {(err.stderr or '').strip()}") from err

    except FileNotFoundError as err:
        if temp_csv.exists():
            temp_csv.unlink()
        raise Exception(f"> Failed to process clumpiness on `{filepath}` - is `find-clumpiness` on PATH? ({err})") from err

    except Exception as err:
        # If it fails, delete the empty/partial CSV so it doesn't leave corrupted data
        if temp_csv.exists():
            temp_csv.unlink()
        raise Exception(f"> Failed to process clumpiness on `{filepath}`: {err}") from err


##############################################
##############################################
def process_clumpiness_csv(filepath_str: str):
    """
    Reads a single CSV, safely skips empty files, and appends ID columns.

    filepath_str: str -> path to a `<neuron_id>_<node_id>.csv` find-clumpiness output.
    Returns (df | None, error message | None).
    """
    filepath = Path(filepath_str)

    # Fast-skip: Files <= 32 bytes physically cannot contain data rows
    if os.path.getsize(filepath) <= 32:
        return None, None

    # [FIX 6b] Parse errors used to be swallowed (`return None`), silently dropping files from the dataset.
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return None, None

        # rsplit -> only the last "_" separates the node id
        neuron_id, node_id = filepath.stem.rsplit('_', 1)

        df['neuron_id'] = neuron_id
        df['node_id'] = node_id

        return df, None

    except Exception as err:
        return None, f"{filepath}: {err}"


################################################
################################################
def compile_unified_dataset(input_directory: str,
                            output_filepath: str,
                            batch_size: int = 2000) -> dict:
    """
    Iterates over CSVs and processes them using joblib for robust parallel execution.

    input_directory: str -> String input clumpiness csv's folder.
    output_filepath: str -> String output of the joined clumpiness file.
    batch_size: str -> Number of itirations per batch of processing.
    Returns a dict with the number of files, rows written and the failed files.
    """

    def get_csv_files():
        with os.scandir(input_directory) as entries:
            for entry in entries:
                if entry.name.endswith('.csv') and entry.is_file():
                    yield entry.path

    writer = None
    n_jobs = min(4, (os.cpu_count() or 1))
    n_files = 0
    n_rows = 0
    failed_files = []

    with tqdm(desc="Compiling Parquet", unit=" files") as pbar:
        for file_chunk in chunked_iterable(get_csv_files(), batch_size):

            # joblib handles the worker pool much more safely on Windows
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(process_clumpiness_csv)(f) for f in file_chunk
            )

            valid_dfs = [df for df, _ in results if df is not None]
            failed_files += [error for _, error in results if error is not None]

            if valid_dfs:
                batch_df = pd.concat(valid_dfs, ignore_index=True)

                # [FIX 6d] Later batches are cast to the first batch's schema (column types could differ between batches).
                if writer is None:
                    table = pa.Table.from_pandas(batch_df, preserve_index=False)
                    writer = pq.ParquetWriter(output_filepath, table.schema, compression='ZSTD')
                else:
                    table = pa.Table.from_pandas(batch_df, schema=writer.schema, preserve_index=False)

                writer.write_table(table)
                n_rows += len(batch_df)

            n_files += len(file_chunk)
            pbar.update(len(file_chunk))

    if writer:
        writer.close()

    if failed_files:
        print(f"> {len(failed_files)} clumpiness files failed to parse (see the returned `failed_files`).")
    print("> Dataset compilation complete.")

    return {"files": n_files, "rows": n_rows, "failed_files": failed_files}


##############################################
##############################################
def read_neuron_clumpiness(neuron_dir: str) -> tuple:
    """
    Reads every find-clumpiness csv of one neuron into wide rows - one row per tree, one value per label combination.
    `<neuron_id>.csv` is the main tree (node_id 0), `<neuron_id>_<node_id>.csv` is the sub-tree of an internal node.

    neuron_dir: str -> path to the neuron's folder inside the clumpiness scores folder.
    Returns (list of row dicts, list of error messages).
    """
    neuron_id = Path(neuron_dir).name
    rows, errors = [], []

    with os.scandir(neuron_dir) as entries:
        for entry in entries:
            if not (entry.name.endswith(".csv") and entry.is_file()):
                continue

            try:
                stem = entry.name[:-4]
                if stem == neuron_id:
                    node_id = 0
                else:
                    # rsplit -> only the last "_" separates the node id
                    prefix, node = stem.rsplit("_", 1)
                    if prefix != neuron_id:
                        raise ValueError("file name does not belong to this neuron")
                    node_id = int(node)

                row = {"neuron_id": neuron_id, "node_id": node_id}
                with open(entry.path, newline="") as f:
                    reader = csv.reader(f)
                    next(reader, None)   # header -> property1,property2,value
                    for line in reader:
                        if not line:
                            continue
                        label1, label2, value = line
                        row[f"{label1}_{label2}"] = float(value)

                rows.append(row)

            except Exception as err:
                errors.append(f"{entry.path}: {err}")

    return rows, errors


###############################################
###############################################
def compile_wide_clumpiness(input_directory: str,
                            output_filepath: str,
                            n_jobs: int = 1) -> dict:
    """
    Joins the find-clumpiness results of every neuron folder into a single wide file (csv or parquet, by extension):
    neuron_id, node_id (0 = root / main tree) and one `<label1>_<label2>` column per label combination,
    null where the combination was not found in the tree.

    input_directory: str -> clumpiness scores folder, one sub-folder per neuron.
    output_filepath: str -> path of the unified file, `.csv` or `.parquet`.
    n_jobs: int -> number of neurons read in parallel.
    Returns a dict with the number of neurons, rows, the label combination columns and the failed files.
    """
    neuron_dirs = sorted(entry.path for entry in os.scandir(input_directory) if entry.is_dir())

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(read_neuron_clumpiness)(d) for d in tqdm(neuron_dirs, desc="Unifying clumpiness", unit=" neurons")
    )

    frames, failed_files = [], []
    for rows, errors in results:
        failed_files += errors
        if rows:
            # Columns of this neuron -> a label combination missing from a tree stays null
            label_cols = sorted({key for row in rows for key in row} - {"neuron_id", "node_id"})
            schema = {"neuron_id": pl.Utf8, "node_id": pl.Int64, **{c: pl.Float64 for c in label_cols}}
            frames.append(pl.DataFrame({c: [row.get(c) for row in rows] for c in schema}, schema=schema))

    if not frames:
        raise ValueError(f"> No clumpiness csv files found in `{input_directory}`.")

    # diagonal -> union of the label combination columns, null where a neuron does not have the column
    unified = pl.concat(frames, how="diagonal")
    label_cols = sorted(set(unified.columns) - {"neuron_id", "node_id"})
    unified = unified.select(["neuron_id", "node_id"] + label_cols).sort(["neuron_id", "node_id"])

    # node_id 0 is reserved for the root -> a sub-tree of a real node 0 would give two rows with the same id
    n_duplicates = int(unified.select(["neuron_id", "node_id"]).is_duplicated().sum())
    if n_duplicates:
        raise ValueError(f"> {n_duplicates} duplicated (neuron_id, node_id) rows - is there a sub-tree of a node with id 0?")

    # File type by extension -> `.csv` (null = empty field) or parquet
    if str(output_filepath).endswith(".csv"):
        unified.write_csv(output_filepath)
    else:
        unified.write_parquet(output_filepath, compression="zstd")

    if failed_files:
        print(f"> {len(failed_files)} clumpiness files failed to parse (see the returned `failed_files`).")

    return {"neurons": len(frames), "rows": len(unified), "label_columns": label_cols, "failed_files": failed_files}

#################################################################################################################
# > Worker function for joblib that joins SWCs file with their appropriate clumpiness results by neuron_id label.
def join_swc_clumpiness(neuron_id: str ,
                        path_2process: str,
                        save_path: str,
                        parquet_path: str,
                        overwrite: bool = False):

    """
    neuron_id: str -> neuron id as string file.
    path_2process: str -> path to the target swc files (to which the labels will be joined).
    save_path: str -> path to the save output folder.
    parquet_path -> path to the unified clumpiness parquet file.
    overwrite: bool -> if False and the output exists, nothing is done (returns None).
    Returns the number of SWC nodes that received a clumpiness score, or None if skipped.
    """
    i = str(neuron_id)
    output_path = os.path.join(save_path, f"{i}.csv")

    # [FIX 6a] Early exit used to ignore overwrite, so re-runs kept stale results.
    if os.path.exists(output_path) and (overwrite is False):
        return None

    # Re-initialize the pyarrow dataset inside the worker
    # This prevents pickling/serialization errors across different CPU cores
    worker_dataset = ds.dataset(parquet_path, format="parquet")

    # Defining required path and loading swc file
    i_path = os.path.join(path_2process, f"{i}.csv")
    i_swc = pd.read_csv(i_path, index_col=0)

    # Filter dataset for the specific neuron_id
    i_label = worker_dataset.to_table(filter=(ds.field("neuron_id") == i)).to_pandas()

    # Optional safety check in case a neuron ID has no corresponding labels
    if i_label.empty:
        return 0

    i_label['node_id'] = i_label['node_id'].astype("int")

    # Creating a unified labels column named 'label'
    i_label.insert(loc=2,
                   column="label",
                   value=i_label['property1'] + "_" + i_label['property2'])

    # [FIX 6a] `pivot_table(aggfunc="first")` silently picked one of several scores for the same node+label.
    duplicates = i_label.duplicated(subset=["node_id", "label"])
    if duplicates.any():
        raise ValueError(f"> Neuron {i}: {duplicates.sum()} duplicated (node_id, label) clumpiness rows - check for stale/duplicate csv files.")

    # Use pivot_table instead of pivot, and specify the index
    flipped_labels = i_label.pivot_table(index=['neuron_id', 'node_id'],
                                         columns='label',
                                         values='value',
                                         aggfunc='first').reset_index()

    # Merging SWC and labels
    i_merged = pd.merge(left=i_swc,
                        right=flipped_labels.iloc[:, 1:],
                        left_on="node_id",
                        right_on="node_id",
                        how="left")

    # Write out the file
    i_merged.to_csv(output_path)

    return int(i_merged["node_id"].isin(flipped_labels["node_id"]).sum())


################################################
################################################
def assign_swc_clumpiness(neuron_id: str,
                          scores: pd.DataFrame,
                          simplified_path: str,
                          raw_path: str,
                          output_dir: str,
                          trees: tuple = ("simplified", "raw"),
                          overwrite: bool = False) -> dict | None:
    """
    Assigns the clumpiness scores of one neuron to the nodes of its SWC tree - the root and the internal nodes
    get their scores, every other node stays null. Joined by node id, so the same scores fit the simplified and the raw tree.

    neuron_id: str -> neuron id.
    scores: pd.DataFrame -> the neuron's rows of the unified clumpiness file: node_id (0 = root) + one column per label combination.
    simplified_path: str -> simplified SWC csv of the neuron (the tree the scores were calculated on).
    raw_path: str -> raw `.swc` file of the neuron.
    output_dir: str -> output folder, the files are saved as `<output_dir>/<simplified | raw>/<neuron_id>.csv`.
    trees: tuple -> which trees to write, "simplified" and / or "raw".
    overwrite: bool -> if False, an existing output file is kept.
    Returns a dict with the number of nodes per written tree and of scored nodes, or None if nothing was written.
    """
    neuron_id = str(neuron_id)
    targets = {tree: os.path.join(output_dir, tree, f"{neuron_id}.csv") for tree in trees}
    targets = {tree: path for tree, path in targets.items() if overwrite or not os.path.exists(path)}
    if not targets:
        return None

    # The simplified tree is always loaded - it defines the root of the neuron
    swc_simplified = pd.read_csv(simplified_path, index_col=0)
    roots = swc_simplified.loc[swc_simplified["parent"] == -1, "node_id"]
    if len(roots) != 1:
        raise ValueError(f"> Neuron {neuron_id}: {len(roots)} root nodes in the simplified tree, expected 1.")

    # The unified file stores the root (main tree) as node_id 0 -> back to the real root id
    scores = scores.copy()
    scores.loc[scores["node_id"] == 0, "node_id"] = roots.iloc[0]
    if scores["node_id"].duplicated().any():
        raise ValueError(f"> Neuron {neuron_id}: duplicated node ids in the clumpiness scores.")

    stats = {"n_scored": len(scores)}
    for tree, output_path in targets.items():
        if tree == "simplified":
            swc = swc_simplified
        elif tree == "raw":
            swc = pd.read_csv(raw_path,
                              comment='#',
                              header=None,
                              sep=r'\s+',
                              names=["node_id", "swc_type", "x", "y", "z", "r", "parent"])
        else:
            raise ValueError(f"> Unknown tree `{tree}` - use \"simplified\" or \"raw\".")

        # Every score has to land on a node - a score without a node comes from an older version of the tree
        n_missing = int((~scores["node_id"].isin(swc["node_id"])).sum())
        if n_missing:
            raise ValueError(f"> Neuron {neuron_id}: {n_missing} scored nodes do not exist in the {tree} tree.")

        # Left join -> leaves and nodes without a sub-tree keep null in the score columns
        swc_scored = pd.merge(left=swc, right=scores, on="node_id", how="left")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        swc_scored.to_csv(output_path, index=False)
        stats[f"n_nodes_{tree}"] = len(swc_scored)

    return stats


def _assign_worker(neuron_id: str, **kwargs) -> tuple:
    """joblib worker of `assign_unified_clumpiness` - never raises, returns (neuron_id, stats | None, error | None)."""
    try:
        return neuron_id, assign_swc_clumpiness(neuron_id=neuron_id, **kwargs), None
    except Exception as err:
        return neuron_id, None, f"{type(err).__name__}: {err}"


####################################################
####################################################
def assign_unified_clumpiness(unified_filepath: str,
                              simplified_dir: str,
                              raw_dir: str,
                              output_dir: str,
                              trees: tuple = ("simplified", "raw"),
                              overwrite: bool = False,
                              n_jobs: int = 1) -> dict:
    """
    Assigns the scores of the unified clumpiness file to the SWC trees of every neuron in it (see `assign_swc_clumpiness`).

    unified_filepath: str -> unified clumpiness csv: neuron_id, node_id (0 = root) + one column per label combination.
    simplified_dir: str -> folder of the simplified SWC csv files (`<neuron_id>.csv`).
    raw_dir: str -> folder of the raw SWC files (`<neuron_id>.swc`).
    output_dir: str -> output folder, the files are saved as `<output_dir>/<simplified | raw>/<neuron_id>.csv`.
    trees: tuple -> which trees to write, "simplified" and / or "raw".
    overwrite: bool -> if False, existing output files are kept.
    n_jobs: int -> number of neurons processed in parallel.
    Returns a dict with the number of neurons, written / skipped neurons, scored nodes and the failed neurons.
    """
    unified = pd.read_csv(unified_filepath, dtype={"neuron_id": str})

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_assign_worker)(neuron_id,
                                scores = scores.drop(columns="neuron_id"),
                                simplified_path = os.path.join(simplified_dir, f"{neuron_id}.csv"),
                                raw_path = os.path.join(raw_dir, f"{neuron_id}.swc"),
                                output_dir = output_dir,
                                trees = trees,
                                overwrite = overwrite)
        for neuron_id, scores in tqdm(unified.groupby("neuron_id", sort=True), desc="Assigning clumpiness", unit=" neurons")
    )

    written = [stats for _, stats, _ in results if stats is not None]
    failed = {neuron_id: error for neuron_id, _, error in results if error is not None}
    if failed:
        print(f"> {len(failed)} neurons failed in the score assignment (see the returned `failed`).")

    return {"neurons": len(results),
            "written": len(written),
            "skipped": len(results) - len(written) - len(failed),
            "scored_nodes": int(sum(stats["n_scored"] for stats in written)),
            "failed": failed}
