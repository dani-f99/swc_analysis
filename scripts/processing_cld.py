#########
# Fixed copy of `scripts/processing.py` - see `BUGFIX_REPORT_claude.md` for the full change log.
# Imports
from scripts.helpers import  chunked_iterable

from joblib import Parallel, delayed
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pandas as pd
import subprocess
import uuid
import json
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

    if (overwrite is False) and (os.path.exists(output_csv)):
        return None

    try:
        # Opening in 'w' mode automatically overwrites the file if it already exists
        with open(output_csv, 'w') as f_out:
            subprocess.run(
                ["find-clumpiness", "-e", "AllExclusive", "-i", str(filepath), "-f", "JSON"],
                stdout=f_out,             # Dumps output straight to the file
                stderr=subprocess.PIPE,   # Catches errors so they don't print to terminal
                text=True,
                check=True
            )
        return True

    # [FIX 6c] The reason of the failure (find-clumpiness stderr / missing executable) used to be discarded.
    except subprocess.CalledProcessError as err:
        if output_csv.exists():
            output_csv.unlink()
        raise Exception(f"> find-clumpiness failed on `{filepath}`: {(err.stderr or '').strip()}") from err

    except FileNotFoundError as err:
        if output_csv.exists():
            output_csv.unlink()
        raise Exception(f"> Failed to process clumpiness on `{filepath}` - is `find-clumpiness` on PATH? ({err})") from err

    except Exception as err:
        # If it fails, delete the empty/partial CSV so it doesn't leave corrupted data
        if output_csv.exists():
            output_csv.unlink()
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
