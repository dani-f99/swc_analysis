#########
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
                               overwrite: bool = False, 
                               ) -> None:
    """
    Parses a nested JSON tree of a neuron and exports all internal sub-trees 
    (excluding the main root and leaves) into individual JSON files.
    
    input_json_path: str -> File path of the json file to be processed.
    neuron_number: str -> String which represents the neuron id.
    overwrite: bool -> If argument 'overwrite' is True, will overwrite data if already exists (Defualt is False). 
    output_dir: str -> String path of the output folder to which the processed file will be saved to.
    """
    input_path = Path(input_json_path)
    out_dir = Path(output_dir)
    
    # Create the output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the original JSON tree
    with open(input_path, 'r') as f:
        tree_data = json.load(f)

    def traverse(current_node, is_main_root=False):
        # Validate node structure: [{"nodeID": "...", "nodeLabels": [...]}, [children]]
        if not current_node or len(current_node) != 2:
            return

        node_info, children = current_node
        node_id = node_info.get("nodeID")

        # A node is internal if the children array is not empty
        if children:
            # Export the sub-tree if it is NOT the absolute root
            if not is_main_root:
                output_filename = f"{neuron_number}_{node_id}.json"
                output_filepath = out_dir / output_filename

                # Only write if overwrite is True OR the file doesn't exist
                # Using pathlib's .exists() is cleaner than os.path.exists()
                if overwrite or not output_filepath.exists():
                    with open(output_filepath, 'w') as out_f:
                        json.dump(current_node, out_f)

            # ALWAYS recursively process all children, even if we skipped writing the parent
            for child in children:
                traverse(child, is_main_root=False)

    # Initiate traversal, explicitly flagging the first node as the main root
    traverse(tree_data, is_main_root=True)


#####################################
#####################################
def _process_single_neuron(filepath:str):
    """
    Worker function executed in parallel. 
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
        return

    else:
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
            
        except (subprocess.CalledProcessError, FileNotFoundError, Exception):
            # If it fails, delete the empty/partial CSV so it doesn't leave corrupted data
            if output_csv.exists():
                output_csv.unlink()

            raise Exception("> Failed to process clumpiness")


##############################################
##############################################
def process_clumpiness_csv(filepath_str: str):
    """
    Reads a single CSV, safely skips empty files, and appends ID columns.

    filepath_str: str ->
    """
    filepath = Path(filepath_str)
    
    # Fast-skip: Files <= 32 bytes physically cannot contain data rows
    if os.path.getsize(filepath) <= 32:
        return None
        
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return None
            
        neuron_id, node_id = filepath.stem.split('_')
        
        df['neuron_id'] = neuron_id
        df['node_id'] = node_id
        
        return df
        
    except Exception:
        return None


################################################
################################################
def compile_unified_dataset(input_directory: str, 
                            output_filepath: str, 
                            batch_size: int = 2000) -> None:
    """
    Iterates over CSVs and processes them using joblib for robust parallel execution.

    input_directory: str -> String input clumpiness csv's folder.
    output_filepath: str -> String output of the joined clumpiness file.
    batch_size: str -> Number of itirations per batch of processing.
    """

    def get_csv_files():
        with os.scandir(input_directory) as entries:
            for entry in entries:
                if entry.name.endswith('.csv') and entry.is_file():
                    yield entry.path

    writer = None
    n_jobs = min(4, (os.cpu_count() or 1))
    
    with tqdm(desc="Compiling Parquet", unit=" files") as pbar:
        for file_chunk in chunked_iterable(get_csv_files(), batch_size):
            
            # joblib handles the worker pool much more safely on Windows
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(process_clumpiness_csv)(f) for f in file_chunk
            )
            
            valid_dfs = [df for df in results if df is not None]
            
            if valid_dfs:
                batch_df = pd.concat(valid_dfs, ignore_index=True)
                table = pa.Table.from_pandas(batch_df)
                
                if writer is None:
                    writer = pq.ParquetWriter(output_filepath, table.schema, compression='ZSTD')
                
                writer.write_table(table)
            
            pbar.update(len(file_chunk))
            
    if writer:
        writer.close()
    print("> Dataset compilation complete.")

#################################################################################################################
# > Worker function for joblib that joins SWCs file with their appropriate clumpiness results by neuron_id label.
def join_swc_clumpiness(neuron_id: str , 
                        path_2process: str, 
                        save_path: str, 
                        parquet_path: str):

    """
    neuron_id: str -> neuron id as string file.
    path_2process: str -> path to the target swc files (to which the labels will be joined).
    save_path: str -> path to the save output folder.
    parquet_path -> path to the labels parquet file.
    
    """
    i = neuron_id
    output_path = os.path.join(save_path, f"{i}.csv")

    # Early exit if the file already exists
    if os.path.exists(output_path):
        return

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
        return
        
    i_label['node_id'] = i_label['node_id'].astype("int")

    # Creating a unified labels column named 'label'
    i_label.insert(loc=2, 
                   column="label",
                   value=i_label['property1'] + "_" + i_label['property2'])

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