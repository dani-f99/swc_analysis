from scripts.preprocessing import get_neurons_info
from scripts.processing import process_single_clumpiness, compile_unified_dataset, join_swc_clumpiness
from scripts.pipeline import process_neuron
from scripts.helpers import read_json

from joblib import Parallel, delayed
import pyarrow.dataset as ds
from pathlib import Path
from tqdm import tqdm
import polars as pl
import numpy as np
import os

n_steps = 6

if __name__ == '__main__':
    print("-----------------------------------------------------------------")
    print(f"> Loading paths and creating folders. (1/{n_steps})")
    # Loading config presets
    config = read_json(path="config.json")
    swc_path, labels_path, prquet_labels_path, overwrite_info, data_limit, n_threads, n_jobs =  [config["swc_path"].split(","),
                                                                                                 config["labels_path"].split(","),
                                                                                                 config["prquet_labels_path"].split(","),
                                                                                                 config["overwrite_info"],
                                                                                                 config["data_limit"],
                                                                                                 config["n_threads"],
                                                                                                 config["n_jobs"]]

    bool_val = overwrite_info.lower()
    overwrite_par = (True if bool_val == "true" else False)

    # Force Polars to use a single thread to prevent nested parallelism crashes
    os.environ["POLARS_MAX_THREADS"] = str(n_threads)

    # Safe folder creation before multiprocessing starts to avoid race conditions
    folders_to_create = ["data", 
                         os.path.join("data", "input_swc"), 
                         os.path.join("data", "input_swc", "simplified"),
                         os.path.join("data", "output_json")]
    
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)


    swc_path = os.path.join(*swc_path)
    labels_path = os.path.join(*labels_path)
    prquet_labels_path = os.path.join(*prquet_labels_path)
    print("> Files and paths loaded.")

    print("-----------------------------------------------------------------")
    print(f"> Creating unified labels file for every relevent SWC tree. (2/{n_steps})")
    # Getting a list of the aviable SWC file in the swc input folder
    swc_files = [i.split(".")[0] for i in os.listdir(swc_path)]

    # Creating parquete file -> only relevent swc file by super-type
    get_neurons_info(overwrite_parquet=overwrite_par)

    # Getting relevent swc
    parquet_labels = pl.scan_parquet(prquet_labels_path)
    labels_parquet = parquet_labels.select(["neuron"]).collect().to_pandas()

    # Getting list of relevent + real SWC file
    swc_relv = np.intersect1d(swc_files, labels_parquet.neuron.values)

    # Limit the number of tasks to run
    try:
        n_tasks = int(data_limit)
    except:
        n_tasks = None

    if isinstance(n_tasks, int):
        tasks = swc_relv[:n_tasks]
    else:
        tasks = swc_relv

    print("-----------------------------------------------------------------")
    print(f"Starting processing of {len(tasks)} neurons. (3/{n_steps})")

    # Execute in parallel using Joblib
    # n_jobs=4 limits the pool to 4 cores to prevent memory exhaustion. 
    # You can increase this if your system has plenty of RAM.
    results = Parallel(n_jobs=int(n_jobs), backend="loky")(delayed(process_neuron)(neuron) for neuron in tqdm(tasks))

    # Print any errors that were caught during execution
    for res in results:
        if "Error" in res:
            print(res)

    # Calculating Clumpiness
    print("-----------------------------------------------------------------")

    # Define paths
    output_clumpiness = os.path.join("data", "output_clumpiness")
    json_path = os.path.join("data", "output_json")
    jsons_2process = os.listdir(json_path)
    input_jsons = [os.path.join(json_path, i) for i in jsons_2process]
    print(f"> Calculating clumpiness per JSON tree file ({len(jsons_2process)} trees). (4/{n_steps})")

    
    json_files = jsons_2process
    
    if not json_files:
        print("> Error")
        raise Exception(f"No JSON files found in `{json_path}`.")

    
    # 1. Define the delayed tasks
    tasks = (delayed(process_single_clumpiness)(filepath, output_clumpiness) for filepath in input_jsons)
    
    # 2. Execute with return_as="generator" so it yields as each file finishes
    parallel_runner = Parallel(n_jobs=n_jobs, return_as="generator")(tasks)
    
    # 3. Wrap the execution in tqdm to track true completion
    for _ in tqdm(parallel_runner, total=len(json_files), desc="Calculating Clumpiness", ncols=100):
        pass # The actual work is done in the background; we just iterate to update the bar


    print("-----------------------------------------------------------------")
    input_dir = os.path.join("data", "output_clumpiness")
    output_file = os.path.join("data", "unified_clumpiness.parquet")
    print(f"> Joining clumpiness data to unified file at {output_file}. (5/{n_steps})")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    compile_unified_dataset(input_dir, output_file)


    print("-----------------------------------------------------------------")
    print(f"> Assigning clumpiness results into the simplified SWC files. (6/{n_steps})")

    # Define paths (from config)
    path_2process, parquet_path = [os.path.join(*config["results_swc_path"].split(",")), 
                                   os.path.join(*config["results_labels_path"].split(","))]

    save_path = os.path.join("data", "output_results", Path(path_2process).name)

    # Ensure the output directory exists before spawning workers.
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Get list of relevant neurons (from the parquet)
    dataset = ds.dataset(parquet_path, format="parquet")
    table_neurons = dataset.to_table(columns=["neuron_id"]).to_pandas()
    df_neuronsids = table_neurons['neuron_id'].unique()

    # List of SWC files
    swc_dir = [i.split(".")[0] for i in os.listdir(path_2process)]

    # Neurons to process
    relv_files = np.intersect1d(df_neuronsids, swc_dir)

    # > Iterating over the SWC files in parallel
    # n_jobs=-1 tells joblib to use all available CPU cores
    _capture = Parallel(n_jobs=n_jobs)(delayed(join_swc_clumpiness)(i, path_2process, save_path, parquet_path) 
                                   for i in tqdm(relv_files, desc="Dispatching Tasks"))


    print("-----------------------------------------------------------------")
    print("> Processing complete!")
    

