from scripts.preprocessing import get_neurons_info
from scripts.pipeline import process_neuron
from scripts.helpers import read_json

from joblib import Parallel, delayed
from tqdm import tqdm
import polars as pl
import numpy as np
import os


if __name__ == '__main__':
    print("-----------------------------------------------------------------")
    print("> Loading paths and creating folders. (1/5)")
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
    print("> Creating unified labels file for every relevent SWC tree. (2/5)")
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
    print(f"Starting processing of {len(tasks)} neurons. (3/5)")

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
    json_path = os.path.join("data", "output_json")
    jsons_2process = os.listdir(json_path)
    print(f"> Calculating clumpiness per JSON tree file ({len(jsons_2process)} trees). (4/5)")
    

