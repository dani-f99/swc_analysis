from scripts.preprocessing import get_neurons_info
from scripts.pipeline import process_neuron

from joblib import Parallel, delayed
from tqdm import tqdm
import polars as pl
import numpy as np
import os


if __name__ == '__main__':
    # Define paths
    swc_path = os.path.join("data", "input_swc", "sk_lod1_783_healed")
    labels_path = os.path.join("data", "input_labels", "processed_swc_data_princeton")
    prquet_labels_path = os.path.join("data", "input_labels", "swc_labels.parquet")

    # Safe folder creation before multiprocessing starts to avoid race conditions
    folders_to_create = ["data", 
                            os.path.join("data", "input_swc"), 
                            os.path.join("data", "input_swc", "simplified"),
                            os.path.join("data", "output_json")]

    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)

    # Getting a list of the aviable SWC file in the swc input folder
    swc_files = [i.split(".")[0] for i in os.listdir(swc_path)]

    # Creating parquete file -> only relevent swc file by super-type
    get_neurons_info(overwrite_parquet=False)

    # Getting relevent swc
    parquet_labels = pl.scan_parquet(prquet_labels_path)
    labels_parquet = parquet_labels.select(["neuron"]).collect().to_pandas()

    # Getting list of relevent + real SWC file
    swc_relv = np.intersect1d(swc_files, labels_parquet.neuron.values)

    # Select the batch you want to run
    tasks = swc_relv[:20] 

    print(f"Starting processing of {len(tasks)} neurons...")

    # Execute in parallel using Joblib
    # n_jobs=4 limits the pool to 4 cores to prevent memory exhaustion. 
    # You can increase this if your system has plenty of RAM.
    results = Parallel(n_jobs=4, backend="loky")(delayed(process_neuron)(neuron) for neuron in tqdm(tasks))

    # Print any errors that were caught during execution
    for res in results:
        if "Error" in res:
            print(res)

    # Create clumpiness file results
    #....
