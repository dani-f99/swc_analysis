from scripts.processing import join_swc_clumpiness
from scripts.helpers import read_json

from joblib import Parallel, delayed
import pyarrow.dataset as ds
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os


n_steps=6
config = read_json()
n_jobs = int(config["n_jobs"])


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


