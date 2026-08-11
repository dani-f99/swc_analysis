from scripts.preprocessing import simplify_swc_topology, swc2json, get_neurons_info
from scripts.processing import generate_internal_subtrees
from scripts.helpers import mkdir

from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import polars as pl
import numpy as np
import json
import os


# Force Polars to use a single thread to prevent nested parallelism crashes
os.environ["POLARS_MAX_THREADS"] = "4"

def process_neuron(neuron_itr):
    try:
        #########################
        #  1. Load labels parquet
        # Parquet labels path
        prquet_labels_path = os.path.join("data", "input_labels", "swc_labels.parquet")


        # Load exactly the labels of the example swc file
        parquet_labels = pl.scan_parquet(prquet_labels_path)


        # Only the relevnt column in the parquet file
        labels_parquet = parquet_labels.select(["neuron", "node_id", "type"]).filter(pl.col("neuron") == str(neuron_itr)).collect().to_pandas()


        #########################
        #  2. Import the swc file
        neuron_path = os.path.join("data","input_swc", "sk_lod1_783_healed", f"{neuron_itr}.swc")
        neuron_swc = pd.read_csv(neuron_path, 
                                    comment='#', 
                                    header=None, 
                                    sep=r'\s+', 
                                    names=["node_id", "swc_type", "x", "y", "z", "r", "parent"])


        #######################
        #  3. simplify swc file
        simple_swc = simplify_swc_topology(neuron_swc, swc_name=f"{neuron_itr}", save_csv=False)  


        #########################################
        #  4. attach synapse labels + neuron type
        swc_labeled = pd.merge(left=simple_swc, 
                                right=labels_parquet[["node_id", "type"]].drop_duplicates(), 
                                left_on="node_id", 
                                right_on="node_id", 
                                how="left")   


        ##########################
        #  5. save simplified file
        for i in ["data", os.path.join("data", "input_swc"), os.path.join("data", "input_swc", "simplified")]:
            if os.path.exists(i) is False:
                os.mkdir(i)

        save_path = os.path.join("data", "input_swc", "simplified", f"{neuron_itr}.csv")
        swc_labeled["type"] = swc_labeled.groupby("node_id")["type"].unique().apply(lambda X : X[0] if len(X) <= 1 else ",".join(X)) # joining labels if more then 2 per node
        swc_labeled = swc_labeled.drop_duplicates(subset=["node_id", "parent"], keep="first")                                        # dropping rows with the same parent+node_id
        swc_labeled.to_csv(save_path)


        #########################################
        #  6. convert to json for find-clumpiness
        swc2json(swc_dataset=swc_labeled.drop_duplicates(),
                    neuron_id=neuron_itr,
                    save_json=True,
                    save_path=os.path.join("data", "output_json"))


        ##################################################################
        # 7. Devide main tree to multiple sub-trees for each internal node
        # Example Usage:
        generate_internal_subtrees(input_json_path = os.path.join("data","output_json",f"{neuron_itr}_0.json"), 
                                    neuron_number = neuron_itr, 
                                    output_dir = os.path.join("data", "output_json"))
        

        return f"Success: {neuron_itr}"

    except Exception as e:
        return f"Error on {neuron_itr}: {e}"