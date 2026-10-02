# Fixed copy of `scripts/pipeline.py` - see `BUGFIX_REPORT_claude.md` for the full change log.
from scripts.preprocessing_claude import simplify_swc_topology, swc2json, attach_node_labels
from scripts.processing_claude import generate_internal_subtrees
from scripts.helpers import read_json


import pandas as pd
import polars as pl
import os


def process_neuron(neuron_itr: str,
                   labels_parquet_path: str,
                   swc_neurons_path: str,
                   swc_simp_path: str,
                   json_path: str
                   ):
    """
    Processing pipeline of SWC files.
    neuron_itr:str -> neuron id to process.
    labels_parquet_path:str -> path to the processed labels parquet file (output).
    swc_neurons_path:str -> path to the neurons swc files folder (input).
    swc_simp_path:str -> path to the simplified swc file folders (output).
    json_path:str -> path to the JSON output folder(output).
    """
    try:
        #########################
        #  1. Load labels parquet
        # Load exactly the labels of the example swc file
        parquet_labels = pl.scan_parquet(labels_parquet_path)

        # Only the relevnt column in the parquet file
        labels_parquet = parquet_labels.select(["neuron", "node_id", "type"]).filter(pl.col("neuron").cast(pl.Utf8) == str(neuron_itr)).collect().to_pandas()


        #########################
        #  2. Import the swc file
        neuron_path = os.path.join(swc_neurons_path, f"{neuron_itr}.swc")
        neuron_swc = pd.read_csv(neuron_path,
                                 comment='#',
                                 header=None,
                                 sep=r'\s+',
                                 names=["node_id", "swc_type", "x", "y", "z", "r", "parent"])


        #######################
        #  3. simplify swc file
        # [FIX 2] synapse-labelled nodes are never removed by the simplification
        simple_swc = simplify_swc_topology(swc_input=neuron_swc,
                                           swc_name=f"{neuron_itr}",
                                           output_path=swc_simp_path,
                                           save_csv=False,
                                           keep_nodes=labels_parquet["node_id"].dropna().unique())


        #########################################
        #  4. attach synapse labels + neuron type
        # [FIX 1] node_id-aligned labels (was: merge + misaligned groupby assignment)
        swc_labeled = attach_node_labels(simple_swc, labels_parquet)


        ##########################
        #  5. save simplified file
        save_path = os.path.join(swc_simp_path, f"{neuron_itr}.csv")

        overwrite_info = read_json(path="config.json")["overwrite_info"]
        overwrite_par = str(overwrite_info).strip().lower() == "true"

        if (overwrite_par) or (os.path.exists(save_path) is False):
            swc_labeled.to_csv(save_path)


        #########################################
        #  6. convert to json for find-clumpiness
        swc2json(swc_dataset=swc_labeled,
                 neuron_id=neuron_itr,
                 save_json=True,
                 save_path=json_path,
                 overwrite=overwrite_par)


        ##################################################################
        # 7. Devide main tree to multiple sub-trees for each internal node
        # [FIX 7] swc2json writes `<id>.json`; the old code read `<id>_0.json`, which no longer exists.
        generate_internal_subtrees(input_json_path = os.path.join(json_path, f"{neuron_itr}.json"),
                                   neuron_number = neuron_itr,
                                   output_dir = json_path,
                                   overwrite=overwrite_par)


        return f"Success: {neuron_itr}"

    except Exception as e:
        return f"Error on {neuron_itr}: {e}"
