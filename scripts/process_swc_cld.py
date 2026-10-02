# Fixed copy of the `ProcessSWC` class from `notebook_test.ipynb` - see `BUGFIX_REPORT_claude.md` for the full change log.
# Import - Custom scripts
from scripts.helpers import read_json
from scripts.preprocessing_claude import get_neurons_info, simplify_swc_topology, swc2json, attach_node_labels
from scripts.processing_claude import generate_internal_subtrees

# Imports - python
from pathlib import Path
import pandas as pd
import polars as pl
import os



#######################
# Config value parsers
def _parse_bool(value) -> bool:
    """`bool("False")` is True - parse the text instead. Accepts real booleans too."""
    return str(value).strip().lower() == "true"


def _parse_limit(value) -> int | None:
    """"None" / "" / null -> None (no limit), anything else -> int."""
    if value is None or str(value).strip().lower() in ("none", "null", ""):
        return None
    return int(value)



# Pipeline class
class ProcessSWC():
    """
    SWC process steps:
    Step 0 - Preparing for pipeline run, creating folder, creating report, loading labels file (if exists).
    Step 1 - Creating parquete labels file. The purpose of this file is to label each SWC neuron (by class criteria) and each synapse (post / pre).
    Step 2 - SWC file simplification. Removing internal nodes without external leaf.
    Step 3 - Adding labels to the simplified SWC file.
    Step 4 - Covnerting simplified SWC file + labels to JSON format for the clumpiness calculation.
    Step 5 - Dividing each json to multiple sub-jsons (sub-trees) of each internal node.
    Step 6 - Clumpiness calculation via the find-clumpiness program.
    Step 7 - Assigning clumpiness scores to the internal nodes of the SWC tree (simplified / non-simplified tree - based on node id).

    Overwrite rule (same for every step): if `overwrite_info` is False and the step's output exists, the step is
    skipped and returns {"skipped": True}. Otherwise the output is (re)written.
    Every per-neuron step returns a small dict of statistics, to be collected into a per-neuron record.
    """



    #####################################################
    ###################### STEP 0 #######################
    #####################################################
    def __init__(self, config_file: str = "config.json", keep_labeled_nodes: bool = True):
        """
        Initiating process step of the class.
        config_file: str -> path to the `config.json` file. Relative paths inside the config are resolved
                            against the config file's folder (not the current working directory).
        keep_labeled_nodes: bool -> if True (default), step 2 never removes synapse-labelled nodes, so no label is lost
                                    by the simplification. False = old behaviour (root / leaves / branch points only).
        """
        self.keep_labeled_nodes = keep_labeled_nodes

        config = read_json(path=config_file)
        base_dir = Path(config_file).resolve().parent

        self.dict_config = {
                            "run_name":       config["run_name"],                                              # string value of the pipeline run name
                            "swc_path":       str(base_dir.joinpath(*config["swc_path"].split(","))),          # path of the raw swc files folder
                            "labels_path":    str(base_dir.joinpath(*config["labels_path"].split(","))),       # path of the nodes labels folder (one sub-folder with `connectors.pkl` per batch)
                            "labels_ftr_path": str(base_dir.joinpath(*config["labels_ftr_path"].split(","))),  # path of the neurons labels .ftr file (super_class filter)
                            "overwrite_info": _parse_bool(config["overwrite_info"]),                           # overwrite files if the `run_name` haven't been changed
                            "data_limit":     _parse_limit(config["data_limit"]),                              # number (int) limit of files to process, None = all
                            "n_threads":      int(config["n_threads"]),                                        # number of threads to be used in the analysis
                            "n_jobs":         int(config["n_jobs"])                                            # number of jobs to process in parallel
                            }

        # Reading config for paths
        run_dir = base_dir / "output" / self.dict_config["run_name"]
        self.dict_paths = {
                           "0-run_log": str(run_dir / "0-run_log"),
                           "1-output_labels": str(run_dir / "1-output_labels"),
                           "2-swc_simplified": str(run_dir / "2-swc_simplified"),
                           "3-swc_simplified_labeled": str(run_dir / "3-swc_simplified_labeled"),
                           "4-json": str(run_dir / "4-json"),
                           "5_json_divided": str(run_dir / "5_json_divided"),
                           "6-clumpiness_scores": str(run_dir / "6-clumpiness_scores"),
                           "7-swc_clumpiness_result": str(run_dir / "7-swc_clumpiness_result")
                            }

        # Labels parquet path - defined here so workers never depend on `step01` having run in the same process
        self.labels_path = os.path.join(self.dict_paths["1-output_labels"], "swc_labels.parquet")

        # Creating the folders defined in `self.dict_paths`
        for folder in self.dict_paths.values():
            os.makedirs(folder, exist_ok=True)



    ###############################
    ########## STEP 1 #############
    ###############################
    def step01_create_labels(self) -> list:
        """
        First step of the pipeline after the initiation, creation of a
        parqute format labels file for each neuron in the folder.
        Returns the sorted list of neuron ids (str) that have both a raw SWC file and labels,
        cut to `data_limit` if set.
        """

        # Creating parquete file -> only relevent swc file by super-type.
        # `get_neurons_info` itself keeps an existing file when overwrite is False.
        get_neurons_info(parquet_path = self.dict_paths["1-output_labels"],
                         labels_path = self.dict_config["labels_path"],
                         swc_labels_file_path = self.dict_config["labels_ftr_path"],
                         overwrite_parquet = self.dict_config["overwrite_info"])

        # Getting a list of the aviable SWC file in the swc input folder (`.stem` only removes the last extension)
        swc_files = {Path(i).stem for i in os.listdir(self.dict_config["swc_path"]) if i.endswith(".swc")}

        # Getting relevent swc (ids compared as strings on both sides)
        labeled_neurons = set(pl.scan_parquet(self.labels_path)
                                .select(pl.col("neuron").cast(pl.Utf8))
                                .unique()
                                .collect()["neuron"]
                                .to_list())

        swc_relv = sorted(swc_files & labeled_neurons)

        return swc_relv[:self.dict_config["data_limit"]]   # [:None] -> all



    ##############################
    ########## STEP 2 ############
    ##############################
    def step02_simplify_swc(self,
                            neuron_id: str | int) -> dict:
        """
        Simplification step of the original raw SWC neuron file.
        neuron_id: str | int -> neuron id number of the file in the `swc_input` folder.
        """

        save_path = os.path.join(self.dict_paths["2-swc_simplified"], f"{neuron_id}.csv")
        if (self.dict_config["overwrite_info"] is False) and os.path.exists(save_path):
            return {"skipped": True}

        # Loading neuron file
        neuron_path = os.path.join(self.dict_config["swc_path"], f"{neuron_id}.swc")
        neuron_swc = pd.read_csv(neuron_path,
                                 comment='#',
                                 header=None,
                                 sep=r'\s+',
                                 names=["node_id", "swc_type", "x", "y", "z", "r", "parent"])

        # Synapse-labelled nodes of this neuron -> never removed by the simplification (report issue #2)
        labeled_nodes = None
        if self.keep_labeled_nodes:
            labeled_nodes = (pl.scan_parquet(self.labels_path)
                               .filter(pl.col("neuron").cast(pl.Utf8) == str(neuron_id))
                               .select(pl.col("node_id").drop_nulls().unique())
                               .collect()["node_id"]
                               .to_list())

        #Simplifing SWC file
        swc_simplified = simplify_swc_topology(swc_input = neuron_swc,
                                               swc_name = f"{neuron_id}",
                                               output_path = self.dict_paths["2-swc_simplified"],
                                               save_csv=True,
                                               keep_nodes=labeled_nodes)

        return {"n_nodes_raw": len(neuron_swc),
                "n_nodes_simplified": len(swc_simplified),
                "n_roots_simplified": int((swc_simplified["parent"] == -1).sum())}



    ###############################
    ########### STEP 3 ############
    ###############################
    def step03_label_joining(self,
                             neuron_id: str | int) -> dict:
        """
        3rd step in the pipeline - adding labels to the simplified swc file.
        neuron_id: str | int -> neuron id number of the file in the `swc_input` folder.
        """

        save_path = os.path.join(self.dict_paths["3-swc_simplified_labeled"], f"{neuron_id}.csv")
        if (self.dict_config["overwrite_info"] is False) and os.path.exists(save_path):
            return {"skipped": True}

        # Loading relevent rows from the labels praquet file
        labels_parquet = (pl.scan_parquet(self.labels_path)
                            .filter(pl.col("neuron").cast(pl.Utf8) == str(neuron_id))
                            .select(["node_id", "type"])
                            .collect()
                            .to_pandas())

        # Attach nodes labels to the simplified SWC files (aligned by node_id - see FIX 1)
        swc_simplified = pd.read_csv(os.path.join(self.dict_paths["2-swc_simplified"], f"{neuron_id}.csv"), index_col=0)
        swc_labeled = attach_node_labels(swc_simplified, labels_parquet)

        # Saving output file
        swc_labeled.to_csv(save_path)

        labeled_nodes_raw = labels_parquet["node_id"].nunique()
        labeled_nodes_kept = int(swc_labeled["type"].notna().sum())
        return {"label_rows": len(labels_parquet),
                "labeled_nodes_raw": int(labeled_nodes_raw),
                "labeled_nodes_kept": labeled_nodes_kept,                           # == raw when keep_labeled_nodes, unless label ids are missing from the SWC
                "labeled_nodes_lost": int(labeled_nodes_raw - labeled_nodes_kept)}



    ###############################
    ########### STEP 4 ############
    ###############################
    def step04_json_creation(self,
                             neuron_id: str | int) -> dict:
        """
        4th step in the pipeline - creating json file from the simplified swc + labels file.
        neuron_id: str | int -> neuron id number of the file in the `swc_input` folder.
        """

        # labels swc file import
        swc_labeled = pd.read_csv(os.path.join(self.dict_paths["3-swc_simplified_labeled"], f"{neuron_id}.csv"), index_col = 0)

        # JSON conversion function - raises on duplicated nodes, several roots or unreachable nodes
        stats = swc2json(swc_dataset = swc_labeled,
                         neuron_id = str(neuron_id),
                         save_json = True,
                         save_path = self.dict_paths["4-json"],
                         overwrite = self.dict_config["overwrite_info"])

        return stats if stats is not None else {"skipped": True}



    ###############################
    ########### STEP 5 ############
    ###############################
    def step05_internal_node_div(self,
                                 neuron_id: str | int) -> dict:
        """
        5th step of the pipeline. Creating seperate JSON file for each sub-tree whithin the main tree.
        This step is useful for the individual clumpiness scoring of each internal node.
        Output: `5_json_divided/<neuron_id>/<neuron_id>_<node_id>.json`.
        neuron_id: str | int -> neuron id number of the file in the `swc_input` folder.
        """

        # Required paths
        path_main_json = os.path.join(self.dict_paths["4-json"], f"{neuron_id}.json")

        # JSON split function
        n_subtrees = generate_internal_subtrees(input_json_path = path_main_json,
                                                neuron_number = str(neuron_id),
                                                output_dir = self.dict_paths["5_json_divided"],
                                                overwrite = self.dict_config["overwrite_info"])

        return {"n_subtrees": n_subtrees} if n_subtrees is not None else {"skipped": True}



    ########################################
    ################ STEP 6 ################
    ########################################
    def step06_clumpiness_calculation(self):
        """
        """
        pass



    ##################################
    ############# STEP 7 #############
    ##################################
    def step07_result_annealing(self):
        """
        """
        pass
