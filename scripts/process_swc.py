# Fixed copy of the `ProcessSWC` class from `notebook_test.ipynb` - see `_archive/BUGFIX_REPORT.md` for the full change log.
# Import - Custom scripts
from scripts.helpers import read_json
from scripts.visualize import plot_scored_neurons
from scripts.preprocessing import get_neurons_info, simplify_swc_topology, swc2json, attach_node_labels
from scripts.processing import generate_internal_subtrees, process_single_clumpiness, compile_wide_clumpiness, assign_unified_clumpiness

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
    Step 7 - Joining the clumpiness results of all neurons into a single unified file (one row per node, one column per label combination).
    Step 8 - Assigning the clumpiness scores to the root and the internal nodes of each neuron's SWC file (simplified / raw - based on node id).
    Step 9 - Saving the interactive topology plot of each neuron as an html file (simplified / raw).

    Overwrite rule (same for every per-neuron step): if `overwrite_info` is False and the step's output exists, the step is
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
                           "7-swc_clumpiness_result": str(run_dir / "7-swc_clumpiness_result"),
                           "8-swc_clumpiness": str(run_dir / "8-swc_clumpiness"),
                           "9-swc_clumpiness_plots": str(run_dir / "9-swc_clumpiness_plots")
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
    def step06_clumpiness_calculation(self,
                                       neuron_id: str | int) -> dict:
        """
        6th step, calculating the clumpiness for each json file - the main tree and every sub-tree of step 5.
        Output: `6-clumpiness_scores/<neuron_id>/<json file name>.csv` (`<neuron_id>.csv` is the main tree).
        neuron_id: str | int -> neuron id number of the file in the `swc_input` folder.
        """

        neuron_id = str(neuron_id)

        # Required paths - main tree first, then its sub-trees
        path_main = os.path.join(self.dict_paths["4-json"], f"{neuron_id}.json")
        path_div = os.path.join(self.dict_paths["5_json_divided"], neuron_id)
        paths2process = [path_main] + sorted(os.path.join(path_div, i) for i in os.listdir(path_div) if i.endswith(".json"))
        output_path = os.path.join(self.dict_paths["6-clumpiness_scores"], neuron_id)
        os.makedirs(output_path, exist_ok=True)

        # Scores of trees that no longer exist (older tree version) are removed, same as the sub-trees in step 5
        tree_names = {Path(i).stem for i in paths2process}
        for stale_file in Path(output_path).glob("*.csv"):
            if stale_file.stem not in tree_names:
                stale_file.unlink()

        # Clumpiness calculation - an existing csv is kept when overwrite is False
        n_written = 0
        for i in paths2process:
            written = process_single_clumpiness(filepath = i,
                                                output_dir = output_path,
                                                overwrite = self.dict_config["overwrite_info"])
            n_written += int(written is True)

        if n_written == 0:
            return {"skipped": True}

        return {"n_trees": len(paths2process),
                "n_clumpiness_written": n_written}



    ##################################
    ############# STEP 7 #############
    ##################################
    def step07_result_annealing(self) -> dict:
        """
        7th step, joining the clumpiness results of every neuron (step 6) into a single unified file.
        Output: `7-swc_clumpiness_result/unified_clumpiness.csv` - one row per tree with neuron_id,
        node_id (0 = root / main tree) and one `<label1>_<label2>` column per label combination (null if not found).
        Not a per-neuron step, and always rebuilt - it is an aggregate of whatever step 6 has produced so far.
        """

        save_path = os.path.join(self.dict_paths["7-swc_clumpiness_result"], "unified_clumpiness.csv")

        return compile_wide_clumpiness(input_directory = self.dict_paths["6-clumpiness_scores"],
                                       output_filepath = save_path,
                                       n_jobs = self.dict_config["n_jobs"])



    ########################################
    ################ STEP 8 ################
    ########################################
    def step08_score_assignment(self,
                                trees: tuple = ("simplified", "raw")) -> dict:
        """
        8th step, assigning the clumpiness scores of the unified file (step 7) to the root and the internal nodes
        of each neuron's SWC file - every other node stays null. The join is based on the node id.
        Output: `8-swc_clumpiness/<simplified | raw>/<neuron_id>.csv` - the SWC nodes + one column per label combination.
        Not a per-neuron step - it runs on every neuron of the unified file.
        trees: tuple -> which trees to write: "simplified" (step 3 file, with the labels) and / or "raw" (input swc file).
        """

        unified_path = os.path.join(self.dict_paths["7-swc_clumpiness_result"], "unified_clumpiness.csv")

        return assign_unified_clumpiness(unified_filepath = unified_path,
                                         simplified_dir = self.dict_paths["3-swc_simplified_labeled"],
                                         raw_dir = self.dict_config["swc_path"],
                                         output_dir = self.dict_paths["8-swc_clumpiness"],
                                         trees = trees,
                                         overwrite = self.dict_config["overwrite_info"],
                                         n_jobs = self.dict_config["n_jobs"])



    ########################################
    ################ STEP 9 ################
    ########################################
    def step09_plot_creation(self,
                             trees: tuple = ("simplified", "raw")) -> dict:
        """
        9th step, saving the interactive topology plot of each neuron (step 8 file) as a stand-alone html file -
        node fill = clumpiness score, shape and colour = label. See `plot_neuron_topology` in `scripts/visualize.py`.
        Output: `9-swc_clumpiness_plots/<simplified | raw>/<neuron_id>.html`, the same layout as step 8.
        Not a per-neuron step - it runs on every neuron of the step 8 folder.
        trees: tuple -> which trees to plot: "simplified" and / or "raw".
        """

        return plot_scored_neurons(scored_dir = self.dict_paths["8-swc_clumpiness"],
                                   labels_dir = self.dict_paths["3-swc_simplified_labeled"],
                                   output_dir = self.dict_paths["9-swc_clumpiness_plots"],
                                   trees = trees,
                                   overwrite = self.dict_config["overwrite_info"],
                                   n_jobs = self.dict_config["n_jobs"])
