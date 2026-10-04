from scripts.helpers import mkdir, read_json, TimeStamp, log_to_report
from scripts.preprocessing import simplify_swc_topology, swc2json
from scripts.processing import generate_internal_subtrees


import pandas as pd
import polars as pl
import unittest
import os


class PipelineSWC(unittest.TestCase):
    """
    Pipeline that process SWC files, simplify, attaching relevent metadata and calculating clumpiness.
    """

    ##################
    # Class initiation 
    @classmethod
    def setUpClass(cls):
        print("> Initializing Preprocessing Pipeline Environment")

            # Loading config presets
        config = read_json(path="config.json")
        cls.run_name, cls.swc_path, cls.labels_path, cls.prquet_labels_path, cls.overwrite_info, cls.data_limit, cls.n_threads, cls.n_jobs =  [config["run_name"],
                                                                                                                                               config["swc_path"].split(","),
                                                                                                                                               config["labels_path"].split(","),
                                                                                                                                               config["prquet_labels_path"].split(","),
                                                                                                                                               config["overwrite_info"],
                                                                                                                                               config["data_limit"],
                                                                                                                                               config["n_threads"],
                                                                                                                                               config["n_jobs"]]
        # Getting the pipeline run time
        cls.pipeline_time = TimeStamp()
        cls.pipeline_time.set_start()
        

        # Creating required folders (if dosent exists)
        mkdir(["data", "input_swc", "input_labels", "output_json", "output_clumpiness", "output_results", "reports"])

        # Creating report file for the run
        log_to_report(file_path = os.path.join("data", "reports"), 
                      message = f"0. Pipeline run {cls.run_name} initialized successfully (start time: {cls.pipeline_time.start}).", 
                      add_timestamp= False)


    # Step 1
    def test_01_translate_convert_extract(self):
        #  1. Load labels parquet
        # Parquet labels path
        prquet_labels_path = os.path.join("data", "input_labels", "swc_labels.parquet")

        # Load exactly the labels of the example swc file
        parquet_labels = pl.scan_parquet(prquet_labels_path)

        # Only the relevnt column in the parquet file
        labels_parquet = parquet_labels.select(["neuron", "node_id", "type"]).filter(pl.col("neuron") == str(neuron_itr)).collect().to_pandas()

        