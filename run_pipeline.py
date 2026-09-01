# Custom functions
from scripts.preprocessing import get_neurons_info
from scripts.processing import process_single_clumpiness, compile_unified_dataset, join_swc_clumpiness
from scripts.pipeline import process_neuron
from scripts.helpers import read_json, TimeStamp, log_to_report

# Python modules
from joblib import Parallel, delayed
import pyarrow.dataset as ds
from pathlib import Path
from tqdm import tqdm
import polars as pl
import numpy as np
import traceback
import os



# Debugging configuration 
debug = True
n_steps = 6
report_spacer = "-----------------------------------------------------------------"

if debug:
    steps_bool = [True,  # Step 1 [V]
                  True,  # Step 2 [V]
                  True,  # Step 3 [V]
                  True,  # Step 4 [ ]
                  True, # Step 5 [ ]
                  True, # Step 6 [ ]
                 ]

else:
    steps_bool = [True] * n_steps

if __name__ == '__main__':
    ################################################################
    ### Step 1 - Preparing paths and required files for analysis ###
    ################################################################
    if steps_bool[0]:
        try:
            # Loading config presets
            config = read_json(path="config.json")
            run_name, swc_path, labels_path, overwrite_info, data_limit, n_threads, n_jobs =  [config["run_name"],
                                                                                               config["swc_path"].split(","),
                                                                                               config["labels_path"].split(","),
                                                                                               config["overwrite_info"],
                                                                                               config["data_limit"],
                                                                                               config["n_threads"],
                                                                                               config["n_jobs"]]



            print(report_spacer)
            # Setting time stamp calss for step 1
            time1 = TimeStamp()       
            time1.set_start()

            print(f"> Loading paths and creating folders. (1/{n_steps})")

            bool_val = overwrite_info.lower()
            overwrite_par = (True if bool_val == "true" else False)

            # Force Polars to use a single thread to prevent nested parallelism crashes
            os.environ["POLARS_MAX_THREADS"] = str(n_threads)

            # Safe folder creation before multiprocessing starts to avoid race conditions
            path_simplified_swc, path_clumpiness, path_json, path_results, path_labels = [os.path.join("output", run_name, "simplified_swc"),
                                                                                          os.path.join("output", run_name, "output_clumpiness"),
                                                                                          os.path.join("output", run_name, "output_json"),
                                                                                          os.path.join("output", run_name, "output_results"),
                                                                                          os.path.join("output", run_name, "output_labels")]
            
            path_report = os.path.join("output", f"{run_name}", "_reports", f"{run_name}_report.txt")

            folders_to_create = ["data",
                                "output",
                                os.path.join("output", run_name),
                                os.path.join("output", run_name, "_reports"), 
                                os.path.join("data", "input_swc"), 
                                path_simplified_swc,
                                path_clumpiness,
                                path_json,
                                path_results,
                                path_labels
                                ]


            
            for folder in folders_to_create:
                os.makedirs(folder, exist_ok=True)


            swc_path = os.path.join(*swc_path)
            labels_path = os.path.join(*labels_path)
            prquet_labels_path = os.path.join("output", run_name, "output_labels", "swc_labels.parquet")
            print("> Files and paths loaded.")


            # Setting up report & Getting end time 
            time1.set_end()
            time1_start, time1_end, time1_delta = time1.get_times()
            log_to_report(file_path=path_report,
                        message=f"""{report_spacer} 
                        \n> Initializing step 1 
                        \n> Start time: {time1_start} 
                        \n> End time: {time1_end} 
                        \n Elpased time: {time1_delta}.""",
                        add_timestamp=True,
                        mode="w")

        # Step 1 failure 
        except Exception as err:
            error_message = traceback.format_exc()
            log_to_report(file_path=path_report,
                          message=f"{report_spacer} \n>  step 1 Failed \n> Error messege:\n> {error_message}\n",
                          add_timestamp=True,
                          mode="w")

            raise Exception("> Step 1 Failed.")



    ############################################################
    ### Step 2 - Creating parqute file for the needed labels ###
    ############################################################
    if steps_bool[1]:
        try:
            # Setting time stamp calss for step 2
            time2 = TimeStamp()       
            time2.set_start()

            print(report_spacer)
            print(f"> Creating unified labels file for every relevent SWC tree. (2/{n_steps})")
            # Getting a list of the aviable SWC file in the swc input folder
            swc_files = [i.split(".")[0] for i in os.listdir(swc_path)]

            # Creating parquete file -> only relevent swc file by super-type
            get_neurons_info(parquet_path=os.path.join("output", run_name, "output_labels"),
                             labels_path=os.path.join("data", "input_labels"),
                             overwrite_parquet=overwrite_par)

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



            # Setting up report & Getting end time 
            time2.set_end()
            time2_start, time2_end, time2_delta = time2.get_times()
            log_to_report(file_path=path_report,
                          message=f"""{report_spacer} 
                                      \n> Step 2 finished successufly.
                                      \n> number of SWC files: {len(swc_files)}, files to process: {len(swc_relv)} ({round(len(swc_relv)/len(swc_files), 3)}). 
                                      \n> Start time: {time2_start} 
                                      \n> End time: {time2_end} 
                                      \n Elpased time: {time2_delta}.\n""",
                          add_timestamp=True)

        # Step 2 failure 
        except Exception as err:
            error_message = traceback.format_exc()
            log_to_report(file_path=path_report,
                          message=f"\n{report_spacer} \n>  step 2 Failed \n> Error messege:\n> {error_message}",
                          add_timestamp=True)

            raise Exception("> Step 2 Failed.")



    #########################################################################################
    ### Step 3 - Simplifing SWC neurons files, creating JSON representation for each tree ###
    #########################################################################################
    if steps_bool[2]:
        try:
            # Setting time stamp calss for step 3
            time3 = TimeStamp()       
            time3.set_start()

            print(report_spacer)
            print(f"Starting processing of {len(tasks)} neurons. (3/{n_steps})")

            # Execute in parallel using Joblib
            # n_jobs=4 limits the pool to 4 cores to prevent memory exhaustion. 
            # You can increase this if your system has plenty of RAM.
            results = Parallel(n_jobs=int(n_jobs), backend="loky")(delayed(process_neuron)(neuron,               # neuron id
                                                                                           prquet_labels_path,   # path to parquet label file
                                                                                           swc_path,             # path to raw swc files folder
                                                                                           path_simplified_swc,  # path to the simplified swc output folder
                                                                                           path_json             # path to the json output folder
                                                                                           ) for neuron in tqdm(tasks))

            # Print any errors that were caught during execution
            for res in results:
                if "Error" in res:
                    print(res)

            # Setting up report & Getting end time 
            time3.set_end()
            time3_start, time3_end, time3_delta = time3.get_times()
            log_to_report(file_path=path_report,
                          message=f"""{report_spacer} 
                                      \n> Step 3 finished successufly.
                                      \n> number of neurons files processed: {len(tasks)}. 
                                      \n> Start time: {time3_start} 
                                      \n> End time: {time3_end} 
                                      \n Elpased time: {time3_delta}.\n""",
                          add_timestamp=True)
                 
        # Step 3 failure
        except Exception as err:
            error_message = traceback.format_exc()
            log_to_report(file_path=path_report,
                          message=f"\n{report_spacer} \n>  step 3 Failed \n> Error messege:\n> {error_message}\n",
                          add_timestamp=True)

            raise Exception("> Step 3 Failed.")


    #########################################################################################
    ### Step 4 - Calcating clumpiness score for each internal node for each internal node ###
    #########################################################################################
    if steps_bool[3]:
        try:
            # Setting time stamp calss for step 4
            time4 = TimeStamp()       
            time4.set_start()

            print(report_spacer)

            # Define paths
            output_clumpiness = path_clumpiness
            json_path = path_json
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


            # Setting up report & Getting end time 
            time4.set_end()
            time4_start, time4_end, time4_delta = time4.get_times()
            log_to_report(file_path=path_report,
                          message=f"""{report_spacer} 
                                      \n> Step 4 finished successufly.
                                      \n> number of neurons files processed: {len(jsons_2process)}. 
                                      \n> Start time: {time4_start} 
                                      \n> End time: {time4_end} 
                                      \n Elpased time: {time4_delta}.\n""",
                          add_timestamp=True)

        # Step 4 failure
        except Exception as err:
            error_message = traceback.format_exc()
            log_to_report(file_path=path_report,
                          message=f"\n{report_spacer} \n>  step 4 Failed \n> Error messege:\n> {error_message}\n",
                          add_timestamp=True)

            raise Exception("> Step 4 Failed.")


    ############################################################
    ### Step 5 - Unify clumpiness results into a single file ###
    ############################################################
    if steps_bool[4]:
        try:
            # Setting time stamp calss for step 5
            time5 = TimeStamp()       
            time5.set_start()

            print(report_spacer)
            input_dir = path_clumpiness
            output_file = os.path.join("output", run_name, "unified_clumpiness.parquet")
            print(f"> Joining clumpiness data to unified file at {output_file}. (5/{n_steps})")

            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            compile_unified_dataset(input_dir, output_file)

            # Setting up report & Getting end time 
            time5.set_end()
            time5_start, time5_end, time5_delta = time5.get_times()
            log_to_report(file_path=path_report,
                          message=f"""{report_spacer} 
                                      \n> Step 5 finished successufly.
                                      \n> Start time: {time5_start} 
                                      \n> End time: {time5_end} 
                                      \n Elpased time: {time5_delta}.\n""",
                          add_timestamp=True)

        # Step 5 failure
        except Exception as err:
            error_message = traceback.format_exc()
            log_to_report(file_path=path_report,
                          message=f"\n{report_spacer} \n>  step 5 Failed \n> Error messege:\n> {error_message}\n",
                          add_timestamp=True)

            raise Exception("> Step 5 Failed.")


    ##################################################################################
    ### Step 6 - Assigning the clumpiness score for each SWC file (internal nodes) ###
    ##################################################################################
    if steps_bool[5]:
        try:
            # Setting time stamp calss for step 5
            time6 = TimeStamp()       
            time6.set_start()

            print(report_spacer)
            print(f"> Assigning clumpiness results into the simplified SWC files. (6/{n_steps})")

            # Define paths (from config)
            path_2process, parquet_path = [os.path.join("output", run_name, "simplified_swc"),
                                           os.path.join("output", run_name, "output_clumpiness", "unified_clumpiness.parquet")]


            save_path = os.path.join("output", run_name, "output_results", Path(path_2process).name)

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

            # Setting up report & Getting end time 
            time6.set_end()
            time6_start, time6_end, time6_delta = time6.get_times()
            log_to_report(file_path=path_report,
                          message=f"""{report_spacer} 
                                      \n> Step 6 finished successufly.
                                      \n> Files processed: {len(relv_files)}
                                      \n> Start time: {time6_start} 
                                      \n> End time: {time6_end} 
                                      \n Elpased time: {time6_delta}.\n""",
                          add_timestamp=True)
            

        # Step 6 failure
        except Exception as err:
            print("> Step 6 failed.")
            error_message = traceback.format_exc()
            log_to_report(file_path=path_report,
                          message=f"\n{report_spacer} \n>  step 6 Failed \n> Error messege:\n> {error_message}\n",
                          add_timestamp=True)

            raise Exception("> Step 6 Failed.")


    print(report_spacer)
    log_to_report(file_path=path_report,
                message=f"""{report_spacer} 
                            \n> Pipeline finished.
                            \n> Start time: {time1_start} 
                            \n> End time: {time6_end}""",
                add_timestamp=True)