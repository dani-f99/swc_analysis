from datetime import datetime
from pathlib import Path
import itertools
import unittest
import json
import sys
import os


############################################################################################
# Reading information from json file. Used to extract the parameters from the `config.json`.
def read_json(path:str = "config.json") -> dict:
    """
    path : str -> path of the json file
    """

    with open(path) as config:
        config_f = json.load(config)

    return config_f


######################
######################
def mkdir(dirs: list,
          print_msg: bool = False):
    """
    Custom functions that create folders based on list of provided folder names.
    dirs : list -> list of folder names (str) to be created.
    print_msg: bool -> if True will print progression messeges.
    """
    for dir in dirs:
        if os.path.exists(dir):
            msg = f"{dir} path already exists."
        else:
            os.mkdir(dir)
            msg = f"{dir} folder was created."

        if print_msg:
            print(msg)


#########################################
#########################################
def get_files(dir:str = "input") -> list:
    """
    Custom function that get all of the file names in a folder, cheecks that
    every neuron has both swc and ftr files and finnaly returns list of vaiable
    neurons (with both swc and ftr files). 
    dir : str -> folder path.
    """

    # Getting the unique neuron id's in the folder
    files_list = [i.split(".")[0] for i in os.listdir(dir)]
    files_set = set(files_list)

    # Validating that every neuron has both swc and ftr files
    files_good = []
    files_bad = []
    for f in files_set:
        f_count = files_list.count(f)
        if f_count == 2:
            files_good.append(f)
        elif f_count == 1:
            files_bad.append(f)

    # If neuron missing file, print messege
    if len(files_bad) >= 1:
        print(f"Found neuron_id with missing files: {files_bad}.")

    return files_good


#####################################################
# Used to visualize process bar in joblib intirations
def draw_progress_bar(current, 
                      total, 
                      bar_length=40):
    """
    Draws a progress bar in the terminal to match the bash script's UI.

    current:
    total: 
    bar_length: int ->
    
    
    """
    if total == 0:
        return
    percent = int((current * 100) / total)
    filled = int((current * bar_length) / total)
    empty = bar_length - filled
    bar = '#' * filled + '-' * empty
    # \r overwrites the current terminal line
    sys.stdout.write(f"\r[{bar}] {percent}% ({current}/{total})")
    sys.stdout.flush()


#####################################
#####################################
def chunked_iterable(iterable, 
                     size):
    """
    Yields batches of a specified size from an iterable.
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


##################
##################
class TimeStamp():
    """
    Custom class that takes time at the start and end and return start, end and interval
    """
    def __init__(self):
        self.start, self.end, self.inter = None, None, None

    # T=0
    def set_start(self):
       self.start_obj = datetime.now()
       self.start = self.start_obj.strftime("[%H:%M:%S_%Y-%d-%m-%Y]")

    # T=END
    def set_end(self):
        self.end_obj = datetime.now()
        self.end = self.end_obj.strftime("[%H:%M:%S_%Y-%d-%m-%Y]")

    # DELTA T (T_END - T0)
    def get_times(self):
        total_seconds = int((self.end_obj - self.start_obj).total_seconds())
        hours, remainder_seconds = divmod(total_seconds, 3600)
        minutes, _ = divmod(remainder_seconds, 60)
        self.elapsed_min = (hours, minutes)
        print(f"> Start time: {self.start}, end time: {self.end} (elapsed time: {hours}h:{minutes:02d}m).")

        return self.start, self.end, self.elapsed_min

############################################################################
############################################################################
def log_to_report(file_path: str, 
                  message: str, 
                  add_timestamp: bool = False):

    """
    Custom function that generate report
    file_path: str -> Path to which the report will be saved, will contain the file name.
    msg: str -> messege to appaend to the report.
    add_timesmp -> add timestamp to the start of the report.
    """
    path = Path(file_path)
    
    # 1. Ensure the target directory exists (creates it if it doesn't)
    if path.parent.name:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    # Optional: Format with the datetime logic you just used
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
    else:
        log_entry = f"{message}\n"
        
    # 2. 'a' (append) mode automatically creates the file if missing, 
    # or appends to the end if it already exists.
    with open(path, "a", encoding="utf-8") as file:
        file.write(log_entry)


##################################################################################################################################################################
##################################################################################################################################################################
# BETA # BETA # BETA # BETA # BETA # BETA # BETA # BETA # BETA # BETA # BETA # BETA # BETA # BETA # BETA # BETA # BETA # BETA # BETA # BETA # BETA # BETA # BETA #
##################################################################################################################################################################
##################################################################################################################################################################

# A small helper to send output to both CMD and a file
class OutputTee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


############################
############################
# Running Pipeline with test
run_name = read_json()["run_name"]
def run_pipeline(test_pipeline,
                 pipeline_name:str = run_name
                 ):
    """
    test_pipeline -> the pipeline uninitited unittest pipeline we want to run
    pipeline_name : str -> pipeline name in string format.
    """
    reports_path = os.path.join("data", "reports")
    report_name = f"{pipeline_name}_unitest_report.txt"
    mkdir([reports_path])
    

    # f is the text file -> sys.stdout is the CMD consol
    with open(os.path.join(reports_path, report_name), "w", encoding="utf-8") as f:
        # sys.stdout is the CMD console
        # f is your text file
        dual_stream = OutputTee(sys.stdout, f)
        
        runner = unittest.TextTestRunner(
            stream=dual_stream, 
            verbosity=2, 
            descriptions=True
        )

        # Initialize the runner
        runner = unittest.TextTestRunner(
                stream=dual_stream, 
                verbosity=2, 
                descriptions=True
                )
        
        # unitest  loader object
        loader = unittest.TestLoader()

        # Load tests from the specific class
        suite = loader.loadTestsFromTestCase(test_pipeline) 
        
        # Run with high verbosity for detail
        result = runner.run(suite)
        
        # Custom detailed summary
        print("\n--- PIPELINE EXECUTION SUMMARY ---")
        if result.wasSuccessful():
            print("Final Status: SUCCESS V")
        else:
            print(f"Final Status: FAILED X ({len(result.failures) + len(result.errors)} issues found)")



