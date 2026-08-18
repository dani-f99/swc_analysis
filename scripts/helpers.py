import itertools
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
