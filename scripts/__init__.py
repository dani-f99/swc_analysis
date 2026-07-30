from scripts.helpers import mkdir
import os 

folders = ["data", "swc_input", "input_labels", "output_json", "output_clumpiness"]
req_folders = ["data"] + [os.path.join(folders[0], i) for i in folders[1:]]

for i in req_folders:
    mkdir(req_folders)