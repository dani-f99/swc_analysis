from scripts.helpers import mkdir
import os 

folders = ["data", "input_swc", "input_labels"]
req_folders = ["data"] + [os.path.join(folders[0], i) for i in folders[1:]]

for i in req_folders:
    mkdir(req_folders)