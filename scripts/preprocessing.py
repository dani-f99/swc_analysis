#################################
from scripts.helpers import mkdir, get_files
import pandas as pd
import os


###############################
###############################
def get_neurons_info(path : str, 
                    file_name : str = "connectors.pkl") -> pd.DataFrame:
    """
    Custom function that itirate over all of the sub-folders and concatenating the required labels file.
    path : str -> The path of the sub-folders which contains the relevent labels information.
    file_name : str -> The name of the file which contains the labels information.
    """
    # Mapping the folders
    folders = os.listdir(path)
    paths = []

    # Getting list of folders with the connector file
    for i in folders:
        i_path = os.path.join(path, i, file_name)
        if os.path.exists(i_path):
            paths.append(i_path)

    # Creating Unified dataframe for all of the connector files
    df_mega = pd.concat([pd.read_pickle(i) for i in paths])

    return df_mega


#############################################
#############################################
def unify_labels(path : str,
                 output_folder : str = "_processed_data"):
    """
    Custom function that access the neurons nodes labels and create a single parquet file for easy access.
    path : str -> location of the metadata folder.
    """
    # Creating output folder
    mkdir(output_folder)

    try:
        labels_df = get_neurons_info(path)
        labels_df.to_parquet(os.path.join(output_folder, "neurons_nodes.labels.parquet"), 
                             engine="pyarrow", 
                             compression="zstd")
    except:
        print("> Failed to load labels file, cheeck `path` argument file path.")


####################################################################
####################################################################
def simplify_swc_topology(swc_input : pd.DataFrame,
                          swc_name : str,
                          save_csv : bool = True,
                          output_path : str = "simplified_swc") -> pd.DataFrame:
    """
    Custom function that covnerts swc neuron file to simplified format without excessive internal nodes.
    swc_input : pd.DataFrame / string file path -> input data
    swc_name : str -> file name, will be used as tamplte for the saved simplified tree (if needed).
    save_csv : bool -> if True, will save the simplified tree as pd.DataFrame, if False will return the tree as csv format in the output_path.
    outout_path : str -> if return_df is False, will return the output simplified swc as csv file in the output_path folder.
    """

    # Choosing import method (string for path / pd.DataFrame)
    if isinstance(swc_input, pd.DataFrame):
        df = swc_input

    elif isinstance(swc_input, str):
        try:
            df = pd.read_csv(swc_input)
        except:
            print(f"> Invalid swc input path ({swc_input}), please confirm that the path is correct.")
    
    
    # Create a dictionary for fast lookup of parent-child relationships
    parents = dict(zip(df['node_id'], df['parent']))
    
    # 1. Count the number of children for each node
    children_counts = {node: 0 for node in parents.keys()}
    for node, parent_id in parents.items():
        if parent_id in children_counts:
            children_counts[parent_id] += 1
            
    # 2. Identify the core nodes we need to keep
    # Keep the node if it is the root (-1), a leaf (0 children), or a branch (>1 children)
    nodes_to_keep = set()
    for node, parent_id in parents.items():
        if parent_id == -1 or children_counts[node] != 1:
            nodes_to_keep.add(node)
            
    # 3. Reroute the parent IDs for the kept nodes to bypass the deleted middle nodes
    new_parents = {}
    for node in nodes_to_keep:
        current_parent = parents[node]
        
        # Traverse up the original tree until we hit a node that was kept
        while current_parent != -1 and current_parent not in nodes_to_keep:
            current_parent = parents.get(current_parent, -1)
            
        new_parents[node] = current_parent
        
    # 4. Filter the dataframe and apply the updated parent connections
    df_out = df[df['node_id'].isin(nodes_to_keep)].copy()
    df_out['parent'] = df_out['node_id'].map(new_parents)
    
    # Clean up the index and sorting so the file output is neat
    df_out = df_out.sort_values('node_id').reset_index(drop=True)
    
    # Saving the file if needed (defualt it True)
    if save_csv:
        mkdir(output_path)
        df_out.to_csv(os.path.join(f"{swc_name}.csv"))

    return df_out


###########################
###########################
def wsc2json(auto_phrase : bool = True,
             neuron_id: str = None, 
             ftr_file : pd.DataFrame = None,
             input_folder: str = "input", 
             output_folder: str = "output",
             print_msg : bool = False) -> None:
    """
    Converts SWC and FTR files into a JSON structure suitable for find-clumpiness.
    auto_phrase : bool -> automaticly phrase over the input folder and look for the files.
    swc_file : str -> File name of the swc neuron tree file.
    ftr_file : str -> pd.DataFrame of the labels.
    input_folder : str -> Input file path (contains the swc and ftr files).
    output_folder : str -> Output file path, will save the find-clumpiness suitible json file.
    print_msg : bool -> if True will print progression messeges.
    """
    
    # Will automaticcly get all of the files
    if auto_phrase:
        files = get_files(input_folder)
    
    # Manual file input
    else:
        files = [neuron_id]

    for file in files:
        if print_msg:
            print(f">>> Processing neuron {file}.")

        # Loading SWC file & defining type int rows (node_id and parent)
        swc_path = os.path.join(input_folder, f"{file}.swc")
        swc_df = pd.read_csv(swc_path, 
                            comment='#', 
                            header=None, 
                            sep=r'\s+', 
                            names=["node_id", "swc_type", "x", "y", "z", "r", "parent"])
        swc_df["node_id"] = swc_df["node_id"].astype(int)
        swc_df["parent"] = swc_df["parent"].astype(int)

        # Loading ftr (feather) file and 
        # Dropping duplicated rows -> due to interactions with multiple neurons
        if isinstance(ftr_file, pd.DataFrame):
            ftr_df = ftr_file
        else:
            ftr_path = os.path.join(input_folder, f"{file}.ftr")
            ftr_df = pd.read_feather(ftr_path)
        ftr_df = ftr_df.drop_duplicates(subset=["node_id", "type"])
        
        # Grouping by node_id and aggregate all labels into a single list
        ftr_grouped = ftr_df.groupby("node_id").agg({"type": lambda x: x.dropna().unique().tolist()}).reset_index()

        # Merging swc and ftr dataframes.
        merged_df = pd.merge(left=swc_df[["node_id", "swc_type", "parent"]], 
                            right=ftr_grouped,
                            on="node_id",
                            how="left")
        if print_msg:
            print("Data merged successfully.")
        

        # Preparing for the json tree construction
        children_map = {}
        node_labels = {}
        root = None
        
        for _, row in merged_df.iterrows():
            node = str(int(row['node_id']))
            parent_val = row['parent']
            
            # Labels are already aggregated into a list from the groupby
            # If label found -> add to the labels dicts with the node_id as key
            labels = row['type']
            if isinstance(labels, list):
                node_labels[node] = labels
            else:
                node_labels[node] = []
                
            # Handle topology and find the root
            # Assigning root nodes
            if pd.isna(parent_val) or parent_val == -1: 
                root = node
            
            # Assigning rest of nodes
            else:
                parent = str(int(parent_val))
                if parent not in children_map:
                    children_map[parent] = []
                children_map[parent].append(node)
        if print_msg:
            print("Neuron tree mapped.")
        
        # Incase there is no defined root note with parent values of -1.
        if root is None:
            raise ValueError("Could not find the root node (a node where parent is -1).")
            
        # Anti-infinite loop section, preventing from `node -> parent`, `parent -> node` loop to occure
        # List of visited nodes
        itirated = set()

        def build_node(node_id):
            # Stop execution if returning to previously visited node
            if node_id in itirated:
                raise RecursionError(f"Cycle detected in SWC file at node {node_id}. Fix the source data.")
            itirated.add(node_id)
            
            node_dict = {"nodeID": node_id,
                        "nodeLabels": node_labels.get(node_id, [])} # Returning the node label, if not found returns empty list
            
            children_list = []
            if node_id in children_map:
                for child_id in children_map[node_id]:
                    children_list.append(build_node(child_id))
                    
            return [node_dict, children_list]
        
        # Build JSON, from the parent node to the leaves.
        final_json = build_node(root)
        
        # Export
        mkdir([output_folder])
        output_name = f"{file}.json"
        output_path = os.path.join(output_folder, output_name)

        with open(output_path, 'w') as f:
            json.dump(final_json, f, separators=(',', ':')) 
        
        if print_msg:
            print(f"Data successfully formatted and saved to {output_path} \n")