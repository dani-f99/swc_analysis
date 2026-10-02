#################################
# Fixed copy of `scripts/preprocessing.py` - see `BUGFIX_REPORT_claude.md` for the full change log.
from tqdm import tqdm
import pandas as pd
import json
import ast
import sys
import os



############################################################################
############################################################################
def get_neurons_info(parquet_path : str ,
                     labels_path: str,
                     swc_labels_file_path : str,
                     swc_labels_filter : tuple = (True, ("super_class",["central", "optic", "visual_centrifugal", "visual_projection"])),
                     nodes_labels_folder : str = "processed_swc_data_princeton",
                     nodes_labels_name : str = "connectors.pkl",
                     overwrite_parquet : bool = True
                    ) -> dict | None:
    """
    parquet_path : str -> path to the folder which contains the parquet file (saved to).
    labels_path: str -> path to the labels ROOT folder (the one containing `swc_labels_file` and `nodes_labels_folder`).
    swc_labels_filter : tuple -> if [0] is True filter out unwanted swc file on the base of their super_class label [1][0] and their super type labels rquired [1][1] of the touple.
    swc_labels_file : str -> name of the neurons swc labels file.
    nodes_labels_folder : str -> name of the folders containing the nodes labels data.
    nodes_labels_name : str -> name of the file which contains the nodes labels (pre / post synaptic).
    overwrite_parquet : bool -> If true will overwrite the labels+metadata parquet file (defualt is True).
    Returns a dict with run statistics, or None if the existing parquet was preserved.
    """

    parquet_path = os.path.join(parquet_path, "swc_labels.parquet")
    parquet_exists = os.path.exists(parquet_path)

    # [FIX 3a] The old branching called `os.remove` on a missing file when overwrite=True (first run crash).
    if parquet_exists and (overwrite_parquet is False):
        print("> Function execution halted, old `swc_labels.parquet` file preserved.")
        return None

    if parquet_exists:
        os.remove(parquet_path)
        print("> Old `swc_labels.parquet` deleted, creating a new file.")
    else:
        print("> `swc_labels.parquet` havne't been found, creating a new file.")

    # 1. Generating a required neurons dataframe with their super-class type.
    # [FIX 3d] The old `if swc_labels_filter:` tested the tuple itself (always True), so the [0] flag was ignored.
    swc_labels = None
    if swc_labels_filter and swc_labels_filter[0]:
        try:
            filter_on = swc_labels_filter[1][0]
            filter_by = swc_labels_filter[1][1]

            swc_labels = pd.read_feather(swc_labels_file_path)
            swc_labels = swc_labels.loc[swc_labels[filter_on].isin(filter_by), ["neuron", filter_on]].drop_duplicates()
            swc_labels.neuron = swc_labels.neuron.astype("str")

        except Exception as err:
            raise Exception(f"> Error occured while tying to generate relevent neurons labels from `{swc_labels_file_path}`. "
                            f"Check `labels_path` and the `swc_labels_filter` argument input. ({err})") from err


    # 2. Loading the nodes labels files, cheecking for required neurons and saving the data to parquet (concat) with each itiration for storage efficincy.
    nodes_labels_path = os.path.join(labels_path)
    folders = sorted(os.listdir(nodes_labels_path))

    folders_skipped = []
    rows_written = 0
    parquet_created = False

    for i in tqdm(folders, desc="Procssing metadata files", unit="files"):
        i_path = os.path.join(nodes_labels_path, i, nodes_labels_name)

        # [FIX 3b] The write used to sit outside this check, so a folder without a connectors file
        # re-wrote the previous folder's `temp_labels` (duplicate rows / NameError on the first folder).
        if os.path.exists(i_path) is False:
            folders_skipped.append(i)
            continue

        temp_labels = pd.read_pickle(i_path)
        temp_labels.neuron = temp_labels.neuron.astype("str")

        # [FIX 3c] No more `except: pass` - a failed merge used to append the UNFILTERED labels.
        if swc_labels is not None:
            temp_labels = pd.merge(left=temp_labels,
                                   right=swc_labels,
                                   left_on="neuron",
                                   right_on="neuron",
                                   how="inner")

        if temp_labels.empty:
            continue

        temp_labels.to_parquet(parquet_path,
                               engine="fastparquet",
                               compression="zstd",
                               append=parquet_created)
        parquet_created = True
        rows_written += len(temp_labels)

    if folders_skipped:
        print(f"> {len(folders_skipped)} folders without `{nodes_labels_name}` skipped: {folders_skipped}")

    if parquet_created is False:
        raise Exception(f"> No labels were written - check `{nodes_labels_path}` and the neurons filter.")

    return {"folders": len(folders),
            "folders_skipped": folders_skipped,
            "rows_written": rows_written}



####################################################################
####################################################################
def simplify_swc_topology(swc_input : pd.DataFrame,
                          swc_name : str,
                          output_path : str,
                          save_csv : bool = True,
                          keep_nodes = None,
                          ) -> pd.DataFrame:
    """
    Custom function that covnerts swc neuron file to simplified format without excessive internal nodes.
    Kept nodes: the root, leaves (0 children), branch points (>1 children) and every node in `keep_nodes`.
    Other nodes with exactly one child are removed.
    swc_input : pd.DataFrame / string file path -> input data
    swc_name : str -> file name, will be used as tamplte for the saved simplified tree (if needed).
    save_csv : bool -> if True, will save the simplified tree as csv in the output_path folder.
    outout_path : str -> folder to which the csv file is saved (if save_csv is True).
    keep_nodes : iterable | None -> node ids that are never removed, e.g. the synapse-labelled nodes. Without it,
                                    labels on removed cable nodes are lost when labels are joined after this step (report issue #2).
    """

    # Choosing import method (string for path / pd.DataFrame)
    if isinstance(swc_input, pd.DataFrame):
        df = swc_input

    elif isinstance(swc_input, str):
        # [FIX 6e] The old code only printed on failure and then crashed on an undefined `df`.
        try:
            df = pd.read_csv(swc_input)
        except Exception as err:
            raise Exception(f"> Invalid swc input path ({swc_input}), please confirm that the path is correct.") from err

    else:
        raise TypeError(f"> `swc_input` must be a pd.DataFrame or a path string, got {type(swc_input)}.")


    # Create a dictionary for fast lookup of parent-child relationships
    parents = dict(zip(df['node_id'], df['parent']))

    # 1. Count the number of children for each node
    children_counts = {node: 0 for node in parents.keys()}
    for node, parent_id in parents.items():
        if parent_id in children_counts:
            children_counts[parent_id] += 1

    # 2. Identify the core nodes we need to keep
    # Keep the node if it is the root (-1), a leaf (0 children), a branch (>1 children) or a forced node (e.g. labelled)
    # [FIX 2] `keep_nodes` - synapse-labelled cable nodes used to be removed here, losing ~85% of the labels.
    always_keep = {int(n) for n in keep_nodes} if keep_nodes is not None else set()
    nodes_to_keep = set()
    for node, parent_id in parents.items():
        if parent_id == -1 or children_counts[node] != 1 or node in always_keep:
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
        df_out.to_csv(os.path.join(output_path, f"{swc_name}.csv"))

    return df_out



####################################################################
####################################################################
def attach_node_labels(swc_df: pd.DataFrame,
                       labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    [FIX 1] Replacement for the merge + `groupby(...).unique()` assignment, which aligned the labels on the
    DataFrame's row index instead of on `node_id` and put labels on the wrong nodes (or erased them).
    swc_df : pd.DataFrame -> (simplified) swc table with a `node_id` column.
    labels_df : pd.DataFrame -> labels rows of the same neuron with `node_id` and `type` columns.
    Returns a copy of `swc_df` with a `type` column: one label, or comma-joined sorted labels if a node has several.
    """
    labels_per_node = (labels_df[["node_id", "type"]]
                       .dropna()
                       .drop_duplicates()
                       .groupby("node_id")["type"]
                       .agg(lambda labels: ",".join(sorted(labels.astype(str)))))

    swc_labeled = swc_df.copy()
    swc_labeled["type"] = swc_labeled["node_id"].map(labels_per_node)   # aligned by node_id, one row per node

    return swc_labeled



####################################################################
####################################################################
def swc2json(swc_dataset,
             neuron_id: str,
             save_json: bool = False,
             save_path: str = None,
             print_msg: bool = False,
             overwrite: bool = False) -> dict | None:
    """
    Converts SWC and FTR files into a JSON structure suitable for find-clumpiness.
    swc_dataset : pd.DataFrame / str -> SWC information with labels attached (synpase annotations).
    neuron_id: str -> Neuron name (id).
    save_json : bool -> If True will save the JSON file into `save_path`.
    save_path: str -> Folder to which save the processed json file.
    print_msg : bool = False -> if True will print messege about the processed neuron.
    overwrite : bool -> if False and the json already exists, the function returns None without work.
    Returns a dict with tree statistics (n_nodes, n_labeled, depth), or None if skipped.
    """
    output_file = os.path.join(save_path, f"{neuron_id}.json")

    if (overwrite is False) and (os.path.exists(output_file)):
        return None

    # Load csv or import pd.DataFrame object
    if isinstance(swc_dataset, pd.DataFrame):
        swc_df = swc_dataset.copy()
    elif isinstance(swc_dataset, str):
        try:
            swc_df = pd.read_csv(swc_dataset, index_col=0)
        except Exception as err:
            raise Exception("> Invalid string input for `swc_dataset` argument") from err
    else:
        raise TypeError(f"> `swc_dataset` must be a pd.DataFrame or a path string, got {type(swc_dataset)}.")

    swc_df["node_id"] = swc_df["node_id"].astype(int)
    swc_df["parent"] = swc_df["parent"].astype(int)

    # [FIX 5c] Duplicate node rows used to append the same child twice (duplicated sub-trees,
    # or a false "cycle" error). Fail loudly instead of relying on upstream `drop_duplicates`.
    duplicated_nodes = swc_df.loc[swc_df["node_id"].duplicated(), "node_id"].unique()
    if len(duplicated_nodes) > 0:
        raise ValueError(f"> {len(duplicated_nodes)} duplicated node_id rows in neuron {neuron_id} (e.g. {list(duplicated_nodes[:5])}).")

    # Preparing for the json tree construction
    children_map = {}
    node_labels = {}
    roots = []

    for _, row in swc_df.iterrows():
        node = str(int(row['node_id']))
        parent_val = row['parent']

        # Safely parse the labels column whether it's a list, a stringified list, or NaN
        raw_labels = row['type']
        parsed_labels = []

        if isinstance(raw_labels, list):
            # If passed directly as a DataFrame where lists are preserved
            parsed_labels = [str(x).strip().lower() for x in raw_labels]
        elif pd.notna(raw_labels):
            if isinstance(raw_labels, str):
                raw_labels = raw_labels.strip()
                if raw_labels.startswith('[') and raw_labels.endswith(']'):
                    try:
                        # Safely evaluate the stringified list "['label1', 'label2']"
                        evaluated = ast.literal_eval(raw_labels)
                        if isinstance(evaluated, list):
                            parsed_labels = [str(x).strip().lower() for x in evaluated]
                        else:
                            parsed_labels = [str(evaluated).strip().lower()]
                    except (ValueError, SyntaxError):
                        parsed_labels = [raw_labels.lower()]
                else:
                    # Handle basic comma-separated strings if they occur
                    parsed_labels = [x.strip().lower() for x in raw_labels.split(',')]

        node_labels[node] = parsed_labels

        # Handle topology and find the root
        if pd.isna(parent_val) or parent_val == -1:
            roots.append(node)

        # Assigning rest of nodes
        else:
            parent = str(int(parent_val))
            if parent not in children_map:
                children_map[parent] = []
            children_map[parent].append(node)

    if print_msg:
        print("Neuron tree mapped.")

    # Incase there is no defined root note with parent values of -1.
    if len(roots) == 0:
        raise ValueError("Could not find the root node (a node where parent is -1).")

    # [FIX 5a] The old code kept only the LAST root, silently dropping every other component.
    if len(roots) > 1:
        raise ValueError(f"> Neuron {neuron_id} has {len(roots)} root nodes (parent == -1): {roots[:10]}. The tree is disconnected.")

    root = roots[0]

    # [FIX 5b] Deep trees exceeded Python's default recursion limit (1000) in both `build_node` and `json.dump`.
    # Compute the depth iteratively and raise the limit only as much as needed.
    depth = 0
    stack = [(root, 1)]
    seen_depth = set()
    while stack:
        node_id, node_depth = stack.pop()
        if node_id in seen_depth:
            continue
        seen_depth.add(node_id)
        depth = max(depth, node_depth)
        for child_id in children_map.get(node_id, []):
            stack.append((child_id, node_depth + 1))

    required_limit = depth * 3 + 200
    if sys.getrecursionlimit() < required_limit:
        sys.setrecursionlimit(required_limit)

    # Anti-infinite loop section, preventing from `node -> parent`, `parent -> node` loop to occure
    itirated = set()

    def build_node(node_id):
        # Stop execution if returning to previously visited node
        if node_id in itirated:
            raise RecursionError(f"Cycle detected in SWC file at node {node_id}. Fix the source data.")
        itirated.add(node_id)

        node_dict = {
            "nodeID": node_id,
            "nodeLabels": node_labels.get(node_id, [])
        }

        children_list = []
        if node_id in children_map:
            for child_id in children_map[node_id]:
                children_list.append(build_node(child_id))

        return [node_dict, children_list]

    # Build JSON, from the parent node to the leaves.
    final_json = build_node(root)

    # [FIX 5d] Every SWC row must be reachable from the root, otherwise nodes were silently left out of the JSON.
    if len(itirated) != len(node_labels):
        unreachable = [n for n in node_labels if n not in itirated]
        raise ValueError(f"> {len(unreachable)} nodes of neuron {neuron_id} are unreachable from root {root} (e.g. {unreachable[:5]}).")

    # Export
    if save_json and isinstance(save_path, str):
        os.makedirs(save_path, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(final_json, f, separators=(',', ':'))

    return {"n_nodes": len(itirated),
            "n_labeled": sum(1 for v in node_labels.values() if v),
            "depth": depth}
