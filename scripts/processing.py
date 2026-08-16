from pathlib import Path
import subprocess
import json
import sys
import csv
import os


####################################################
import json
import os
from pathlib import Path

def generate_internal_subtrees(input_json_path: str, 
                               neuron_number: str,
                               overwrite: bool = False, 
                               output_dir: str = os.path.join("data", "output_json")) -> None:
    """
    Parses a nested JSON tree of a neuron and exports all internal sub-trees 
    (excluding the main root and leaves) into individual JSON files.
    """
    input_path = Path(input_json_path)
    out_dir = Path(output_dir)
    
    # Create the output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the original JSON tree
    with open(input_path, 'r') as f:
        tree_data = json.load(f)

    def traverse(current_node, is_main_root=False):
        # Validate node structure: [{"nodeID": "...", "nodeLabels": [...]}, [children]]
        if not current_node or len(current_node) != 2:
            return

        node_info, children = current_node
        node_id = node_info.get("nodeID")

        # A node is internal if the children array is not empty
        if children:
            # Export the sub-tree if it is NOT the absolute root
            if not is_main_root:
                output_filename = f"{neuron_number}_{node_id}.json"
                output_filepath = out_dir / output_filename

                # Only write if overwrite is True OR the file doesn't exist
                # Using pathlib's .exists() is cleaner than os.path.exists()
                if overwrite or not output_filepath.exists():
                    with open(output_filepath, 'w') as out_f:
                        json.dump(current_node, out_f)

            # ALWAYS recursively process all children, even if we skipped writing the parent
            for child in children:
                traverse(child, is_main_root=False)

    # Initiate traversal, explicitly flagging the first node as the main root
    traverse(tree_data, is_main_root=True)



####################################################
# Clumpiness calculation
#!/usr/bin/env python3
def draw_progress_bar(current, total, bar_length=40):
    """Draws a progress bar in the terminal to match the bash script's UI."""
    if total == 0:
        return
    percent = int((current * 100) / total)
    filled = int((current * bar_length) / total)
    empty = bar_length - filled
    bar = '#' * filled + '-' * empty
    # \r overwrites the current terminal line
    sys.stdout.write(f"\r[{bar}] {percent}% ({current}/{total})")
    sys.stdout.flush()

def call_clumpiness(INPUT_DIR = os.path.join("data", "output_json"),
                    OUTPUT_DIR = os.path.join("data", "output_clumpiness"),
                    OUTPUT_FILE = os.path.join("data", "clumpiness.csv")):
    # 1. Ensure the output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 2. RESUME CAPABILITY: Load already processed IDs
    processed_ids = set()
    if OUTPUT_FILE.exists():
        print("Found existing output file. Indexing already processed neuron IDs...")
        with open(OUTPUT_FILE, 'r', newline='') as f:
            reader = csv.reader(f)
            # Skip header
            next(reader, None) 
            for row in reader:
                if row:
                    # Strip quotes from the first column to match bash tr -d '"'
                    clean_id = row[0].strip('"') 
                    processed_ids.add(clean_id)
        print(f"Found {len(processed_ids)} already processed neurons.")
    else:
        # Create fresh file with headers
        with open(OUTPUT_FILE, 'w', newline='') as f:
            f.write("neuron_id,groups,clumpiness\n")

    # 3. PREPARE FILE LIST
    print("Scanning input directory for JSON files...")
    input_path = Path(INPUT_DIR)
    
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Directory '{INPUT_DIR}' does not exist.")
        sys.exit(1)

    # Using pathlib.glob handles large directory lists gracefully in memory
    json_files = list(input_path.glob("*.json"))
    total_files = len(json_files)

    if total_files == 0:
        print(f"No JSON files found in {INPUT_DIR}.")
        sys.exit(0)

    # Filter out already processed files (neuron_id is the filename without extension)
    files_to_process = [f for f in json_files if f.stem not in processed_ids]
    total_to_process = len(files_to_process)

    if total_to_process <= 0:
        print(f"All {total_files} files have already been processed. Nothing to do!")
        sys.exit(0)

    print(f"Total files: {total_files} | Remaining to process: {total_to_process}")

    # 4 & 5. EXECUTION LOOP
    current_count = 0
    
    # Open the CSV in append mode so we can write lines as they finish
    with open(OUTPUT_FILE, 'a', newline='') as out_f:
        for filepath in files_to_process:
            neuron_id = filepath.stem
            
            try:
                # Run the command and capture stdout
                result = subprocess.run(
                    ["find-clumpiness", "-e", "AllExclusive", "-i", str(filepath), "-f", "JSON"],
                    capture_output=True,
                    text=True,
                    check=True # Raises exception if command fails (non-zero exit status)
                )
                
                # Parse output (Replaces the awk command)
                lines = result.stdout.strip().splitlines()
                
                # lines[0] is the header, skip it
                for line in lines[1:]:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        col1 = parts[0].strip()
                        col2 = parts[1].strip()
                        col3 = parts[2].strip()
                        
                        # Format string to match: "neuron_id",col1-col2,col3
                        formatted_result = f'"{neuron_id}",{col1}-{col2},{col3}\n'
                        out_f.write(formatted_result)
                        
            except subprocess.CalledProcessError as e:
                # Triggers if find-clumpiness returns an error code
                print(f"\nError processing {neuron_id}: {e.stderr}")
            except FileNotFoundError:
                # Triggers if the system literally cannot find the find-clumpiness executable
                print("\nError: 'find-clumpiness' command not found. Ensure it is installed and in your system PATH.")
                sys.exit(1)

            # Update progress UI
            current_count += 1
            draw_progress_bar(current_count, total_to_process)



####################################################
def process_single_clumpiness(filepath: str, 
                              output_dir: str,
                              overwrite: bool = False):
    """
    Takes a JSON file, runs find-clumpiness, and saves the exact output 
    to a CSV file with the same name as the input, overwriting if it exists.
    """
    filepath = Path(filepath)
    output_csv = Path(output_dir) / f"{filepath.stem}.csv"

    if (overwrite is False) and (os.path.exists(output_csv)):
        return

    else:
        try:
            # Opening in 'w' mode automatically overwrites the file if it already exists
            with open(output_csv, 'w') as f_out:
                subprocess.run(
                    ["find-clumpiness", "-e", "AllExclusive", "-i", str(filepath), "-f", "JSON"],
                    stdout=f_out,             # Dumps output straight to the file
                    stderr=subprocess.PIPE,   # Catches errors so they don't print to terminal
                    text=True,
                    check=True
                )
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError, Exception):
            # If it fails, delete the empty/partial CSV so it doesn't leave corrupted data
            if output_csv.exists():
                output_csv.unlink()
            return False