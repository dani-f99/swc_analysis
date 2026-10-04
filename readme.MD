--------------------------------------------------------------------------------
                              COMPUTATIONAL BIOLOGY LAB
                                  Haifa University
                                   Daniel Fridman
                                        2026
--------------------------------------------------------------------------------

# PROJECT: SWC Neuron Morphological Analysis

## 1. OVERVIEW
This program is a Python pipeline that computes clumpiness scores of synapse labels (`pre` / `post`) on neuron SWC trees. Each neuron is simplified, labelled, converted to a JSON tree and split into one sub-tree per internal node. The clumpiness of the main tree and of every sub-tree is calculated with the `find-clumpiness` program. The scores are then joined into one unified table, assigned back to the root and the internal nodes of each neuron's SWC file, and drawn as an interactive topology plot per neuron.

The pipeline runs neuron by neuron (in parallel with joblib), checks the output of every step and writes an html report of the run.

## 2. PREREQUISITES

### Required Python Modules:
- pandas
- NumPy
- polars
- pyarrow
- joblib
- tqdm
- plotly
- nbformat (optional - shows the plots inside the notebook; without it they open in the browser)

### Required Program:
- [find-clumpiness](https://github.com/GregorySchwartz/find-clumpiness) - has to be on the `PATH` (try `find-clumpiness --help`). It is needed for step 6 only; all other steps run without it.


## 3. CONFIGURATION (config.json)
Paths are written as comma separated folder names (e.g. `"data,input_swc,sk_lod1_783_healed"`) and are relative to the folder of `config.json`.

| Key | Meaning |
|---|---|
| `run_name` | Name of the run. All results are written to `output/<run_name>/`. |
| `swc_path` | Folder of the raw SWC files (`<neuron_id>.swc`), e.g. `data/input_swc/sk_lod1_783_healed`. |
| `labels_path` | Folder of the node (synapse) labels, e.g. `data/input_labels/processed_swc_data_princeton`. |
| `labels_ftr_path` | Neuron labels `.ftr` file, e.g. `data/input_labels/neuron_data_full_article_princeton.ftr`. |
| `overwrite_info` | `"False"`: outputs that already exist are kept and the step is skipped. `"True"`: outputs are written again. |
| `data_limit` | Number of neurons to process, `"None"` = all (for testing purposes). |
| `n_jobs` | Number of neurons processed in parallel. |
| `n_threads` | Not used by the current pipeline. |
| `results_swc_path`, `results_labels_path` | Not used by the current pipeline. |


## 4. PIPELINE STEPS
The steps are methods of the `ProcessSWC` class (`scripts/process_swc.py`). Steps 2-6 run per neuron; steps 1 and 7-9 run once for the whole run.

| Step | What it does | Output (under `output/<run_name>/`) |
|---|---|---|
| 0 | Reads the config and creates the output folders. | all folders |
| 1 | Builds the labels file of all neurons (neuron class + pre / post synapse per node). | `1-output_labels/swc_labels.parquet` |
| 2 | Simplifies the SWC tree. Labelled nodes are always kept. | `2-swc_simplified/<neuron_id>.csv` |
| 3 | Adds the labels to the simplified tree (`type` column). | `3-swc_simplified_labeled/<neuron_id>.csv` |
| 4 | Converts the labelled tree to the JSON format of find-clumpiness. | `4-json/<neuron_id>.json` |
| 5 | Writes one sub-tree JSON per internal node (root excluded). | `5_json_divided/<neuron_id>/<neuron_id>_<node_id>.json` |
| 6 | Runs find-clumpiness on the main tree and on every sub-tree. | `6-clumpiness_scores/<neuron_id>/<neuron_id>.csv` (main tree) and `<neuron_id>_<node_id>.csv` |
| 7 | Joins all scores into one unified table. Always rebuilt. | `7-swc_clumpiness_result/unified_clumpiness.csv` |
| 8 | Assigns the scores to the root and the internal nodes of each neuron's SWC file, for the simplified and for the raw tree. | `8-swc_clumpiness/<simplified or raw>/<neuron_id>.csv` |
| 9 | Saves the interactive topology plot of each neuron. | `9-swc_clumpiness_plots/<simplified or raw>/<neuron_id>.html` |


## 5. USAGE GUIDE
1. Configure the `config.json` file as needed (see section 3).
2. Install [find-clumpiness](https://github.com/GregorySchwartz/find-clumpiness) and make sure it exists on your system.
3. Run the pipeline, preferably on a unix server capable of handling parallel task processing:

```
python main.py                   # first 10 neurons, steps 1-9
python main.py  -n 0             # all neurons
python main.py  -n 50 --selection random --n-jobs 4 --overwrite true
python main.py  --last-step 5    # stop before the clumpiness calculation
python maint.py --neuron-ids 720575940596125868 720575940597856265
python main.py  --help           # all options
```

   Every step prints a message when it starts and shows a progress bar, e.g. `> Step 6 - clumpiness scores (10 neurons)`.
   The script can be started from any folder. Its exit code is `0` when every neuron and every step passed, `1` when something failed and `2` when nothing was run (e.g. find-clumpiness is missing).

4. Open the report of the run: `output/<run_name>/0-run_log/test_report_<date>_<time>.html` (see section 6).

The same run can be done step by step in `dev_notebook.ipynb`: the pipeline class cell, followed by the test cell (settings at its top: number of neurons, selection, overwrite).

### Plotting a single neuron
```python
from scripts.visualize import plot_neuron_topology

plot_neuron_topology("output/<run_name>/8-swc_clumpiness/simplified/<neuron_id>.csv")
```
Zoom and pan with the mouse, hover a node for its id, label and scores, and switch the score column in the menu of the figure. `save_html="plot.html", show=False` writes the plot to a file instead of showing it.


## 6. OUTPUT

### Unified clumpiness table (step 7)
One row per tree - the main tree and every sub-tree of every neuron.

| Column | Meaning |
|---|---|
| `neuron_id` | Neuron id. It has 18 digits - Excel rounds it, `pandas.read_csv` reads it correctly. |
| `node_id` | Root node of the sub-tree. `0` = the main tree (the whole neuron). |
| `post_post`, `post_pre`, `pre_post`, `pre_pre` | Clumpiness of each label combination. Empty when the combination was not found in the tree. |

### Scored SWC files (step 8)
The SWC nodes (`node_id`, `swc_type`, `x`, `y`, `z`, `r`, `parent`; the simplified file also has the `type` label column) with the clumpiness columns added. The root and the internal nodes of the simplified tree carry their scores, every other node is empty. In these files the root carries its real node id, not `0`.

### Topology plots (step 9)
One html file per neuron and tree. The fill colour of a node is its clumpiness score, the shape and colour show the label: triangle = `pre`, square = `post`, diamond = both. The files load the plotly library from the internet when opened.

### Run report (`0-run_log`)
`test_report_<date>_<time>.html` and a `.csv` with the same results. The report shows:
- the result of every tested neuron and of the validation checks of steps 2-6,
- the run-level steps 7-9 with their statistics and links to their outputs,
- the runtime of every step per neuron,
- a link to the plots of every neuron,
- the errors and tracebacks of failed neurons and steps.


## 7. DIRECTORY STRUCTURE

```
config.json            configuration of the run
run_test.py            runs the pipeline from the command line
dev_notebook.ipynb     the pipeline class and a test run, step by step
scripts/
    process_swc.py     ProcessSWC class - one method per step
    preprocessing.py   functions of steps 1-4 (labels, simplification, JSON)
    processing.py      functions of steps 5-8 (sub-trees, clumpiness, unified table, score assignment)
    visualize.py       topology plot (step 9)
    report.py          html report of a run
    helpers.py         small helper functions
    pipeline.py        single-function version of steps 2-5 (not used by run_test.py)
data/
    input_swc/         raw SWC files
    input_labels/      label files
output/<run_name>/     results, one folder per step (see section 4)
_archive/              earlier versions of the scripts and notebooks
```

## 8. RESOURCES
- find clumpiness - [https://github.com/GregorySchwartz/find-clumpiness](https://github.com/GregorySchwartz/find-clumpiness)
- fly brainscan (swc files) - [https://codex.flywire.ai/api/download?dataset=fafb](https://codex.flywire.ai/api/download?dataset=fafb)
