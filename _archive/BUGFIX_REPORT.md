# SWC analysis pipeline: bug-fix report

**Date:** 2026-10-01
**Scope:** `scripts/preprocessing.py`, `scripts/processing.py`, `scripts/pipeline.py`, and the `ProcessSWC` class in `notebook_test.ipynb`
**Status:** Fixes are in new `*_claude` files. The original files are unchanged. Steps 1–5 were validated on real data. Steps 6–7 (find-clumpiness and joining the results) are not implemented in the class yet, and were not run.

---

## 1. Summary

The pipeline's clumpiness results are invalid because of one label-alignment bug in the label-joining step (issue #1). The bug erased most synapse labels and moved some of the remaining ones onto the wrong nodes, so find-clumpiness was scoring trees that were mostly unlabelled and partly mislabelled.

There is also one open design question (issue #2): tree simplification removes about 85% of the labelled nodes before labels are attached. That is not a coding error, but it strongly affects what the scores measure, and it needs a deliberate decision.

The remaining issues are crashes, silent data loss, or stale-file risks that did not corrupt the current results but would on other inputs or settings.

**Measured effect on the existing results** (`output/swc_analysis_test`, first 20 neurons):

| Quantity | Count |
|---|---|
| Labelled nodes in the raw data | 9,225 |
| Labelled nodes that survive simplification (correct expected count) | 1,424 |
| Labelled nodes in the old pipeline's output | 216 |
| ... of which carry a label on a node that has no label | 177 |

So about 3% of the expected labels were kept in their correct place.

**Action required:** every result produced by `run_pipeline.py` / `scripts/pipeline.py` must be regenerated.

---

## 2. Files

| New file | Replaces | Notes |
|---|---|---|
| `scripts/preprocessing_claude.py` | `scripts/preprocessing.py` | Fixes #3, #5, #6e; adds `attach_node_labels` (fix #1) |
| `scripts/processing_claude.py` | `scripts/processing.py` | Fixes #4, #6a–#6d |
| `scripts/pipeline_claude.py` | `scripts/pipeline.py` | Fixes #1, #7 and the overwrite parsing |
| `scripts/process_swc_claude.py` | `ProcessSWC` in `notebook_test.ipynb` | Fixes #1, #8 |

The function signatures are unchanged except for the added `overwrite` parameter on `join_swc_clumpiness`. Some functions now return statistics where they used to return `None`.

**To adopt:** change the imports from `scripts.preprocessing` to `scripts.preprocessing_claude`, and so on, or move the fixed code into the original files.

**Config layout (updated 2026-10-01):** `get_neurons_info` in `preprocessing_claude.py` was changed to take the two label inputs separately:
- `labels_path` is the connectors folder (`"data,input_labels,processed_swc_data_princeton"`);
- the new key `labels_ftr_path` is the `.ftr` neurons file (`"data,input_labels,neuron_data_full_article_princeton.ftr"`).

`ProcessSWC` in `process_swc_claude.py` reads both keys. (The original `get_neurons_info` expected the parent folder; the old `run_pipeline.py` hid this by hardcoding `data/input_labels`.)

**Using the fixed class in the notebook:** the cell after the class cell replaces `ProcessSWC` with `scripts.process_swc_claude.ProcessSWC` when `mode == "claude"`, reloading the `_claude` modules so that edits are picked up without restarting the kernel.

---

## 3. Issues

Severity levels:
- **Critical:** corrupts results.
- **High:** silent data loss, or a crash on a normal configuration.
- **Medium:** a latent crash, or a stale-data risk.
- **Low:** diagnostics and robustness.

### #1: Labels are aligned by row position, not by node (Critical)

- **Location:** `scripts/pipeline.py:70`; also step 3 of `ProcessSWC` in `notebook_test.ipynb`.
- **Code:**
  ```python
  swc_labeled["type"] = swc_labeled.groupby("node_id")["type"].unique().apply(...)
  ```
- **Root cause:** the `groupby` result is a Series indexed by `node_id`. Assigning a Series to a column matches values by the DataFrame's **index**, which after `pd.merge` is `0..n-1`. Row *i* therefore receives the label of the node whose `node_id == i`. Simplified trees have sparse node IDs (e.g. 1, 2, 11, 62...), so most rows find nothing and become NaN. Rows whose position happens to equal some node ID get that unrelated node's label.
- **Symptom:** none visible. No error is raised, and the files have the right shape.
- **Reproduction:** for a tree with `node_id = [1, 5, 9, 12]` and labels on nodes 5, 9 and 12, the merge gives the correct labels, and after this line all four rows are NaN.
- **Fix:** a new `attach_node_labels()` in `preprocessing_claude.py`:
  - group the labels per `node_id` (several labels are joined with commas, sorted);
  - attach them with `swc["node_id"].map(...)`, i.e. aligned by `node_id`;
  - no merge, so no duplicated rows and no `drop_duplicates` needed afterwards.

  It is used by `pipeline_claude.process_neuron` and `ProcessSWC.step03_label_joining`.
- **Verification:** for 5 neurons, every node's label set in the step-3 output equals the label set computed independently from the parquet. For neuron `720575940596125868`, 41 nodes are now labelled, against 5 in the old output.

### #2: Simplification removes most labelled nodes (Open decision)

- **Location:** `simplify_swc_topology`, `preprocessing.py:133`. The behaviour is unchanged in the `_claude` version, which only adds a docstring note.
- **Behaviour:** only the root, the leaves and the branch points are kept. Every node with exactly one child is removed. Labels are attached **after** simplification, so synapses located on the removed cable nodes are discarded.
- **Measured** (5 neurons; the per-neuron `labeled_nodes_lost` statistic from step 3):

  | Neuron | Labelled nodes, raw | Kept | Lost |
  |---|---|---|---|
  | 720575940596125868 | 274 | 41 | 233 |
  | 720575940597856265 | 875 | 117 | 758 |
  | 720575940597944841 | 507 | 63 | 444 |
  | 720575940598267657 | 1304 | 166 | 1138 |
  | 720575940599333574 | 619 | 115 | 504 |

  About 85% of the labelled nodes are lost.
- **Decision (2026-10-01): keep the labelled nodes.**
  - `simplify_swc_topology` takes a new `keep_nodes` argument. These nodes are never removed, even when they have exactly one child.
  - `ProcessSWC.step02` (with `keep_labeled_nodes=True`, the default) and `pipeline_claude.process_neuron` pass in the neuron's labelled node IDs.
  - Every synapse stays on its own node, and no label is moved.
- **Verified** (same 5 neurons, test cell check `ok_no_labels_lost`): all labelled nodes are kept, e.g. 274/274 for `…125868`, previously 41. No label ID was missing from the raw SWC.
- **Consequences:**

  | | Before | After |
  |---|---|---|
  | Simplified tree size | 206 nodes | 439 nodes (2–2.6× across the 5 neurons) |
  | Sub-trees per neuron | 97 | 330 (about 3.4×) |
  | Runtime of steps 2–5 | ~3.5 s/neuron | ~8.7 s/neuron |

  Step 6 runtime will grow roughly in line with the number of sub-trees.
- **Open check:** 92% of the labelled nodes are now **internal** nodes (cable nodes with one child), not leaves (3,588 internal vs 310 leaves across the 5 neurons). Before running step 6 at scale, confirm that find-clumpiness counts labels on internal nodes. If it only reads leaf labels, those 92% are still ignored. A minimal test: score a small tree with one label on an internal node, then the same tree with that label moved to a leaf, and compare the outputs.

### #3: `get_neurons_info`, labels parquet creation

| ID | Severity | Problem | Fix |
|---|---|---|---|
| 3a | High | With `overwrite_parquet=True` and no existing file, the `else` branch calls `os.remove()` on a missing file, which raises `FileNotFoundError` on every first run. `ProcessSWC.step01` hid this with its own existence check, which in turn made `overwrite` ineffective. | Remove the file only if it exists; otherwise create it. |
| 3b | High | `to_parquet` was outside the `if os.path.exists(connectors.pkl)` block. A folder without a connectors file re-appends the previous folder's data (duplicate rows); if the first folder has none, you get a `NameError`. *Did not trigger on the current data: all 93 folders have the file.* | The write moved inside the check; missing folders are listed in the output. |
| 3c | High | `except: pass` around the neuron-filter merge: if the merge failed, the **unfiltered** labels for all neurons were appended. | The bare except was removed, so errors are raised. |
| 3d | Low | `if swc_labels_filter:` tested the tuple itself (always true), so the `[0]` on/off flag was ignored. | It now tests `swc_labels_filter[0]`. |
| 3e | Low | Error messages hid the real cause. | Messages now include the path and the original exception. The function returns a stats dict. |

### #4: `generate_internal_subtrees`

| ID | Severity | Problem | Fix |
|---|---|---|---|
| 4a | Medium | `output_filepath` is a `str`, but `.exists()` was called on it, so any run with `overwrite=False` raises `AttributeError` (only `overwrite=True` short-circuited past it). With `overwrite=True`, sub-tree files from an older version of the tree remained next to the new ones. | Uses `Path`. `overwrite=True` empties the neuron's folder first; `overwrite=False` with an existing folder skips the neuron and returns `None`. |
| 4b | Medium | `Path(output_dir, neuron_number)` raises `TypeError` when the ID is an `int`. | `str(neuron_number)`. |
| 4c | Info | The root's sub-tree (the whole neuron) is never written. This is by design, according to the docstring. | Unchanged; documented. |

The function now returns the number of sub-trees written. Validation: for every neuron tested, the number of sub-tree files equals the number of internal nodes minus the root.

### #5: `swc2json`

| ID | Severity | Problem | Fix |
|---|---|---|---|
| 5a | High | If there are several nodes with `parent == -1`, `root` is overwritten, so only the **last** connected component is exported and the others are dropped silently. Simplification creates extra roots when a parent ID is missing from the file. | Raises `ValueError` listing the roots. |
| 5b | Medium | `build_node` and `json.dump` are recursive, so trees deeper than ~1000 levels raise `RecursionError`. | The depth is computed iteratively and the recursion limit raised as needed. Tested on a 3000-level tree. |
| 5c | Medium | Duplicate `node_id` rows add the same child twice (duplicated sub-trees, or a false "cycle" error). Upstream `drop_duplicates()` calls were masking this. | Raises `ValueError` on duplicate node IDs. |
| 5d | Medium | Nodes that can't be reached from the root (cycles, broken parent links) were silently left out. | Raises if the number of visited nodes differs from the number of rows. |
| 5e | Low | `pd.notna(list)` was evaluated before the list check, so list-typed labels would raise "ambiguous truth value". | The checks were reordered. |

The function returns `{n_nodes, n_labeled, depth}`. Validation: the JSON node count equals the CSV row count for every neuron tested.

### #6: Other functions

| ID | Severity | Location | Problem | Fix |
|---|---|---|---|---|
| 6a | Medium | `join_swc_clumpiness` | Returned early whenever the output existed, with no overwrite option, so re-runs kept stale results. `pivot_table(aggfunc="first")` silently picked one of several scores for the same node and label. | Added an `overwrite` parameter; raises on duplicate `(node_id, label)` rows. Returns the number of scored nodes. |
| 6b | Medium | `process_clumpiness_csv` | `except Exception: return None` silently dropped CSVs it could not parse. `split('_')` breaks on IDs that contain `_`. | Returns `(df, error)`; `compile_unified_dataset` collects and reports the failures. Uses `rsplit('_', 1)`. |
| 6c | Low | `process_single_clumpiness` | The find-clumpiness stderr was captured and then thrown away. | The stderr text is included in the raised exception. |
| 6d | Low | `compile_unified_dataset` | The parquet schema was fixed by the first batch, so a later batch with different column types crashes. | Later batches are cast to the writer's schema. Returns `{files, rows, failed_files}`. |
| 6e | Low | `simplify_swc_topology` | An unreadable path printed a message and then crashed with `NameError`. | Raises a clear error. |

### #7: `pipeline.py`, sub-tree input file name (High)

`process_neuron` read `<id>_0.json`, but `swc2json` writes `<id>.json`. With the current `swc2json`, every neuron fails at the sub-tree step. The failure was hidden because `process_neuron` returns an error string, which `run_pipeline.py` only prints. Fixed in `pipeline_claude.py`.

### #8: `ProcessSWC` class (notebook)

| Problem | Fix |
|---|---|
| `bool(config["overwrite_info"])`: `bool("False")` is `True`, so `overwrite` was always on. *Confirmed: the config says `"False"` and the notebook showed `True`.* | `_parse_bool`, which compares the text. |
| `data_limit` stayed the string `"None"`. | `_parse_limit` returns `int` or `None`. |
| `step01` skipped all work if the parquet existed, never produced a neuron list, and set `self.labels_path` only when called. | `labels_path` is set in `__init__`. `step01` always returns the sorted relevant IDs (strings on both sides, `Path.stem`), cut to `data_limit`. |
| The overwrite rules differed per step, so a re-run could mix new step-2 output with old step-3 output. | One rule for all steps: skip if the output exists and `overwrite` is false. |
| Paths were relative to the current working directory. | Resolved against the config file's folder. |
| Steps returned nothing. | Each step returns a stats dict, ready for a per-neuron record. |

---

## 4. Validation

All validation outputs were written to a temporary folder, not to `output/`.

**End-to-end on real data:** `ProcessSWC` steps 1–5 ran on 5 neurons (`data_limit=5`, `overwrite=True`). Building the labels parquet over 93 label folders took about 5 minutes. All 5 neurons passed:
- labels equal the ground truth computed independently from the parquet;
- the JSON node count equals the CSV row count;
- the number of sub-tree files equals the number of internal nodes minus the root;
- each simplified tree has exactly one root.

**Edge cases (synthetic):**

| Case | Result |
|---|---|
| Two roots | `ValueError` |
| Duplicate node | `ValueError` |
| Cycle | `ValueError` (unreachable nodes) |
| 3000-level tree | Exported |
| Sub-trees with `overwrite=False` and an existing folder | Skipped |
| `int` neuron ID | Works |
| `pipeline_claude.process_neuron` on a real neuron | Success; 41 labelled nodes (old: 5) |

**Not validated:**
- steps 6–7 (find-clumpiness is not installed in the environment used for validation, and the class methods are still empty);
- `compile_unified_dataset` and `join_swc_clumpiness`, which were changed but not run.

---

## 5. Remaining work

1. Decide on issue #2 before regenerating any results.
2. Implement steps 6–7 in the class. Read only `5_json_divided/<neuron_id>/`, never the whole folder.
3. Add a `run_neuron` / `run_all` orchestration that collects the per-step stats into one record per neuron.
4. `run_pipeline.py` still imports the original modules and was not modified. Its step 4 lists the whole `output_json` folder, which now contains per-neuron sub-folders. Replace it with the class rather than patching it.
5. ~~Update `config.json`~~: done, via the `labels_ftr_path` key (see section 2).
