--------------------------------------------------------------------------------
                              COMPUTATIONAL BIOLOGY LAB
                                  Haifa University
                                   Daniel Fridman
                                        2026
--------------------------------------------------------------------------------

# PROJECT: SWC Neuron Morphological Analysis

## 1. OVERVIEW
This program utilizes a Python data pipeline to compute clumpiness scores for neuron SWC tree datasets and extract sub-trees for non-leaf internal nodes. The pipeline parses recursive JSON representations of SWC files to map the neuronal hierarchies. Parallel task processing is implemented using joblib to batch-process thousands of files efficiently. Once the metric computations are complete, the resulting dataframes are exported directly into compressed files using ZSTD compression.

## 2. PREREQUISITES
Please ensure the following modules are installed:

### Required Python Modules:
- Pandas 
- NumPy 
- joblib
- pyarrow 
- tqdm

### Required Program:
- [find-clumpiness](https://github.com/GregorySchwartz/find-clumpiness)


## 3. CONFIGURATION (config.json)
Before running the analysis, configure the `config.json` file as follows: 
1. run_name - 
2. swc_path - Path to the folder which contains the raw swc datasets `sk_lod1_783_healed` folder (`under data/input/input_swc`).
3. labels_path - Path to the folder which contains labels files: `processed_swc_data_princeton` (folder)  and `neuron_data_full_article_princeton.ftr` (`under data/input/input_labels`).
4. results_swc_path - path to which save the simplified output of the simplified swc files.
5. results_labels_path - path to which save the relevent swc's metadata labels.
6. overwrite_info - overwrite labels parquet files (bool).
7. data_limit - limit number of dataset to be processed (for testing purposes).
8. n_threads - Number of CPU thread to be utilized during the pipeline execution.
9. n_jobs - Number of parallel jobs during the processing.


## 4. USAGE GUIDE
1. Configure the `config.json` file as needed (see section above). 
2. Install [find-clumpiness](https://github.com/GregorySchwartz/find-clumpiness) and make sure it exsists on your system (try find-clumpiness --help).
2. Run the pipeline script `run_pipeline.py`, preferably on a unix server capable of handling high-throughput parallel task processing. 
3. Review the `output/results_output` directory for the extracted non-leaf internal node sub-trees and ZSTD-compressed metrics files.

## 5. DIRECTORY STRUCTURE
The program uses the following folder structure: 

- input/  : Input folder. 
- output/ : Result output, tables to be used in downstream analysis. 

## 6. RESOURCES
- find clumpiness - [https://github.com/GregorySchwartz/find-clumpiness](https://github.com/GregorySchwartz/find-clumpiness)
- fly brainscan (swc files) - [https://codex.flywire.ai/api/download?dataset=fafb](https://codex.flywire.ai/api/download?dataset=fafb)