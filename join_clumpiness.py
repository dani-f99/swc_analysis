import os
import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed

def process_clumpiness_csv(filepath_str: str):
    """
    Reads a single CSV, safely skips empty files, and appends ID columns.
    """
    filepath = Path(filepath_str)
    
    # Fast-skip: Files <= 32 bytes physically cannot contain data rows
    if os.path.getsize(filepath) <= 32:
        return None
        
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return None
            
        neuron_id, node_id = filepath.stem.split('_')
        
        df['neuron_id'] = neuron_id
        df['node_id'] = node_id
        
        return df
        
    except Exception:
        return None

def chunked_iterable(iterable, size):
    """Yields batches of a specified size from an iterable."""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

def compile_unified_dataset(input_directory: str, output_filepath: str, batch_size: int = 2000) -> None:
    """
    Iterates over CSVs and processes them using joblib for robust parallel execution.
    """
    def get_csv_files():
        with os.scandir(input_directory) as entries:
            for entry in entries:
                if entry.name.endswith('.csv') and entry.is_file():
                    yield entry.path

    writer = None
    n_jobs = min(4, (os.cpu_count() or 1))
    
    with tqdm(desc="Compiling Parquet", unit=" files") as pbar:
        for file_chunk in chunked_iterable(get_csv_files(), batch_size):
            
            # joblib handles the worker pool much more safely on Windows
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(process_clumpiness_csv)(f) for f in file_chunk
            )
            
            valid_dfs = [df for df in results if df is not None]
            
            if valid_dfs:
                batch_df = pd.concat(valid_dfs, ignore_index=True)
                table = pa.Table.from_pandas(batch_df)
                
                if writer is None:
                    writer = pq.ParquetWriter(output_filepath, table.schema, compression='ZSTD')
                
                writer.write_table(table)
            
            pbar.update(len(file_chunk))
            
    if writer:
        writer.close()
    print("\nDataset compilation complete.")

if __name__ == "__main__":
    input_dir = os.path.join("data", "output_clumpiness")
    output_file = os.path.join("data", "unified_clumpiness.parquet")
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    compile_unified_dataset(input_dir, output_file)