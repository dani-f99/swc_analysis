from scripts.pipeline_unitest import PipelineSWC
from scripts.helpers import run_pipeline

# Execute the pipeline
if __name__ == "__main__":
    pipeline1_result = run_pipeline(PipelineSWC, pipeline_name="swc_analysis_test")
