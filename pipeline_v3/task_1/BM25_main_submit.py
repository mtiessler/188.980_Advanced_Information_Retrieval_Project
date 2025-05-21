import logging
logging.basicConfig(
    filename="BM25_main_submit_run.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

import config
from pipeline import RetrievalPipeline
import os
import shutil
import gzip
logging.info("Starting Main Execution")
logging.info(f"Current Working Directory: {os.getcwd()}")
#logging.info(f"Using device: {config.DEVICE}")
logging.info(f"DEMO MODE ACTIVE: {config.IS_DEMO_MODE}")
if config.IS_DEMO_MODE:
    logging.info(f"Processing max {getattr(config, 'DEMO_FILES_LIMIT', 1)} document file(s).")

def prepare_submission_root(submission_root):
    """
    Create the submission root directory and copy the metadata file into it.

    Args:
        submission_root (str): Path to the root submission directory.

    Returns:
        None
    """    
    os.makedirs(submission_root, exist_ok=True)
    shutil.copy("ir-metadata.yml", submission_root)

def prepare_submission_run_file(output_dir, sub_folder, run_file_prefix, must_contain=None):
    """
    Copy the run file into the correct submission folder, 
    rename to run.txt, compress to run.txt.gz
    and remove all intermediate files except the compressed file.

    Args:
        output_dir (str): Directory where run files are located.
        sub_folder (str): Submission subfolder to copy files into.
        run_file_prefix (str): Prefix of the run file to look for.
        must_contain (str, optional): Additional string that must be in the filename.

    """
    os.makedirs(sub_folder, exist_ok=True)
    all_files = os.listdir(output_dir)
    # Find run files matching the prefix and optional substring
    run_files = [
        os.path.join(output_dir, f)
        for f in all_files
        if f.startswith(run_file_prefix) and f.endswith(".txt") and (must_contain is None or must_contain in f) 
    ]
    if len(run_files) != 1:
        raise RuntimeError(f"Expected exactly one run file, found {len(run_files)}: {run_files}")

    # Copy and rename to run.txt
    src_file = run_files[0]
    dst_file = os.path.join(sub_folder, "run.txt")
    shutil.copy(src_file, dst_file)

    # Compress run.txt to run.txt.gz
    with open(dst_file, 'rb') as f_in, gzip.open(dst_file + '.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    # Remove the uncompressed run.txt
    os.remove(dst_file)

    # Remove the original run_*.txt from sub_folder if it exists
    file_in_sub_folder = os.path.join(sub_folder, os.path.basename(src_file))
    if os.path.exists(file_in_sub_folder):
        os.remove(file_in_sub_folder)


if __name__ == "__main__":
    pipeline = RetrievalPipeline()

    try:
        # --- Prepare Submission Folders ---
        # For BM25: create root submission directory and copy metadata
        bm25_root = os.path.join(config.OUTPUT_DIR_SUBMISSON, "Submission_BM25")
        prepare_submission_root(bm25_root)

        #############################################################
        # --- Run Pipeline for Time T1 2024-11 ---
        config.QUERIES_FILE = config.QUERIES_FILE_SUBMISSON_T1
        print(config.QUERIES_FILE)

        pipeline.load_data()            # Load documents, queries, and qrels
        pipeline.setup_preprocessing()  # Initialize the preprocessor

        # --- Run BM25 and save run file for T1 ---
        bm25_rank_run, bm25_rank_system_name = pipeline.run_bm25_rank()
        sub_folder = os.path.join(bm25_root, "2024-11")
        prepare_submission_run_file(config.OUTPUT_DIR, sub_folder, "run_BM25_")

        #############################################################
        # --- Run Pipeline for Time T2 2025-01 ---
        config.QUERIES_FILE = config.QUERIES_FILE_SUBMISSON_T2
        print(config.QUERIES_FILE)

        pipeline.load_data()
        pipeline.setup_preprocessing()

        # --- Run BM25 and save run file for T2 ---
        bm25_rank_run, bm25_rank_system_name = pipeline.run_bm25_rank()
        #sub_folder = os.path.join(config.OUTPUT_DIR_SUBMISSON, "Submission_BM25", "2025-01")
        sub_folder = os.path.join(bm25_root, "2025-01")
        prepare_submission_run_file(config.OUTPUT_DIR, sub_folder, "run_BM25_")

    except Exception as e:
        logging.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
    finally:
        logging.info("Main Execution Finished")
