import logging
logging.basicConfig(
    filename="main_run.log",
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
logging.info(f"Using device: {config.DEVICE}")
logging.info(f"Cross-Encoder Model: {config.CROSS_ENCODER_MODEL_NAME}")
logging.info(f"DEMO MODE ACTIVE: {config.IS_DEMO_MODE}")
if config.IS_DEMO_MODE:
    logging.info(f"Processing max {getattr(config, 'DEMO_FILES_LIMIT', 1)} document file(s).")

# copy metadate-file into submission folder
def prepare_submission_root(submission_root):
    os.makedirs(submission_root, exist_ok=True)
    shutil.copy("ir-metadata.yml", submission_root)


# copies run files into right submission folder
#renames to run.txt, compresses to run.txt.gz, and removes all intermediate files except run.txt.gz.
def prepare_submission_run_file(output_dir, sub_folder, run_file_prefix, must_contain=None):
    os.makedirs(sub_folder, exist_ok=True)
    all_files = os.listdir(output_dir)
    run_files = [
        os.path.join(output_dir, f)
        for f in all_files
        if f.startswith(run_file_prefix) and f.endswith(".txt") and (must_contain is None or must_contain in f) 
            and not_contain not in f
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
        # For BM25
        bm25_root = os.path.join(config.OUTPUT_DIR_SUBMISSON, "Submission_BM25")
        prepare_submission_root(bm25_root)

        # For BERT
        bert_root = os.path.join(config.OUTPUT_DIR_SUBMISSON, "Submission_BERT")
        prepare_submission_root(bert_root)

        # For RERANK
        rerank_root = os.path.join(config.OUTPUT_DIR_SUBMISSON, "Submission_RERANK")
        prepare_submission_root(rerank_root)

        #############################################################
        #############################################################

        # --- Run Pipeline ---
        # --- Time T1 2024-11 ---
        config.QUERIES_FILE = config.QUERIES_FILE_SUBMISSON_T1
        print(config.QUERIES_FILE)

        pipeline.load_data()
        pipeline.setup_preprocessing()

        # --- Run Traditional IR (BM25) ---
        bm25_rank_run, bm25_rank_system_name = pipeline.run_bm25_rank()
        sub_folder = os.path.join(bm25_root, "2024-11")
        prepare_submission_run_file(config.OUTPUT_DIR, sub_folder, "run_BM25_")

        # ## copy from ouput to submission folder:
        # # -- create folder structure if not exists
        # sub_folder = os.path.join(config.OUTPUT_DIR_SUBMISSON, "Submission_BM25", "2024-11")
        # os.makedirs(sub_folder, exist_ok=True)

        # # Get the run*.txt files from OUTPUT_DIR and copy them to the submission folder
        # # List all files in the output directory
        # all_files = os.listdir(config.OUTPUT_DIR)

        # # Filter for files that start with 'run_BM25_' and end with '.txt'
        # run_files = [
        #     os.path.join(config.OUTPUT_DIR, f)
        #     for f in all_files
        #     if f.startswith("run_BM25_") and f.endswith(".txt")
        # ]

        # # Copy each file to the submission folder
        # for file_path in run_files:
        #     shutil.copy(file_path, sub_folder)

        # # After you have your run_files list (should contain only one file)
        # if len(run_files) != 1:
        #     raise RuntimeError(f"Expected exactly one run file, found {len(run_files)}: {run_files}")

        # # 1) Copy and rename to run.txt
        # src_file = run_files[0]
        # dst_file = os.path.join(sub_folder, "run.txt")
        # shutil.copy(src_file, dst_file)

        # # 2) Compress run.txt to run.txt.gz
        # with open(dst_file, 'rb') as f_in, gzip.open(dst_file + '.gz', 'wb') as f_out:
        #     shutil.copyfileobj(f_in, f_out)

        # # 3) Remove the uncompressed run.txt
        # os.remove(dst_file)

        # # 4) Remove the original run_BM25_.txt
        # file_in_sub_folder = os.path.join(sub_folder, os.path.basename(src_file))
        # if os.path.exists(file_in_sub_folder):
        #     os.remove(file_in_sub_folder)

        '''
        # --- Run Pipeline 2: Representation Learning (MS-MARCO) ---
        bert_dense_run, bert_dense_system_name = pipeline.run_bert_dense()
        sub_folder = os.path.join(bert_root, "2024-11")
        prepare_submission_run_file(config.OUTPUT_DIR, sub_folder, "run_BERT_")

        # --- Run Pipeline 3: Neural Re-ranking (BM25 + MS-MARCO) ---
        hybrid_run, hybrid_system_name = pipeline.run_hybrid_rerank()
        sub_folder = os.path.join(rerank_root, "2024-11")
        prepare_submission_run_file(config.OUTPUT_DIR, sub_folder, "run_BM25_", must_contain="_then_")
        '''
       #############################################################

        # --- Time T2 2025-01 ---
        # --- Run Pipeline 1: Traditional IR (BM25) ---
        config.QUERIES_FILE = config.QUERIES_FILE_SUBMISSON_T2
        print(config.QUERIES_FILE)

        pipeline.load_data()
        pipeline.setup_preprocessing()

        # --- Run Traditional IR (BM25) ---
        bm25_rank_run, bm25_rank_system_name = pipeline.run_bm25_rank()
        #sub_folder = os.path.join(config.OUTPUT_DIR_SUBMISSON, "Submission_BM25", "2025-01")
        sub_folder = os.path.join(bm25_root, "2025-01")
        prepare_submission_run_file(config.OUTPUT_DIR, sub_folder, "run_BM25_")

        '''
        # --- Run Pipeline 2: Representation Learning (MS-MARCO) ---
        bert_dense_run, bert_dense_system_name = pipeline.run_bert_dense()
        sub_folder = os.path.join(bert_root, "2025-01")
        prepare_submission_run_file(config.OUTPUT_DIR, sub_folder, "run_BERT_")

        # --- Run Pipeline 3: Neural Re-ranking (BM25 + MS-MARCO) ---
        hybrid_run, hybrid_system_name = pipeline.run_hybrid_rerank()
        sub_folder = os.path.join(rerank_root, "2025-01")
        prepare_submission_run_file(config.OUTPUT_DIR, sub_folder, "run_BM25_", must_contain="_then_")
        '''


    except Exception as e:
        logging.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
    finally:
        logging.info("Main Execution Finished")