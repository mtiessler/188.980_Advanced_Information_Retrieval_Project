import logging
logging.basicConfig(
    filename="BM25_main_run.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

import config
from pipeline import RetrievalPipeline
import os

logging.info("Starting Main Execution")
logging.info(f"Current Working Directory: {os.getcwd()}")
#logging.info(f"Using device: {config.DEVICE}")
logging.info(f"DEMO MODE ACTIVE: {config.IS_DEMO_MODE}")
if config.IS_DEMO_MODE:
    logging.info(f"Processing max {getattr(config, 'DEMO_FILES_LIMIT', 1)} document file(s).")


if __name__ == "__main__":
    # --- BM25 Parameter Search Examples (commented out) ---
    # This values where used for grid search and manual tuning.
    # 1) k1s = [1.2, 1.5, 1.8, 2.0]; bs = [0.6, 0.7, 0.75, 0.85]
    # 2) k1s = [0.95, 1.0, 1.05, 1.1, 1.2]; bs = [0.65, 0.7, 0.75]
    # 3) k1s = [0.9, 0.85]; bs = [0.75, 0.8]
    # 4) k1s = [0.8, 0.75]; bs = [0.7, 0.75, 0.8]
    # 5) k1s = [0.7, 0.65, 0.6]; bs = [0.75, 0.8, 0.85]
    # 6) k1s = [0.55, 0.5, 0.4]; bs = [0.75, 0.8, 0.85]
    # 7) k1s = [0.1, 0.2, 0.3]; bs = [0.7, 0.75, 0.8, 0.85]

    # Example of best parameters found so far (for documentation):
    # top 1 for abstract: k=1.0/b=0.7: nDCG@10: 0.1778
    # top 1 for fullText: k=0.2/b=0.8: nDCG@10: 0.2002
    # general: b=0.7 - 0.8 delivers good results with all k

    # --- Set BM25 parameters for this run ---
    k1s = [0.2]
    bs = [0.8]

    # Loop over all parameter combinations (expand lists above for grid search)
    for k1 in k1s:
        for b in bs:
            config.BM25_K1 = k1
            config.BM25_B = b
            pipeline = RetrievalPipeline() # Create a new pipeline instance for each run

            try:
                pipeline.load_data()            # Load documents, queries, and qrels
                pipeline.setup_preprocessing()  # Initialize the preprocessor

                print(f"Running with parameters: b={b}, k1={k1}")
                logging.info(f"Running with parameters: b={b}, k1={k1}")

                # --- Run BM25 Ranking Pipeline Traditional IR (BM25 rank_bm25 Cache) ---
                bm25_rank_run, bm25_rank_system_name = pipeline.run_bm25_rank() # Call the new rank_bm25 run method
                print(f"Evaluation performance of parameters: b={b}, k1={k1}")
                logging.info(f"Evaluation performance of parameters: b={b}, k1={k1}")

                # --- Evaluate the Run ---
                pipeline.run_evaluation(bm25_rank_run, bm25_rank_system_name)

            except Exception as e:
                logging.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
            finally:
                logging.info("Main Execution Finished")
