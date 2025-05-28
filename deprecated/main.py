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
logging.info("Starting Main Execution")
logging.info(f"Current Working Directory: {os.getcwd()}")
logging.info(f"Using device: {config.DEVICE}")
logging.info(f"Cross-Encoder Model: {config.CROSS_ENCODER_MODEL_NAME}")
logging.info(f"DEMO MODE ACTIVE: {config.IS_DEMO_MODE}")
if config.IS_DEMO_MODE:
    logging.info(f"Processing max {getattr(config, 'DEMO_FILES_LIMIT', 1)} document file(s).")


if __name__ == "__main__":
    pipeline = RetrievalPipeline()

    try:
        pipeline.load_data()
        pipeline.setup_preprocessing()

        # --- Run Pipeline 1: Traditional IR (BM25) ---
        bm25_rank_run, bm25_rank_system_name = pipeline.run_bm25_rank()
        pipeline.run_evaluation(bm25_rank_run, bm25_rank_system_name)

        # --- Run Pipeline 2: Representation Learning (MS-MARCO) ---
        bert_dense_run, bert_dense_system_name = pipeline.run_bert_dense()
        pipeline.run_evaluation(bert_dense_run, bert_dense_system_name)

        # --- Run Pipeline 3: Neural Re-ranking (BM25 + MS-MARCO) ---
        hybrid_run, hybrid_system_name = pipeline.run_hybrid_rerank()
        pipeline.run_evaluation(hybrid_run, hybrid_system_name)

    except Exception as e:
        logging.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
    finally:
        logging.info("Main Execution Finished")