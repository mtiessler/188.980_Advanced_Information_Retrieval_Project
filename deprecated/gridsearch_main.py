import logging
logging.basicConfig(
    filename="gridsearch_run.log",
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
logging.info(f"Embedding Model: {config.EMBEDDING_MODEL_NAME}")
logging.info(f"Cross-Encoder Model: {config.CROSS_ENCODER_MODEL_NAME}")
logging.info(f"DEMO MODE ACTIVE: {config.IS_DEMO_MODE}")
if config.IS_DEMO_MODE:
    logging.info(f"Processing max {getattr(config, 'DEMO_FILES_LIMIT', 1)} document file(s).")


if __name__ == "__main__":
    # 1) 
    ## k1s = [1.2, 1.5, 1.8, 2.0]
    ## bs = [0.6, 0.7, 0.75, 0.85]
    # 2) 
    ## k1s = [0.95, 1.0, 1.05, 1.1, 1.2]
    ## bs = [0.65, 0.7, 0.75]
    # 3) 
    ##k1s = [0.9, 0.85]
    ##bs = [0.75, 0.8]
    # 4)
    ## k1s = [0.8, 0.75]
    ## bs = [0.7, 0.75, 0.8]
    # 5)
    ## k1s = [0.7, 0.65, 0.6]
    ## bs = [0.75, 0.8, 0.85]
    # 6) 
    ## k1s = [0.55, 0.5, 0.4]
    ## bs = [0.75, 0.8, 0.85]
    # 7)
    ## k1s = [0.1, 0.2, 0.3]
    ## bs = [0.7, 0.75, 0.8, 0.85]

    # top 1 for abstract: k=1.0/b=0.7: nDCG@10: 0.1778
    # top 1 for fullText: k=0.2/b=0.8: nDCG@10: 0.2002

    ## general: b=0.7 - 0.8 delivers good results with all k

    k1s = [0.2]
    bs = [0.8]

    for k1 in k1s:
        for b in bs:
            config.BM25_K1 = k1
            config.BM25_B = b
            pipeline = RetrievalPipeline()

            try:
                pipeline.load_data()
                pipeline.setup_preprocessing()

                print(f"Running with parameters: b={b}, k1={k1}")
                logging.info(f"Running with parameters: b={b}, k1={k1}")

                # --- Run Pipeline 1: Traditional IR (BM25 rank_bm25 Cache) ---
                bm25_rank_run, bm25_rank_system_name = pipeline.run_bm25_rank() # Call the new rank_bm25 run method
                print(f"Evaluation performance of parameters: b={b}, k1={k1}")
                logging.info(f"Evaluation performance of parameters: b={b}, k1={k1}")
                pipeline.run_evaluation(bm25_rank_run, bm25_rank_system_name)
                '''
                # --- Run Pipeline 2: Representation Learning (BERT Dense) ---
                bert_dense_run, bert_dense_system_name = pipeline.run_bert_dense()
                pipeline.run_evaluation(bert_dense_run, bert_dense_system_name)

                # --- Run Pipeline 3: Neural Re-ranking (Hybrid) ---
                hybrid_run, hybrid_system_name = pipeline.run_hybrid_rerank()
                pipeline.run_evaluation(hybrid_run, hybrid_system_name)
                '''

            except Exception as e:
                logging.error(f"An error occurred during pipeline execution: {e}", exc_info=True)
            finally:
                logging.info("Main Execution Finished")