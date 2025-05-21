import logging
import time
import os
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
import gc
import torch

import config
import data_loader
import utils
from evaluation import evaluate_run
from preprocessing import Preprocessor
from retrievers.bm25_rank_retriever import BM25RankRetriever


class RetrievalPipeline:
    """
    Main pipeline class for running document retrieval, ranking and evaluation.
    Handles data loading, preprocessing, BM25 ranking and evaluation.
    """    
    def __init__(self):
        self.documents = {}             # Dictionary of document data
        self.doc_ids_in_order = []      # List of document IDs in original order
        self.queries = {}               # Dictionary of queries
        self.qrels = {}                 # Dictionary/list of relevance judgments
        self.preprocessor = None

        self.bm25_rank_retriever = None

        # Ensure output and cache directories exist
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.CACHE_DIR, exist_ok=True)

    def load_data(self):
        """
        Load documents, queries and qrels using data_loader utilities.
        Logs the number of loaded items and checks for missing data.
        """        
        logging.info("--- Loading Data ---")
        try:
            # Load documents and their order
            self.documents, self.doc_ids_in_order = data_loader.load_documents_structure(
                config.DOCUMENTS_DIR,
                content_field=config.CONTENT_FIELD
            )
            # Load queries and qrels
            self.queries = data_loader.load_queries(config.QUERIES_FILE)
            self.qrels = data_loader.load_qrels(config.QRELS_FILE)

            # Check for missing essential data
            if not self.documents or not self.queries:
                raise RuntimeError(
                    "Failed to load essential data (documents or queries). Check paths and file formats.")
            if not self.qrels:
                logging.warning("QRELs file loaded empty or not found. Evaluation might not work.")

            logging.info(f"Loaded {len(self.documents)} documents ({len(self.doc_ids_in_order)} ordered IDs).")
            logging.info(f"Loaded {len(self.queries)} queries.")
            logging.info(f"Loaded QRELs for {len(self.qrels)} queries.")

        except FileNotFoundError as e:
            logging.error(f"Data loading failed: File not found - {e}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Data loading failed: An unexpected error occurred - {e}", exc_info=True)
            raise
        finally:
            gc.collect()
            logging.info("--- Data Loading Finished ---")

    def setup_preprocessing(self):
        """
        Initialize the Preprocessor if not already set up.
        Uses configuration values for stopword removal and stemming.
        """        
        if self.preprocessor is None:
            logging.info("--- Setting up Preprocessor ---")
            self.preprocessor = Preprocessor(
                remove_stopwords=config.REMOVE_STOPWORDS,
                enable_stemming=config.ENABLE_STEMMING
            )
            logging.info("--- Preprocessor Setup Finished ---")

    def _get_document_text(self, doc_id):
        """
        Retrieve and combine the relevant text fields for a document by ID.
        Handles missing fields and avoids duplicate content.

        Args:
            doc_id (str): Document ID to retrieve.

        Returns:
            str: Combined text from the document fields.

        """        
        doc_data = self.documents.get(doc_id)
        if not doc_data:
            logging.warning(f"Required document data not found for ID: {doc_id}")
            return ""

        # Extract fields with fallback to empty string
        title = doc_data.get('title', '') or ""
        abstract_raw = doc_data.get('abstract', '') or ""
        main_content_raw = doc_data.get(config.CONTENT_FIELD, '') or ""

        # Validate abstract and main content: must be string, not a URL, and >10 chars
        abstract = ""
        if abstract_raw and isinstance(abstract_raw, str) and not abstract_raw.startswith(
                ("http://", "https://")) and len(abstract_raw) > 10:
            abstract = abstract_raw

        main_content = ""
        if main_content_raw and isinstance(main_content_raw, str) and not main_content_raw.startswith(
                ("http://", "https://")) and len(main_content_raw) > 10:
            main_content = main_content_raw

        # Combine fields based on config.CONTENT_FIELD
        if config.CONTENT_FIELD == 'abstract':
            combined_text = f"{title} {abstract}".strip()
        else:
            if abstract and main_content and abstract == main_content:
                combined_text = f"{title} {abstract}".strip()
            elif abstract and main_content:
                combined_text = f"{title} {abstract} {main_content}".strip()
            elif abstract:
                combined_text = f"{title} {abstract}".strip()
            elif main_content:
                combined_text = f"{title} {main_content}".strip()
            else:
                combined_text = title.strip()

        return combined_text if combined_text else ""


    def setup_bm25_rank(self):
        """
        Initialize and build/load the BM25 index. Uses token caching.
        Uses a factory function to stream documents for memory efficiency.
        """        
        if self.bm25_rank_retriever is None:
            logging.info("--- Setting up BM25 (rank_bm25 with Caching) ---")
            if self.preprocessor is None: self.setup_preprocessing()

            token_cache_file = config.BM25_TOKEN_CACHE_FILE

            self.bm25_rank_retriever = BM25RankRetriever(
                preprocessor=self.preprocessor,
                k1=config.BM25_K1,
                b=config.BM25_B,
                token_cache_file=token_cache_file
            )

            def _doc_iterator_factory():
                """
                Factory function to stream documents for BM25 index building.
                """                
                limit = config.DEMO_FILES_LIMIT if config.IS_DEMO_MODE else None
                logging.info(f"BM25 Index: Streaming documents from {config.DOCUMENTS_DIR} (Limit: {limit})")
                return data_loader.stream_document_dicts(
                    config.DOCUMENTS_DIR,
                    content_field=config.CONTENT_FIELD,
                )

            # Pass the factory function to the load/build method
            logging.info(f"BM25: Loading or building index (Chunk size: {config.BM25_TOKEN_CHUNK_SIZE})...")
            index_built = self.bm25_rank_retriever.load_or_build_index(
                doc_iterator_factory=_doc_iterator_factory,
                chunk_size=config.BM25_TOKEN_CHUNK_SIZE,
            )

            if not index_built or self.bm25_rank_retriever.bm25 is None:
                logging.error("BM25 (rank_bm25) index building failed.")
                self.bm25_rank_retriever = None
                return False

            logging.info("--- BM25 (rank_bm25) Setup Finished ---")
        return True

    def run_bm25_rank(self):
        """
        Run BM25 ranking for all queries, save results.
         
        Returns:
            dict: Dictionary of run results with query IDs as keys and ranked document IDs as values.
            str: Execution name for the run, used for saving and logging.   

        """        
        logging.info("--- Starting Pipeline: Basic BM25 (rank_bm25 Cache) ---")
        start_time = time.time()

        # Build a descriptive execution name based on config:
        k1_str = str(config.BM25_K1).replace('.', 'p')
        b_str = str(config.BM25_B).replace('.', 'p')

        bm25_execution_name = f"BM25_k1_{k1_str}_b_{b_str}"
        bm25_execution_name += "_stop" if config.REMOVE_STOPWORDS else ""
        bm25_execution_name += "_stem" if config.ENABLE_STEMMING else ""
        bm25_execution_name += f"_{config.CONTENT_FIELD}"
        bm25_execution_name += "_demo" if config.IS_DEMO_MODE else ""

        # Setup BM25 index and retriever
        if not self.setup_bm25_rank():
            logging.error("BM25 setup failed. Aborting BM25 run.")
            return None, f"{bm25_execution_name}_FAILED"

        if self.bm25_rank_retriever is None or self.bm25_rank_retriever.bm25 is None:
            logging.error("BM25 retriever not ready after setup. Aborting search.")
            return None, f"{bm25_execution_name}_FAILED"

        run_results = {}        # Dictionary to store results for each query
        num_queries = len(self.queries)
        processed_count = 0
        logging.info(f"Processing {num_queries} queries with BM25...")

        # Iterate over all queries and perform BM25 search
        for qid, query_text in tqdm(self.queries.items(), desc=f"BM25 Search ({bm25_execution_name})"):
            if not query_text:
                logging.warning(f"Skipping empty query for QID: {qid}")
                run_results[qid] = {}
                continue
            try:
                #Using FINAL_TOP_K
                query_run_results = self.bm25_rank_retriever.search(query_text, k=config.FINAL_TOP_K)
                run_results[qid] = query_run_results
                processed_count += 1
            except Exception as e:
                logging.error(f"Error processing query QID {qid} with BM25: {e}", exc_info=True)
                run_results[qid] = {}

        logging.info(f"Processed {processed_count}/{num_queries} queries.")

        # Save the run results to file
        run_file_name = f"run_{bm25_execution_name}.txt"
        run_file_path = os.path.join(config.OUTPUT_DIR, run_file_name)
        try:
            utils.save_run(run_results, config.OUTPUT_DIR, run_file_name, bm25_execution_name)
            logging.info(f"BM25 run results saved to {run_file_path}")
        except Exception as e:
            logging.error(f"Failed to save BM25 run results: {e}")

        end_time = time.time()
        logging.info(f"--- Finished Pipeline: Basic BM25 ({(end_time - start_time):.2f} seconds) ---")
        return run_results, bm25_execution_name


    def run_evaluation(self, run_results, execution_name):
        """
        Run evaluation for the given run results and execution name.
        
        Args:
            run_results (dict): Dictionary of run results with query IDs as keys and ranked document IDs as values.
            execution_name (str): Name of the execution for logging and saving results. 
        """
        if run_results is None:
            logging.warning(f"Skipping evaluation for {execution_name}: run results are None.")
            return
        if not self.qrels:
            logging.warning(f"Skipping evaluation for {execution_name}: QRELs not loaded.")
            return
        if not run_results:
             logging.warning(f"Skipping evaluation for {execution_name}: run results dictionary is empty.")
             return
        if not config.EVALUATION_MEASURES:
            logging.warning(f"Skipping evaluation for {execution_name}: No evaluation measures defined in config.")
            return

        logging.info(f"--- Evaluating System: {execution_name} using measures {config.EVALUATION_MEASURES} ---")
        try:
            evaluate_run(
                run_results=run_results,
                qrels=self.qrels,
                metrics_set=set(config.EVALUATION_MEASURES),
                run_name=execution_name,
            )
            logging.info(f"--- Evaluation Finished for {execution_name} ---")
        except FileNotFoundError:
            logging.error("Evaluation failed: QREL file seems unavailable during evaluation step.", exc_info=True)
        except Exception as e:
            logging.error(f"An error occurred during evaluation for {execution_name}: {e}", exc_info=True)
