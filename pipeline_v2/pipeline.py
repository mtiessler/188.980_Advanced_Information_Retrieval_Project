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
from rerankers import BertReranker
from retrievers.dense_retriever import BertEmbedder
from retrievers.dense_retriever import FaissIndex
from transformers import AutoTokenizer, AutoModel
logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more detail
    format='%(asctime)s [%(levelname)s] %(message)s',
)


class RetrievalPipeline:
    def __init__(self):
        self.documents = {}
        self.doc_ids_in_order = []
        self.queries = {}
        self.qrels = {}
        self.preprocessor = None

        self.bm25_rank_retriever = None
        self.bert_reranker = None
        self.bert_dense_retriever = None

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.CACHE_DIR, exist_ok=True)

    def load_data(self):
        logging.info("--- Loading Data ---")
        try:
            self.documents, self.doc_ids_in_order = data_loader.load_documents_structure(
                config.DOCUMENTS_DIR,
                content_field=config.CONTENT_FIELD
            )
            self.queries = data_loader.load_queries(config.QUERIES_FILE)
            self.qrels = data_loader.load_qrels(config.QRELS_FILE)

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
        if self.preprocessor is None:
            logging.info("--- Setting up Preprocessor ---")
            self.preprocessor = Preprocessor(
                remove_stopwords=config.REMOVE_STOPWORDS,
                enable_stemming=config.ENABLE_STEMMING
            )
            logging.info("--- Preprocessor Setup Finished ---")

    def _get_document_text(self, doc_id):
        doc_data = self.documents.get(doc_id)
        if not doc_data:
            logging.warning(f"Required document data not found for ID: {doc_id}")
            return ""

        title = doc_data.get('title', '') or ""
        abstract_raw = doc_data.get('abstract', '') or ""
        main_content_raw = doc_data.get(config.CONTENT_FIELD, '') or ""

        abstract = ""
        if abstract_raw and isinstance(abstract_raw, str) and not abstract_raw.startswith(
                ("http://", "https://")) and len(abstract_raw) > 10:
            abstract = abstract_raw

        main_content = ""
        if main_content_raw and isinstance(main_content_raw, str) and not main_content_raw.startswith(
                ("http://", "https://")) and len(main_content_raw) > 10:
            main_content = main_content_raw

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

    def setup_bert_reranker(self):
        if self.bert_reranker is None:
            logging.info(f"--- Setting up BERT Re-ranker ({config.CROSS_ENCODER_MODEL_NAME}) ---")
            if self.preprocessor is None: self.setup_preprocessing()
            try:
                self.bert_reranker = BertReranker(
                    model_name=config.CROSS_ENCODER_MODEL_NAME,
                    device=config.DEVICE,
                )
                logging.info("--- BERT Re-ranker Setup Finished ---")
            except Exception as e:
                logging.error(f"Failed to initialize BERT Re-ranker: {e}", exc_info=True)
                self.bert_reranker = None
                return False
        return True

        # Placeholder for setting up the dense retriever (Task 2)
    def setup_bert_dense(self):
        if self.bert_dense_retriever is None:
            logging.info("--- Setting up BERT Dense Retriever ---")
            print("--- Setting up BERT Dense Retriever ---")
            try:
                self.bert_embedder = BertEmbedder(config.DENSE_MODEL_NAME, config.DEVICE)
                self.faiss_index = FaissIndex(dimension=self.bert_embedder.model.config.hidden_size)
    
                # Get document texts
                doc_texts = []
                doc_ids = []
                for doc_id in self.doc_ids_in_order:
                    text = self._get_document_text(doc_id)
                    if text:
                        doc_texts.append(text)
                        doc_ids.append(doc_id)
    
                # Embed documents
                doc_embeddings = self.bert_embedder.embed(
                    doc_texts,
                    batch_size=config.DENSE_BATCH_SIZE,
                    preprocessor=self.preprocessor
                )
    
                if not self.faiss_index.build(doc_embeddings, doc_ids):
                    print("Failed to build FAISS index for dense retriever.")
                    raise RuntimeError("Failed to build FAISS index for dense retriever.")
    
                self.bert_dense_retriever = True  # flag to signal setup complete
                print("--- BERT Dense Retriever Setup Finished ---")
                logging.info("--- BERT Dense Retriever Setup Finished ---")
            except Exception as e:
                logging.error(f"Error setting up BERT Dense Retriever: {e}", exc_info=True)
                print("Error setting up BERT Dense Retriever: {e}")
                self.bert_dense_retriever = None
                return False
        return True

    def run_bm25_rank(self):
        logging.info("--- Starting Pipeline: Basic BM25 (rank_bm25 Cache) ---")
        start_time = time.time()

        k1_str = str(config.BM25_K1).replace('.', 'p')
        b_str = str(config.BM25_B).replace('.', 'p')

        bm25_execution_name = f"BM25_k1_{k1_str}_b_{b_str}"
        bm25_execution_name += "_stop" if config.REMOVE_STOPWORDS else ""
        bm25_execution_name += "_stem" if config.ENABLE_STEMMING else ""
        bm25_execution_name += f"_{config.CONTENT_FIELD}"
        bm25_execution_name += "_demo" if config.IS_DEMO_MODE else ""

        # Setup BM25
        if not self.setup_bm25_rank():
            logging.error("BM25 setup failed. Aborting BM25 run.")
            return None, f"{bm25_execution_name}_FAILED"

        if self.bm25_rank_retriever is None or self.bm25_rank_retriever.bm25 is None:
            logging.error("BM25 retriever not ready after setup. Aborting search.")
            return None, f"{bm25_execution_name}_FAILED"

        run_results = {}
        num_queries = len(self.queries)
        processed_count = 0
        logging.info(f"Processing {num_queries} queries with BM25...")

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

        # Save the run results
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

   
    def run_bert_dense(self):
        logging.info("--- Starting Pipeline: BERT Dense Retrieval ---")
        print("--- Starting Pipeline: BERT Dense Retrieval ---")
        start_time = time.time()
    
        dense_execution_name = f"Dense_{config.DENSE_MODEL_NAME.split('/')[-1]}_{config.CONTENT_FIELD}"
        if config.IS_DEMO_MODE:
            dense_execution_name += "_demo"
    
        if not self.setup_bert_dense():
            logging.error("Dense retriever setup failed.")
            print("Dense retriever setup failed.")
            return None, f"{dense_execution_name}_FAILED"
    
        run_results = {}
        queries = list(self.queries.values())
        qids = list(self.queries.keys())
    
        query_embeddings = self.bert_embedder.embed(
            queries,
            batch_size=config.DENSE_BATCH_SIZE,
            preprocessor=self.preprocessor
        )
    
        distances, retrieved_doc_ids = self.faiss_index.search(query_embeddings, k=config.FINAL_TOP_K)
    
        for idx, qid in enumerate(qids):
            doc_score_pairs = {
                doc_id: float(dist) for doc_id, dist in zip(retrieved_doc_ids[idx], distances[idx]) if doc_id
            }
            run_results[qid] = dict(sorted(doc_score_pairs.items(), key=lambda item: item[1], reverse=True))
    
        run_file_name = f"run_{dense_execution_name}.txt"
        try:
            utils.save_run(run_results, config.OUTPUT_DIR, run_file_name, dense_execution_name)
            logging.info(f"Dense run saved: {run_file_name}")
            print(f"Dense run saved: {run_file_name}")
        except Exception as e:
            logging.error(f"Failed to save dense run: {e}")
            print(f"Failed to save dense run: {e}")
    
        logging.info(f"--- Finished BERT Dense Retrieval in {(time.time() - start_time):.2f}s ---")
        return run_results, dense_execution_name


    def run_hybrid_rerank(self):
        logging.info("--- Starting Pipeline: BM25 + BERT Re-rank ---")
        start_time = time.time()

        k1_str = str(config.BM25_K1).replace('.', 'p')
        b_str = str(config.BM25_B).replace('.', 'p')
        reranker_model_id = config.CROSS_ENCODER_MODEL_NAME.split('/')[-1]  # Get model name part
        re_rank_execution_name = (f"BM25_k1_{k1_str}_b_{b_str}"
                                  f"_top{config.BM25_TOP_K}_then_{reranker_model_id}_top{config.FINAL_TOP_K}")
        re_rank_execution_name += "_stop" if config.REMOVE_STOPWORDS else ""
        re_rank_execution_name += "_stem" if config.ENABLE_STEMMING else ""
        re_rank_execution_name += f"_{config.CONTENT_FIELD}"
        re_rank_execution_name += "_demo" if config.IS_DEMO_MODE else ""

        # --- Stage 1: Candidate Retrieval (BM25 Only) ---
        bm25_rank_candidates = {}
        logging.info("Setting up BM25 for candidate retrieval...")
        if not self.setup_bm25_rank():
            logging.error("BM25 setup failed. Aborting Hybrid (BM25+BERT) run.")
            return None, f"{re_rank_execution_name}_FAILED"

        if self.bm25_rank_retriever is None or self.bm25_rank_retriever.bm25 is None:
            logging.error("BM25 retriever not ready. Aborting Hybrid (BM25+BERT) run.")
            return None, f"{re_rank_execution_name}_FAILED"

        logging.info(f"Retrieving top {config.BM25_TOP_K} candidates using BM25...")
        num_queries = len(self.queries)
        retrieved_count = 0
        for qid, query_text in tqdm(self.queries.items(), desc="BM25 Candidates"):
            if not query_text:
                logging.warning(f"Skipping empty query QID: {qid} during candidate retrieval.")
                bm25_rank_candidates[qid] = {}
                continue
            try:
                bm25_rank_candidates[qid] = self.bm25_rank_retriever.search(query_text, k=config.BM25_TOP_K)
                retrieved_count += 1
            except Exception as e:
                logging.error(f"Error retrieving BM25 candidates for QID {qid}: {e}", exc_info=True)
                bm25_rank_candidates[qid] = {}

        logging.info(f"Retrieved BM25 candidates for {retrieved_count}/{num_queries} queries.")
        if not any(bm25_rank_candidates.values()):
            logging.error("No candidates retrieved from BM25. Cannot proceed with re-ranking.")
            # Return empty results if no candidates, but use the intended name
            return {}, re_rank_execution_name


        # --- Prepare Candidates for Re-ranking ---
        logging.info("Preparing candidate texts for re-ranking...")
        candidates_for_reranking = defaultdict(dict)
        total_candidates_prepared = 0
        text_retrieval_errors = 0

        for qid in tqdm(self.queries.keys(), desc="Gathering Candidate Texts"):
            candidate_doc_ids = bm25_rank_candidates.get(qid, {}).keys()
            if not candidate_doc_ids:
                continue

            docs_to_rerank = {}
            query_retrieval_errors = 0
            for doc_id in candidate_doc_ids:
                doc_text = self._get_document_text(doc_id)
                if doc_text:
                    docs_to_rerank[doc_id] = doc_text
                else:
                    query_retrieval_errors += 1
            if query_retrieval_errors > 0:
                logging.warning(f"Could not retrieve text for {query_retrieval_errors} candidates for QID {qid}.")
                text_retrieval_errors += query_retrieval_errors

            candidates_for_reranking[qid] = docs_to_rerank
            total_candidates_prepared += len(docs_to_rerank)

        logging.info(
            f"Prepared text for {total_candidates_prepared} total candidates across {len(candidates_for_reranking)} queries.")
        if text_retrieval_errors > 0:
            logging.warning(f"Total text retrieval errors for candidates: {text_retrieval_errors}")

        # --- Stage 2: BERT Re-ranking ---
        logging.info("Setting up BERT re-ranker...")
        if not self.setup_bert_reranker():
            logging.error("BERT re-ranker setup failed. Aborting Hybrid run.")
            return None, f"{re_rank_execution_name}_FAILED"
        if self.preprocessor is None: self.setup_preprocessing() # Ensure preprocessor exists if not setup before

        logging.info(f"Starting BERT re-ranking (Batch size: {config.RERANK_BATCH_SIZE})...")
        final_run_results = {}
        reranked_count = 0
        for qid, query_text in tqdm(self.queries.items(), desc=f"BERT Re-ranking ({reranker_model_id})"):
            docs_for_query = candidates_for_reranking.get(qid)

            if not query_text:
                logging.warning(f"Skipping re-ranking for empty query QID: {qid}")
                final_run_results[qid] = {}
                continue
            if not docs_for_query:
                # If no candidates were prepared mayb due to text retrieval errors or BM25 returning none
                logging.debug(f"Query QID {qid} has no prepared candidates to re-rank.")
                final_run_results[qid] = {}
                continue

            try:
                # Pass the preprocessor if needed by the reranker implementation
                reranked_scores = self.bert_reranker.rerank(
                    query_text,
                    docs_for_query,
                    preprocessor=self.preprocessor,
                    batch_size=config.RERANK_BATCH_SIZE
                )

                sorted_reranked = sorted(reranked_scores.items(), key=lambda item: float(item[1]), reverse=True)
                final_run_results[qid] = {doc_id: float(score) for doc_id, score in
                                          sorted_reranked[:config.FINAL_TOP_K]}
                reranked_count += 1
            except Exception as e:
                logging.error(f"Error re-ranking candidates for QID {qid}: {e}", exc_info=True)
                final_run_results[qid] = {}

        logging.info(f"Re-ranked candidates for {reranked_count}/{len(candidates_for_reranking)} queries with prepared candidates.") # Adjusted log message

        # --- Save Final Run Results ---
        run_file_name = f"run_{re_rank_execution_name}.txt"
        run_file_path = os.path.join(config.OUTPUT_DIR, run_file_name)
        try:
            utils.save_run(final_run_results, config.OUTPUT_DIR, run_file_name, re_rank_execution_name)
            logging.info(f"Hybrid (BM25+BERT) run results saved to {run_file_path}")
        except Exception as e:
            logging.error(f"Failed to save Hybrid run results: {e}")

        end_time = time.time()
        logging.info(f"--- Finished Pipeline: BM25 + BERT Re-rank ({(end_time - start_time):.2f} seconds) ---")
        return final_run_results, re_rank_execution_name

    def run_evaluation(self, run_results, execution_name):
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