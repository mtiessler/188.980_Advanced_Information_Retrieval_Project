import logging
import time
import os
from collections import defaultdict
from tqdm.auto import tqdm
import gc

from pipeline_v2 import config, data_loader, utils
from pipeline_v2.bm25f_prototype.pyterrier_rank_retriever import PyTerrierRankRetriever
from pipeline_v2.evaluation import evaluate_run
from pipeline_v2.preprocessing import Preprocessor
from pipeline_v2.rerankers import BertReranker

"""
This file contains necessary changes to the data loader in order to allow usage of BM25F
It might contain (several) bugs, since it was not fully run due to the environment issues caused by PyTerrier requiring 
access to a JDK, that made us decide to stop further experimentation considering the time constraints.
"""

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
        if self.bm25_rank_retriever is not None:
            return True  # already initialised

        logging.info("--- Setting up BM25 / PyTerrier ---")
        if self.preprocessor is None:
            self.setup_preprocessing()

        # 1)  build the retriever
        self.bm25_rank_retriever = PyTerrierRankRetriever(
            preprocessor=self.preprocessor,
            index_path="./terrier_idx",
            field_weights={  # BM25F weights
                "title": 2.0,
                "abstract": 1.5,
                "main_content": 1.0,
            },
            num_results=config.BM25_TOP_K  # optional, default 1000
        )

        # 2) factory that streams docs with the new loader
        def _doc_iterator_factory():
            limit = config.DEMO_FILES_LIMIT if config.IS_DEMO_MODE else None
            logging.info(f"BM25 Index: streaming docs (limit={limit})")
            return data_loader.stream_document_dicts_fields(
                config.DOCUMENTS_DIR,
                content_field=config.CONTENT_FIELD,
                demo_limit=limit,
            )

        # 3) build or load the index
        ok = self.bm25_rank_retriever.load_or_build_index(
            doc_iterator_factory=_doc_iterator_factory,
            force_reindex=False,  # flip to True to rebuild
        )
        if not ok:
            logging.error("BM25 index build failed.")
            self.bm25_rank_retriever = None
            return False

        logging.info("--- BM25 / PyTerrier ready ---")
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
            logging.info(f"--- Setting up BERT Dense Retriever ---")
            # Placeholder: Load or initialize the dense retriever model)
            # Placeholder: Load pre-built index if available, or prepare for building
            logging.info("--- BERT Dense Retriever Setup (Placeholder) Finished ---")
            pass # Implement setup logic
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

    # Placeholder for Task 2: Representation Learning (e.g., MS-MARCO based dense retrieval)
    def run_bert_dense(self):
        logging.info("--- Starting Pipeline: Representation Learning (BERT Dense) ---")
        start_time = time.time()

        # Placeholder: Define execution name based on model and parameters
        # Placeholder: dense_model_id = config.DENSE_MODEL_NAME.split('/')[-1]
        dense_execution_name = f"BERT_Dense_Placeholder" # Replace with actual naming logic
        dense_execution_name += f"_{config.CONTENT_FIELD}"
        dense_execution_name += "_demo" if config.IS_DEMO_MODE else ""

        # Placeholder: Setup the dense retriever
        if not self.setup_bert_dense():
             logging.error("BERT Dense setup failed. Aborting Dense run.")
             return None, f"{dense_execution_name}_FAILED"

        # Placeholder: Check if retriever is ready
        # if self.bert_dense_retriever is None: # or index not ready
        #     logging.error("BERT Dense retriever not ready. Aborting search.")
        #     return None, f"{dense_execution_name}_FAILED"

        run_results = {}
        num_queries = len(self.queries)
        processed_count = 0
        logging.info(f"Processing {num_queries} queries with BERT Dense (Placeholder)...")

        # Placeholder: Load MS-MARCO model (or specify path in config)
        # model = AutoModel.from_pretrained(config.DENSE_MODEL_NAME)
        # tokenizer = AutoTokenizer.from_pretrained(config.DENSE_MODEL_NAME)
        logging.info("Placeholder: Load MS-MARCO model")

        # Placeholder: Train/Fine-tune the model
        # Requires training data, loss function, optimizer, training loop
        logging.info("Placeholder: Train/Fine-tune the model")
        # train_model(model, train_data, ...)

        # Placeholder: Evaluate the trained model
        # Requires evaluation data, metrics
        logging.info("Placeholder: Evaluate the trained model")
        # evaluate_model(model, eval_data, ...)

        # Placeholder: Push model to Hugging Face Hub
        # Requires HF credentials, repository setup
        logging.info("Placeholder: Push model to Hugging Face Hub")
        # model.push_to_hub("your-hf-username/your-model-name")
        # tokenizer.push_to_hub("your-hf-username/your-model-name")

        # Placeholder: Perform Inference (Dense Retrieval)
        logging.info("Placeholder: Perform Inference (Dense Retrieval)")
        # Placeholder: Build or load document index
        # self.bert_dense_retriever.build_index(self.documents) or load_index(...)
        for qid, query_text in tqdm(self.queries.items(), desc=f"BERT Dense Search ({dense_execution_name})"):
             if not query_text:
                 logging.warning(f"Skipping empty query for QID: {qid}")
                 run_results[qid] = {}
                 continue
             try:
                 # Placeholder: Actual search call
                 # query_run_results = self.bert_dense_retriever.search(query_text, k=config.FINAL_TOP_K)
                 query_run_results = {} # Replace with actual results
                 run_results[qid] = query_run_results
                 processed_count += 1
             except Exception as e:
                 logging.error(f"Error processing query QID {qid} with BERT Dense: {e}", exc_info=True)
                 run_results[qid] = {}

        logging.info(f"Processed {processed_count}/{num_queries} queries (Placeholder).")

        # Save the run results
        run_file_name = f"run_{dense_execution_name}.txt"
        run_file_path = os.path.join(config.OUTPUT_DIR, run_file_name)
        try:
            #Placeholder: dummy results for placeholder
            utils.save_run(run_results, config.OUTPUT_DIR, run_file_name, dense_execution_name)
            logging.info(f"BERT Dense run results saved to {run_file_path} (Placeholder)")
        except Exception as e:
            logging.error(f"Failed to save BERT Dense run results: {e}")


        end_time = time.time()
        logging.info(f"--- Finished Pipeline: Representation Learning (BERT Dense) ({(end_time - start_time):.2f} seconds) ---")
        #Placeholder: Return placeholder results and name
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