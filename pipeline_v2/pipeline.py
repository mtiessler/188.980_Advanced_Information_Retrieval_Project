import os
import logging
import time
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
import gc

import config
import data_loader
import utils
from evaluation import evaluate_run
from preprocessing import Preprocessor
from retrievers.bm25_rank_retriever import BM25RankRetriever # Use rank_bm25 retriever
from retrievers.dense_retriever import BertEmbedder, FaissIndex
from rerankers import BertReranker

class RetrievalPipeline:
    def __init__(self):
        self.documents = {}
        self.doc_ids_in_order = []
        self.queries = {}
        self.qrels = []
        self.preprocessor = None

        self.bm25_rank_retriever = None # Use rank_bm25 retriever
        self.bert_embedder = None
        self.faiss_index = None
        self.bert_reranker = None

        self.doc_embeddings = None
        self.query_embeddings_dict = {}

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.CACHE_DIR, exist_ok=True)

    def load_data(self):
        logging.info("--- Loading Data ---")
        self.documents, self.doc_ids_in_order = data_loader.load_documents_structure(
            config.DOCUMENTS_DIR,
            content_field=config.CONTENT_FIELD
        )
        self.queries = data_loader.load_queries(config.QUERIES_FILE)
        self.qrels = data_loader.load_qrels(config.QRELS_FILE)

        if not self.documents or not self.queries:
             raise RuntimeError("Failed to load essential data (document structures or queries). Check paths and file formats.")
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
        main_content_raw = doc_data.get(config.CONTENT_FIELD) or ""

        abstract = ""
        if abstract_raw and isinstance(abstract_raw, str) and not abstract_raw.startswith(("http://", "https://")) and len(abstract_raw) > 10:
             abstract = abstract_raw

        main_content = ""
        if main_content_raw and isinstance(main_content_raw, str) and not main_content_raw.startswith(("http://", "https://")) and len(main_content_raw) > 10:
             main_content = main_content_raw

        if config.CONTENT_FIELD == 'abstract':
             combined_text = f"{title} {abstract}".strip()
        else:
             if main_content == abstract:
                  combined_text = f"{title} {abstract}".strip()
             else:
                  combined_text = f"{title} {abstract} {main_content}".strip()

        return combined_text if combined_text else ""


    def setup_bm25_rank(self):
        if self.bm25_rank_retriever is None:
            logging.info("--- Setting up BM25 (rank_bm25 with Caching) ---")
            if self.preprocessor is None: self.setup_preprocessing()

            self.bm25_rank_retriever = BM25RankRetriever(
                preprocessor=self.preprocessor,
                k1=config.BM25_K1,
                b=config.BM25_B
            )

            def _doc_iterator_factory():
                 return data_loader.stream_document_dicts(
                     config.DOCUMENTS_DIR,
                     content_field=config.CONTENT_FIELD
                 )

            # Pass the factory function to the load/build method
            index_built = self.bm25_rank_retriever.load_or_build_index(
                 _doc_iterator_factory, # Pass the function itself
                 chunk_size=config.BM25_TOKEN_CHUNK_SIZE,
                 cache_file=self.bm25_rank_retriever.token_cache_file
            )

            if not index_built or self.bm25_rank_retriever.bm25 is None:
                 logging.error("BM25 (rank_bm25) index building failed.")
                 self.bm25_rank_retriever = None
                 return False

            logging.info("--- BM25 (rank_bm25) Setup Finished ---")
        return True

    def setup_bert_embedder(self):
        if self.bert_embedder is None:
            logging.info("--- Setting up BERT Embedder ---")
            self.bert_embedder = BertEmbedder(
                model_name=config.EMBEDDING_MODEL_NAME,
                device=config.DEVICE
            )
            logging.info("--- BERT Embedder Setup Finished ---")
        return True

    def setup_faiss(self):
        if self.faiss_index is None:
            logging.info("--- Setting up FAISS ---")
            if self.doc_embeddings is None:
                logging.error("Document embeddings not generated. Cannot setup FAISS.")
                return False
            if self.doc_embeddings.shape[0] == 0:
                 logging.error("Document embeddings array is empty. Cannot setup FAISS.")
                 return False
            if len(self.doc_ids_in_order) != self.doc_embeddings.shape[0]:
                 logging.error(f"Mismatch between ordered doc IDs ({len(self.doc_ids_in_order)}) and embeddings count ({self.doc_embeddings.shape[0]}). Cannot setup FAISS.")
                 return False

            dimension = self.doc_embeddings.shape[1]
            self.faiss_index = FaissIndex(dimension)

            dataset_id = os.path.basename(config.BASE_DIR)
            model_id = config.EMBEDDING_MODEL_NAME.replace('/', '_')
            demo_suffix = "_demo" if config.IS_DEMO_MODE else ""
            limit_suffix = f"_limit{getattr(config, 'DEMO_FILES_LIMIT', 1)}" if config.IS_DEMO_MODE else ""
            index_path = os.path.join(config.CACHE_DIR, f"faiss_{dataset_id}{demo_suffix}{limit_suffix}_{model_id}.index")
            doc_id_map_path = os.path.join(config.CACHE_DIR, f"faiss_doc_ids_{dataset_id}{demo_suffix}{limit_suffix}_{model_id}.txt")


            loaded_from_cache = False
            # Use force_regenerate flag from config to control FAISS cache too
            if not config.FORCE_REGENERATE_EMBEDDINGS and os.path.exists(index_path) and os.path.exists(doc_id_map_path):
                 logging.info("Attempting to load cached FAISS index...")
                 if self.faiss_index.load(index_path, doc_id_map_path):
                     if self.faiss_index.index.ntotal == len(self.doc_ids_in_order) and self.faiss_index.doc_ids == self.doc_ids_in_order:
                         logging.info("Successfully loaded cached FAISS index matching current documents.")
                         loaded_from_cache = True
                     else:
                          logging.warning("Cached FAISS index document IDs or count mismatch current data. Will rebuild.")
                          self.faiss_index = FaissIndex(dimension)
                 else:
                      logging.warning("Failed to load cached FAISS index files. Will rebuild.")
            else:
                 if config.FORCE_REGENERATE_EMBEDDINGS:
                      logging.info("FORCE_REGENERATE_EMBEDDINGS is True, rebuilding FAISS index.")
                 else:
                      logging.info("FAISS cache files not found.")


            if not loaded_from_cache:
                 logging.info("Building new FAISS index.")
                 build_success = self.faiss_index.build(self.doc_embeddings, self.doc_ids_in_order)
                 if build_success:
                     self.faiss_index.save(index_path, doc_id_map_path)
                 else:
                     logging.error("FAISS index building failed.")
                     self.faiss_index = None
                     return False

            logging.info("--- FAISS Setup Finished ---")
        return True


    def setup_bert_reranker(self):
         if self.bert_reranker is None:
              logging.info("--- Setting up BERT Re-ranker ---")
              if self.preprocessor is None: self.setup_preprocessing()
              self.bert_reranker = BertReranker(
                   model_name=config.CROSS_ENCODER_MODEL_NAME,
                   device=config.DEVICE
              )
              logging.info("--- BERT Re-ranker Setup Finished ---")
         return True


    def generate_doc_embeddings(self, force_regenerate=config.FORCE_REGENERATE_EMBEDDINGS):
        dataset_id = os.path.basename(config.BASE_DIR)
        model_id = config.EMBEDDING_MODEL_NAME.replace('/', '_')
        demo_suffix = "_demo" if config.IS_DEMO_MODE else ""
        limit_suffix = f"_limit{getattr(config, 'DEMO_FILES_LIMIT', 1)}" if config.IS_DEMO_MODE else ""
        embeddings_path = os.path.join(config.CACHE_DIR, f"doc_embeddings_{dataset_id}{demo_suffix}{limit_suffix}_{model_id}.npy")
        doc_ids_path = os.path.join(config.CACHE_DIR, f"doc_embeddings_ids_{dataset_id}{demo_suffix}{limit_suffix}_{model_id}.txt")


        if self.doc_embeddings is not None and not force_regenerate:
             logging.info("Document embeddings already available in memory.")
             return True

        logging.info("--- Generating/Loading Document Embeddings ---")

        if not force_regenerate and os.path.exists(embeddings_path) and os.path.exists(doc_ids_path):
            logging.info(f"Attempting to load cached document embeddings from {embeddings_path}")
            try:
                loaded_embeddings = np.load(embeddings_path)
                with open(doc_ids_path, 'r', encoding='utf-8') as f:
                    loaded_ids = [line.strip() for line in f]

                # IMPORTANT: Check loaded IDs against the *current* self.doc_ids_in_order
                if len(loaded_ids) == len(self.doc_ids_in_order) and loaded_ids == self.doc_ids_in_order:
                    self.doc_embeddings = loaded_embeddings
                    logging.info(f"Successfully loaded {len(self.doc_embeddings)} cached document embeddings.")
                    logging.info("--- Document Embeddings Ready ---")
                    return True
                else:
                    logging.warning(f"Cached document IDs/count ({len(loaded_ids)}) do not match current expected IDs/count ({len(self.doc_ids_in_order)}). Regenerating embeddings.")
            except Exception as e:
                logging.warning(f"Could not load cached embeddings ({e}). Regenerating.")

        logging.info("Generating document embeddings...")
        if not self.setup_bert_embedder(): return False
        if self.preprocessor is None: self.setup_preprocessing()

        if not self.doc_ids_in_order:
             logging.error("doc_ids_in_order is empty. Cannot generate document embeddings.")
             return False

        logging.info(f"Preparing text for {len(self.doc_ids_in_order)} documents for embedding...")
        doc_texts_for_embedding = []
        retrieval_errors = 0
        for doc_id in tqdm(self.doc_ids_in_order, desc="Retrieving doc texts"):
             text = self._get_document_text(doc_id)
             doc_texts_for_embedding.append(text if text else "")
             if not text:
                  retrieval_errors += 1

        if retrieval_errors > 0:
             logging.warning(f"Could not retrieve valid text for {retrieval_errors} documents during embedding preparation (used empty string).")

        logging.info("Starting embedding generation...")
        self.doc_embeddings = self.bert_embedder.embed(
            doc_texts_for_embedding,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            preprocessor=self.preprocessor
        )

        if self.doc_embeddings is not None and self.doc_embeddings.shape[0] > 0:
            if self.doc_embeddings.shape[0] != len(self.doc_ids_in_order):
                 logging.error(f"Embedding generation mismatch: Expected {len(self.doc_ids_in_order)} embeddings, got {self.doc_embeddings.shape[0]}.")
                 del doc_texts_for_embedding
                 gc.collect()
                 return False

            logging.info(f"Generated {self.doc_embeddings.shape[0]} embeddings with dimension {self.doc_embeddings.shape[1]}.")
            try:
                # Save using the potentially mode-specific path
                np.save(embeddings_path, self.doc_embeddings)
                with open(doc_ids_path, 'w', encoding='utf-8') as f:
                    for doc_id in self.doc_ids_in_order:
                        f.write(f"{doc_id}\n")
                logging.info(f"Document embeddings saved to {embeddings_path}")
                logging.info("--- Document Embeddings Ready ---")
            except Exception as e:
                 logging.error(f"Failed to save document embeddings or IDs: {e}")

            del doc_texts_for_embedding
            gc.collect()
            return True
        else:
            logging.error("Failed to generate document embeddings (result is None or empty).")
            self.doc_embeddings = None
            if 'doc_texts_for_embedding' in locals(): del doc_texts_for_embedding
            gc.collect()
            return False

    def generate_query_embeddings(self):
        if self.query_embeddings_dict:
             logging.info("Query embeddings already generated.")
             return True

        logging.info("--- Generating Query Embeddings ---")
        if not self.setup_bert_embedder(): return False
        if self.preprocessor is None: self.setup_preprocessing()

        query_ids = list(self.queries.keys())
        query_texts = list(self.queries.values())
        if not query_texts:
             logging.error("No query texts found to embed.")
             return False

        query_embeddings_array = self.bert_embedder.embed(
            query_texts,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            preprocessor=self.preprocessor,
            show_progress=True
        )

        if query_embeddings_array is not None and query_embeddings_array.shape[0] == len(query_ids):
            self.query_embeddings_dict = {qid: emb for qid, emb in zip(query_ids, query_embeddings_array)}
            logging.info(f"Generated {len(self.query_embeddings_dict)} query embeddings.")
            logging.info("--- Query Embeddings Ready ---")
            return True
        else:
            logging.error("Failed to generate query embeddings or mismatch in count.")
            return False

    def run_bm25_rank(self):
        logging.info("--- Starting Pipeline: Basic BM25 (rank_bm25 Cache) ---")
        start_time = time.time()
        if not self.setup_bm25_rank():
             logging.error("BM25 (rank_bm25) setup failed. Aborting run.")
             return None, "BM25Rank_Baseline_Failed"

        run_results = {}
        system_name = "BM25Rank_Baseline"

        if self.bm25_rank_retriever is None or self.bm25_rank_retriever.bm25 is None:
             logging.error("BM25 (rank_bm25) retriever not ready after setup. Aborting search loop.")
             return None, system_name

        for qid, query_text in tqdm(self.queries.items(), desc="BM25 (rank_bm25) Processing Queries"):
            run_results[qid] = self.bm25_rank_retriever.search(query_text, k=config.FINAL_TOP_K)

        utils.save_run(run_results, config.OUTPUT_DIR, f"run_{system_name}.txt", system_name)
        end_time = time.time()
        logging.info(f"--- Finished Pipeline: Basic BM25 (rank_bm25 Cache) ({end_time - start_time:.2f} seconds) ---")
        return run_results, system_name

    def run_bert_dense(self):
        logging.info("--- Starting Pipeline: BERT Dense Retrieval (FAISS) ---")
        start_time = time.time()
        system_name = "BERT_Dense_FAISS"

        if not self.generate_doc_embeddings(): return None, system_name
        if not self.generate_query_embeddings(): return None, system_name
        if not self.setup_faiss(): return None, system_name

        run_results = {}
        query_ids_list = list(self.queries.keys())
        query_embeddings_array = np.array([self.query_embeddings_dict[qid] for qid in query_ids_list])

        logging.info("Performing FAISS search for all queries...")
        distances, results_doc_ids_nested = self.faiss_index.search(
            query_embeddings_array,
            k=config.FINAL_TOP_K
        )
        logging.info("FAISS search completed.")

        for i, qid in enumerate(tqdm(query_ids_list, desc="Formatting Dense Results")):
            query_results = {}
            if i < len(results_doc_ids_nested) and i < len(distances):
                 for j, doc_id in enumerate(results_doc_ids_nested[i]):
                      if doc_id is not None and j < len(distances[i]):
                           score = distances[i][j]
                           query_results[doc_id] = float(score)
            else:
                 logging.warning(f"Missing results for query index {i} (qid: {qid}) in FAISS output.")
            run_results[qid] = query_results

        utils.save_run(run_results, config.OUTPUT_DIR, f"run_{system_name}.txt", system_name)
        end_time = time.time()
        logging.info(f"--- Finished Pipeline: BERT Dense Retrieval ({end_time - start_time:.2f} seconds) ---")
        return run_results, system_name


    def run_hybrid_rerank(self):
        logging.info("--- Starting Pipeline: Hybrid Re-rank (BM25Rank/FAISS + BERT) ---")
        start_time = time.time()
        system_name = "Hybrid_ReRank_BM25Rank"

        bm25_rank_candidates = {}
        if self.setup_bm25_rank():
            logging.info("Retrieving initial candidates using BM25 (rank_bm25)...")
            if self.bm25_rank_retriever is None or self.bm25_rank_retriever.bm25 is None:
                  logging.error("BM25 (rank_bm25) retriever not ready for candidate generation.")
            else:
                 for qid, query_text in tqdm(self.queries.items(), desc="BM25 (rank_bm25) Candidates"):
                      bm25_rank_candidates[qid] = self.bm25_rank_retriever.search(query_text, k=config.BM25_TOP_K)
                 logging.info(f"Retrieved BM25 (rank_bm25) candidates for {len(bm25_rank_candidates)} queries.")
        else:
             logging.error("BM25 (rank_bm25) setup failed, cannot get candidates for hybrid model.")

        faiss_candidates = {}
        if self.generate_doc_embeddings() and self.generate_query_embeddings() and self.setup_faiss():
            query_ids_list = list(self.queries.keys())
            query_embeddings_array = np.array([self.query_embeddings_dict[qid] for qid in query_ids_list])
            distances, results_doc_ids_nested = self.faiss_index.search(
                 query_embeddings_array,
                 k=config.FAISS_TOP_K
            )
            for i, qid in enumerate(query_ids_list):
                 query_results = {}
                 if i < len(results_doc_ids_nested) and i < len(distances):
                      for j, doc_id in enumerate(results_doc_ids_nested[i]):
                           if doc_id is not None and j < len(distances[i]):
                               score = distances[i][j]
                               query_results[doc_id] = float(score)
                 faiss_candidates[qid] = query_results
            logging.info(f"Retrieved FAISS candidates for {len(faiss_candidates)} queries.")
        else:
             logging.error("Dense retrieval setup failed, cannot get FAISS candidates for hybrid model.")
             if not bm25_rank_candidates:
                  logging.error("Both BM25 and FAISS candidate generation failed. Aborting Hybrid run.")
                  return None, system_name

        logging.info("Combining candidates...")
        combined_candidates_text = defaultdict(dict)
        for qid in self.queries:
            candidate_doc_ids = set()
            if qid in bm25_rank_candidates: candidate_doc_ids.update(bm25_rank_candidates[qid].keys())
            if qid in faiss_candidates: candidate_doc_ids.update(faiss_candidates[qid].keys())

            if not candidate_doc_ids: continue

            docs_to_rerank = {}
            retrieval_errors = 0
            for doc_id in candidate_doc_ids:
                 doc_text = self._get_document_text(doc_id)
                 if doc_text:
                     docs_to_rerank[doc_id] = doc_text
                 else:
                      retrieval_errors += 1
                      logging.debug(f"Could not find text for candidate document {doc_id} for query {qid}.")
            if retrieval_errors > 0:
                 logging.warning(f"Could not retrieve text for {retrieval_errors} candidate documents for query {qid}.")
            combined_candidates_text[qid] = docs_to_rerank
        logging.info("Candidates combined.")

        logging.info("Starting BERT re-ranking...")
        if not self.setup_bert_reranker(): return None, system_name
        if self.preprocessor is None: self.setup_preprocessing()

        final_run_results = {}
        for qid, query_text in tqdm(self.queries.items(), desc="BERT Re-ranking"):
             docs_for_query = combined_candidates_text.get(qid, {})
             if not docs_for_query:
                  final_run_results[qid] = {}
                  continue
             reranked_scores = self.bert_reranker.rerank(
                 query_text,
                 docs_for_query,
                 preprocessor=self.preprocessor,
                 batch_size=config.RERANK_BATCH_SIZE
             )
             sorted_reranked = sorted(reranked_scores.items(), key=lambda item: item[1], reverse=True)
             final_run_results[qid] = {doc_id: score for doc_id, score in sorted_reranked[:config.FINAL_TOP_K]}

        utils.save_run(final_run_results, config.OUTPUT_DIR, f"run_{system_name}.txt", system_name)
        end_time = time.time()
        logging.info(f"--- Finished Pipeline: Hybrid Re-rank ({end_time - start_time:.2f} seconds) ---")
        return final_run_results, system_name


    def run_evaluation(self, run_results, system_name):
        if run_results is None:
             logging.warning(f"Skipping evaluation for {system_name} because run results are None.")
             return
        logging.info(f"--- Evaluating System: {system_name} ---")
        evaluate_run(
            run_results,
            self.qrels,
            config.EVALUATION_MEASURE,
            system_name
        )
        logging.info(f"--- Evaluation Finished for {system_name} ---")