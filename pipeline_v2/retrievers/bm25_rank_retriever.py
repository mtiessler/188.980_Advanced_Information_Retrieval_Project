import logging
import time
import os
import gc
import pickle
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm
import config
class BM25RankRetriever:
    def __init__(self, preprocessor, k1=1.5, b=0.75):
        self.preprocessor = preprocessor
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.corpus_ids = None
        demo_suffix = "_demo" if config.IS_DEMO_MODE else "" # demo cache :)
        limit_suffix = f"_limit{getattr(config, 'DEMO_FILES_LIMIT', 1)}" if config.IS_DEMO_MODE else ""
        base_id = os.path.basename(config.BASE_DIR)
        self.token_cache_file = os.path.join(config.CACHE_DIR, f"bm25_tokens_{base_id}{demo_suffix}{limit_suffix}.pkl")
        logging.info(f"Initialized BM25RankRetriever (k1={k1}, b={b})")
        logging.info(f"Using token cache file: {self.token_cache_file}")


    def _tokenize_and_cache_chunks(self, doc_iterator, chunk_size, cache_file):
        logging.info(f"Tokenizing documents in chunks of {chunk_size} and caching to {cache_file}...")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        processed_doc_ids = []
        chunk_tokens = []
        chunk_ids = []
        docs_in_chunk = 0
        total_docs_processed = 0

        with open(cache_file, 'wb') as f_cache:
            for doc_dict in tqdm(doc_iterator, desc="Tokenizing & Caching Chunks"):
                if doc_dict and 'id' in doc_dict and 'text' in doc_dict and doc_dict['text']:
                    doc_id = doc_dict['id']
                    text = doc_dict['text']
                    tokens = self.preprocessor.tokenize_and_process(text)
                    chunk_tokens.append(tokens)
                    chunk_ids.append(doc_id)
                    docs_in_chunk += 1
                    total_docs_processed += 1

                    if docs_in_chunk >= chunk_size:
                        pickle.dump(chunk_tokens, f_cache)
                        processed_doc_ids.extend(chunk_ids)
                        logging.debug(f"Wrote chunk of {len(chunk_tokens)} token lists to cache.")
                        chunk_tokens = []
                        chunk_ids = []
                        docs_in_chunk = 0
                        gc.collect()

                else:
                    logging.debug(f"Skipping invalid doc_dict during chunk processing: {doc_dict}")

            if chunk_tokens:
                pickle.dump(chunk_tokens, f_cache)
                processed_doc_ids.extend(chunk_ids)
                logging.debug(f"Wrote final chunk of {len(chunk_tokens)} token lists to cache.")

        logging.info(f"Finished tokenizing and caching {total_docs_processed} documents.")
        if not processed_doc_ids:
             logging.error("No documents were tokenized and cached.")
             if os.path.exists(cache_file):
                  try: os.remove(cache_file)
                  except OSError: pass
             return None, None

        return processed_doc_ids, cache_file

    def _load_tokens_from_cache(self, cache_file):
        logging.info(f"Loading all tokens from cache file: {cache_file}")
        all_tokens = []
        try:
            with open(cache_file, 'rb') as f_cache:
                while True:
                    try:
                        chunk = pickle.load(f_cache)
                        all_tokens.extend(chunk)
                    except EOFError:
                        break
            logging.info(f"Loaded {len(all_tokens)} token lists from cache.")
            return all_tokens
        except Exception as e:
            logging.error(f"Error loading tokens from cache {cache_file}: {e}", exc_info=True)
            return None

    def index(self, doc_iterator_factory, chunk_size, cache_file, force_retokenize=False):
        logging.info("Starting BM25 (rank_bm25) indexing...")
        if self.bm25 is not None and not force_retokenize:
             logging.warning("Index already exists. Set force_retokenize=True to re-index.")
             return True

        if force_retokenize and os.path.exists(cache_file):
             logging.info(f"force_retokenize=True, removing existing cache file: {cache_file}")
             try:
                  os.remove(cache_file)
             except OSError as e:
                  logging.error(f"Error removing existing cache file: {e}")

        cached_tokens_exist = os.path.exists(cache_file)
        processed_doc_ids = None
        used_cache_file = cache_file

        if not cached_tokens_exist:
             logging.info("No token cache found. Tokenizing and caching documents...")
             doc_iterator_for_tokenization = doc_iterator_factory()
             processed_doc_ids, used_cache_file = self._tokenize_and_cache_chunks(
                 doc_iterator_for_tokenization, chunk_size, cache_file
             )
             if processed_doc_ids is None:
                  return False
             del doc_iterator_for_tokenization
             gc.collect()
        else:
             logging.info("Token cache file found. Re-gathering IDs to match cache order...")
             # Get a fresh iterator from the factory for ID gathering
             id_iterator = doc_iterator_factory()
             processed_doc_ids = []
             for doc_dict in tqdm(id_iterator, desc="Re-gathering IDs for cache"):
                 if doc_dict and 'id' in doc_dict:
                      processed_doc_ids.append(doc_dict['id'])
             logging.info(f"Re-gathered {len(processed_doc_ids)} IDs to match existing token cache.")
             del id_iterator
             gc.collect()

        if not processed_doc_ids:
             logging.error("No document IDs were gathered. Cannot proceed.")
             if cached_tokens_exist and os.path.exists(used_cache_file):
                  logging.warning(f"Deleting potentially mismatched token cache: {used_cache_file}")
                  try: os.remove(used_cache_file)
                  except OSError: pass
             return False

        logging.info("Loading tokens into memory for BM25Okapi...")
        corpus_tokens = self._load_tokens_from_cache(used_cache_file)

        if corpus_tokens is None:
             logging.error("Failed to load tokens from cache.")
             return False
        if not corpus_tokens:
             logging.error("Token cache was loaded but resulted in an empty list.")
             return False
        if not any(corpus_tokens):
             logging.error("Loaded tokens, but all documents resulted in empty token lists after preprocessing. Cannot build index.")
             if os.path.exists(used_cache_file):
                 logging.warning(f"Deleting token cache with only empty lists: {used_cache_file}")
                 try: os.remove(used_cache_file)
                 except OSError: pass
             return False

        if len(corpus_tokens) != len(processed_doc_ids):
             logging.error(f"CRITICAL MISMATCH between loaded tokens ({len(corpus_tokens)}) and gathered IDs ({len(processed_doc_ids)}). Deleting potentially corrupt cache: {used_cache_file}")
             if os.path.exists(used_cache_file):
                  try: os.remove(used_cache_file)
                  except OSError: pass
             return False

        logging.info("Initializing BM25Okapi index...")
        start_time = time.time()
        try:
             self.bm25 = BM25Okapi(corpus_tokens, k1=self.k1, b=self.b)
             self.corpus_ids = processed_doc_ids
             end_time = time.time()
             logging.info(f"BM25Okapi index built successfully in {end_time - start_time:.2f} seconds.")
             del corpus_tokens
             gc.collect()
             return True
        except MemoryError:
             logging.error("MemoryError occurred during BM25Okapi initialization. Token list likely too large for RAM.", exc_info=True)
             self.bm25 = None
             self.corpus_ids = None
             if 'corpus_tokens' in locals(): del corpus_tokens
             gc.collect()
             return False
        except Exception as e:
             logging.error(f"Error during BM25Okapi initialization: {e}", exc_info=True)
             self.bm25 = None
             self.corpus_ids = None
             if 'corpus_tokens' in locals(): del corpus_tokens
             gc.collect()
             return False


    def search(self, query_text, k):
        if self.bm25 is None or self.corpus_ids is None:
            logging.error("BM25 (rank_bm25) index not built. Cannot search.")
            return {}

        try:
            query_tokens = self.preprocessor.tokenize_and_process(query_text)
            if not query_tokens:
                logging.warning(f"Query '{query_text[:50]}...' resulted in no tokens after preprocessing.")
                return {}

            logging.debug(f"Searching rank_bm25 for query tokens: {query_tokens[:10]}...")
            doc_scores = self.bm25.get_scores(query_tokens)

            results = sorted(zip(self.corpus_ids, doc_scores), key=lambda x: x[1], reverse=True)

            top_k_pairs = [(doc_id, score) for doc_id, score in results if score > 0][:k]
            top_k_results = dict(top_k_pairs)

            logging.debug(f"rank_bm25 search returned {len(top_k_results)} results.")
            return top_k_results

        except Exception as e:
            logging.error(f"Error during rank_bm25 search for query '{query_text[:50]}...': {e}", exc_info=True)
            return {}

    def load_or_build_index(self, doc_iterator_factory, chunk_size, cache_file):
         logging.info("Attempting to build/load rank_bm25 index...")
         return self.index(doc_iterator_factory, chunk_size, cache_file, force_retokenize=False)
