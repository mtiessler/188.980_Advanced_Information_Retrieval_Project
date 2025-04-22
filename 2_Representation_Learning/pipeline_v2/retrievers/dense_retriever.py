import logging
import time
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from tqdm.auto import tqdm

class BertEmbedder:
    def __init__(self, model_name, device):
        logging.info(f"Loading embedding model: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval() # Set to evaluation mode
        self.device = device
        logging.info("Embedding model loaded.")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad() # Disable gradient calculation for inference
    def embed(self, texts, batch_size, preprocessor=None, show_progress=True):
        if not texts:
            logging.warning("Embed called with empty text list.")
            return np.array([])

        logging.info(f"Generating embeddings for {len(texts)} texts...")
        processed_texts = texts
        if preprocessor:
             logging.debug("Preprocessing texts for BERT embedding...")
             processed_texts = [preprocessor.preprocess_for_bert(text) for text in texts]


        all_embeddings = []
        iterator = range(0, len(processed_texts), batch_size)
        if show_progress:
             iterator = tqdm(iterator, desc="Embedding Batches")

        for i in iterator:
            batch_texts = processed_texts[i:i+batch_size]
            if not batch_texts: continue # Skip empty batches

            try:
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                ).to(self.device)

                model_output = self.model(**encoded_input)
                sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])

                # Normalize embeddings (often improves performance for cosine similarity)
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

                all_embeddings.append(sentence_embeddings.cpu().numpy())
            except Exception as e:
                 logging.error(f"Error embedding batch starting at index {i}: {e}")
                 # Decide how to handle errors: skip batch, add zeros, etc.
                 continue

        logging.info("Embeddings generation finished.")
        if not all_embeddings:
            return np.array([])
        return np.vstack(all_embeddings)


class FaissIndex:
    def __init__(self, dimension):
        logging.info(f"Initializing FAISS index with dimension {dimension}.")
        # Using IndexFlatIP (Inner Product) since embeddings are normalized (equivalent to cosine similarity)
        self.index = faiss.IndexFlatIP(dimension)
        self.doc_ids = [] # To map FAISS index positions back to original doc IDs
        logging.info("FAISS index initialized.")

    def build(self, embeddings, doc_ids):
        if embeddings.shape[0] == 0:
            logging.error("Cannot build FAISS index: Embeddings array is empty.")
            return False
        if embeddings.shape[0] != len(doc_ids):
             logging.error(f"Mismatch between embeddings count ({embeddings.shape[0]}) and doc_ids count ({len(doc_ids)}). Cannot build index.")
             return False

        logging.info(f"Building FAISS index with {embeddings.shape[0]} vectors...")
        start_time = time.time()
        # FAISS expects float32
        embeddings_fp32 = embeddings.astype('float32')
        # Ensure embeddings are normalized
        faiss.normalize_L2(embeddings_fp32)
        self.index.add(embeddings_fp32)
        self.doc_ids = list(doc_ids) # Store
        end_time = time.time()
        logging.info(f"FAISS index built in {end_time - start_time:.2f} seconds. Index size: {self.index.ntotal} vectors.")
        return True

    def search(self, query_embeddings, k):
        if self.index.ntotal == 0:
            logging.error("FAISS index is empty. Cannot search.")
            return [], [] # Return empty distances and IDs lists
        if query_embeddings.shape[0] == 0:
             logging.warning("FAISS search called with empty query embeddings.")
             return [],[]

        logging.debug(f"Searching FAISS index for {query_embeddings.shape[0]} queries, k={k}")
        query_embeddings_fp32 = query_embeddings.astype('float32')
        faiss.normalize_L2(query_embeddings_fp32)

        try:
             distances, indices = self.index.search(query_embeddings_fp32, k)
             # Map indices back to doc_ids
             # Handle cases where index returns -1 (no neighbor found within k)
             result_doc_ids = [[self.doc_ids[idx] if idx != -1 else None for idx in single_query_indices]
                               for single_query_indices in indices]
             logging.debug("FAISS search complete.")
             return distances, result_doc_ids # distances are inner products (higher is better)
        except Exception as e:
             logging.error(f"Error during FAISS search: {e}")
             return [], []


    def save(self, index_path, doc_id_path):
        logging.info(f"Saving FAISS index to {index_path}")
        try:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(self.index, index_path)
            with open(doc_id_path, 'w', encoding='utf-8') as f:
                 for doc_id in self.doc_ids:
                      f.write(f"{doc_id}\n")
            logging.info("FAISS index and doc IDs saved.")
        except Exception as e:
            logging.error(f"Error saving FAISS index or doc IDs: {e}")


    def load(self, index_path, doc_id_path):
        if not os.path.exists(index_path) or not os.path.exists(doc_id_path):
            logging.warning(f"FAISS index files not found: {index_path}, {doc_id_path}. Cannot load.")
            return False
        logging.info(f"Loading FAISS index from {index_path}")
        try:
            self.index = faiss.read_index(index_path)
            loaded_doc_ids = []
            with open(doc_id_path, 'r', encoding='utf-8') as f:
                loaded_doc_ids = [line.strip() for line in f]
            if len(loaded_doc_ids) != self.index.ntotal:
                 logging.error(f"Mismatch between loaded index size ({self.index.ntotal}) and doc ID count ({len(loaded_doc_ids)}). Load failed.")
                 self.index = None # Reset index
                 return False
            self.doc_ids = loaded_doc_ids
            logging.info(f"FAISS index loaded. Size: {self.index.ntotal} vectors.")
            return True
        except Exception as e:
             logging.error(f"Error loading FAISS index or doc IDs: {e}")
             self.index = None # Reset index
             self.doc_ids = []
             return False

