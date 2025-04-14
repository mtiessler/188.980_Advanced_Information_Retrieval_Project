import faiss
import numpy as np
import logging
import os
from .utils import save_pickle, load_pickle, load_config

def build_faiss_index(embeddings, index_type="IndexFlatIP", normalize=False):
    if embeddings.shape[0] == 0:
        logging.warning("Cannot build FAISS index from empty embeddings.")
        return None

    dimension = embeddings.shape[1]
    embeddings_np = embeddings.astype('float32') # FAISS requires float32

    if normalize:
        logging.info("Normalizing embeddings for FAISS index...")
        faiss.normalize_L2(embeddings_np)

    logging.info(f"Building FAISS index of type {index_type} with dimension {dimension}...")
    # TODO if non-sequential IDs: we need to check
    # index = faiss.IndexIDMap(getattr(faiss, index_type)(dimension))
    index = getattr(faiss, index_type)(dimension)

    index.add(embeddings_np)
    logging.info(f"FAISS index built. Total vectors: {index.ntotal}")
    return index

def save_index(index, doc_ids, config):
    index_path = os.path.join(config['output_dir'], config['doc_embeddings_file_name'])
    ids_path = os.path.join(config['output_dir'], config['doc_ids_file_name'])

    if index:
        faiss.write_index(index, index_path)
        logging.info(f"FAISS index saved to {index_path}")
    else:
         logging.warning("FAISS index is None, not saving.")

    save_pickle(doc_ids, ids_path)

def load_index(config):
    index_path = os.path.join(config['output_dir'], config['doc_embeddings_file_name'])
    ids_path = os.path.join(config['output_dir'], config['doc_ids_file_name'])

    try:
        index = faiss.read_index(index_path)
        logging.info(f"FAISS index loaded from {index_path}. Total vectors: {index.ntotal}")
        doc_ids = load_pickle(ids_path)
        if len(doc_ids) != index.ntotal:
            logging.warning(f"Mismatch between number of doc IDs ({len(doc_ids)}) and index size ({index.ntotal}).")
        return index, doc_ids
    except FileNotFoundError:
        logging.error(f"Could not load index or doc IDs. Files not found at {index_path} or {ids_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading index: {e}")
        raise

# Example Usage
if __name__ == "__main__":
    # Assume embeddings and doc_ids are generated from embedder.py example
    dummy_embeddings = np.random.rand(100, 768).astype('float32')
    dummy_doc_ids = [f"doc_{i}" for i in range(100)]

    # Load config
    config = load_config()

    # Build and save
    index = build_faiss_index(dummy_embeddings,
                              index_type=config['faiss_index_type'],
                              normalize=config['normalize_embeddings'])
    if index:
        save_index(index, dummy_doc_ids, config)

        # Load back
        loaded_index, loaded_doc_ids = load_index(config)
        print(f"Loaded index size: {loaded_index.ntotal}")
        print(f"First 5 loaded doc IDs: {loaded_doc_ids[:5]}")