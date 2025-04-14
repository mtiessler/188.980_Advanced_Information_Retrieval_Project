import logging
import faiss
import os
from .utils import load_pickle, load_config
from .embedder import TextEmbedder

class Retriever:
    def __init__(self, config, model_path=None):
        self.config = config
        self.index = None
        self.doc_ids = []
        self.doc_id_to_index_pos = {}

        # Load the embedding model
        model_load_path = model_path or config['model_name_or_path']
        self.embedder = TextEmbedder(model_load_path, device=config['device'], max_length=config['max_seq_length'])

        # Load FAISS index and doc IDs
        try:
            index_path = os.path.join(config['output_dir'], config['doc_embeddings_file_name'])
            ids_path = os.path.join(config['output_dir'], config['doc_ids_file_name'])
            self.index = faiss.read_index(index_path)
            self.doc_ids = load_pickle(ids_path)
            # Create reverse map for faster filtering
            self.doc_id_to_index_pos = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
            logging.info(f"Retriever initialized. Index size: {self.index.ntotal}, Doc IDs: {len(self.doc_ids)}")
        except Exception as e:
            logging.error(f"Failed to initialize Retriever: {e}. Ensure embeddings and index exist.")
            self.index = None

    def search(self, query_text, top_k, allowed_doc_ids=None):
        if self.index is None:
            logging.error("FAISS index not loaded. Cannot perform search.")
            return []

        # 1. Embed the query
        query_embedding = self.embedder._embed_batch([query_text])
        query_embedding = query_embedding.astype('float32')

        if self.config['normalize_embeddings']:
            faiss.normalize_L2(query_embedding)

        # 2. Search FAISS index
        search_k = top_k * 2 if allowed_doc_ids else top_k  # Retrieve more if filtering needed
        search_k = min(search_k, self.index.ntotal)  # Cannot retrieve more than exists

        try:
            scores, indices = self.index.search(query_embedding, k=search_k)
        except Exception as e:
            logging.error(f"FAISS search failed: {e}")
            return []

        # 3. Process results
        results = []
        if indices.size == 0 or scores.size == 0:
            return []  # If empty

        faiss_indices = indices[0]  # Get results for the first (only) query
        similarity_scores = scores[0]

        for i in range(len(faiss_indices)):
            faiss_index_pos = faiss_indices[i]
            if faiss_index_pos == -1:  # Faiss returns -1 for positions < k if index is small
                continue

            doc_id = self.doc_ids[faiss_index_pos]
            score = similarity_scores[i]

            # Apply filtering if needed
            if allowed_doc_ids is None or doc_id in allowed_doc_ids:
                results.append((doc_id, score))

            if len(results) == top_k:
                break  # Stop once we have enough filtered results

        return results

    def format_trec_run(self, query_id, ranked_results, system_name="CLEF_sciBERT"):
        lines = []
        for rank, (doc_id, score) in enumerate(ranked_results):
            lines.append(f"{query_id} Q0 {doc_id} {rank + 1} {score:.6f} {system_name}\n")
        return "".join(lines)


# Example Usage
if __name__ == "__main__":
    # This assumes 'document_embeddings.index' and 'doc_ids.pkl' exist in output_dir
    try:
        config = load_config()
        retriever = Retriever(config)
        if retriever.index:
            query = "machine learning persistence"
            # Example: Simulate filtering for a snapshot (replace with actual snapshot doc IDs)
            allowed_ids_example = set(retriever.doc_ids[:50])  # Get first 50 doc IDs as example filter set

            ranked_list_filtered = retriever.search(query, top_k=10, allowed_doc_ids=allowed_ids_example)
            print(f"\n--- Filtered Search Results (Top 10 from first 50 docs) ---")
            for doc_id, score in ranked_list_filtered:
                print(f"Doc ID: {doc_id}, Score: {score:.4f}")

            ranked_list_all = retriever.search(query, top_k=config['top_k'])
            print(f"\n--- Unfiltered Search Results (Top {config['top_k']}) ---")
            for doc_id, score in ranked_list_all[:10]: # Print top 10 of unfiltered
                print(f"Doc ID: {doc_id}, Score: {score:.4f}")

            # Format for TREC
            trec_output = retriever.format_trec_run("query_test_01", ranked_list_all)
            print("\n--- Example TREC Output (first few lines) ---")
            print("".join(trec_output.splitlines(keepends=True)[:5]))

    except Exception as e:
        print(f"Could not run Retriever example: {e}. Ensure index files exist.")