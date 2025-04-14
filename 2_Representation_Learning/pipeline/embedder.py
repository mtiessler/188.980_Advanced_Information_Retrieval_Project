import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import logging
from .utils import load_config

class TextEmbedder:
    def __init__(self, model_name_or_path, device='cuda', max_length=512):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval() # Set to evaluation mode by default
        self.max_length = max_length
        logging.info(f"Embedder initialized with model: {model_name_or_path}")

    def _embed_batch(self, texts):
        """Generates embeddings for a batch of texts."""
        inputs = self.tokenizer(texts, return_tensors='pt', max_length=self.max_length,
                                truncation=True, padding='max_length')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use CLS token embedding
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings

    def generate_embeddings(self, text_map, batch_size=32):
        """
        Returns two lists -> ordered_ids, ordered_embeddings (numpy array).
        """
        if not text_map:
            return [], np.array([])

        ids_list = list(text_map.keys())
        texts_list = [text_map[id] for id in ids_list]
        all_embeddings = []

        logging.info(f"Generating embeddings for {len(ids_list)} texts...")
        for i in tqdm(range(0, len(ids_list), batch_size), desc="Embedding Batches"):
            batch_texts = texts_list[i:i+batch_size]
            if not batch_texts: # Just in case :P
                continue
            batch_embeddings = self._embed_batch(batch_texts)
            all_embeddings.append(batch_embeddings)

        if not all_embeddings:
             return ids_list, np.array([])

        ordered_embeddings = np.vstack(all_embeddings)
        logging.info(f"Generated embeddings shape: {ordered_embeddings.shape}")
        return ids_list, ordered_embeddings

# Test
if __name__ == "__main__":
    # Dummy data for test
    sample_docs = {
        "doc1": "This is the first document title. This is the abstract.",
        "doc2": "Another title. Second abstract here.",
        "doc3": "Yet another example document."
    }

    # Load config using the function from utils
    config = load_config()

    embedder = TextEmbedder(config['model_name_or_path'], device=config['device'])
    doc_ids, embeddings = embedder.generate_embeddings(sample_docs, batch_size=config['embedding_batch_size'])
    print(f"Generated embeddings for IDs: {doc_ids}")
    print(f"Embeddings matrix shape: {embeddings.shape}")