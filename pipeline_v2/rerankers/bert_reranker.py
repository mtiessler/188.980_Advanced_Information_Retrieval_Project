import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BertReranker:
    def __init__(self, model_name, device):
        logging.info(f"Loading re-ranking model: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        logging.info("Re-ranking model loaded.")

    @torch.no_grad()
    def rerank(self, query, docs_dict, preprocessor, batch_size):
        if not docs_dict:
             logging.debug(f"Re-ranker called with empty docs_dict for query '{query[:50]}...'")
             return {}

        doc_ids = list(docs_dict.keys())
        doc_texts = list(docs_dict.values())

        processed_query = preprocessor.preprocess_for_bert(query)
        processed_docs = [preprocessor.preprocess_for_bert(text) for text in doc_texts]

        pairs = [[processed_query, doc_text] for doc_text in processed_docs]
        all_scores = []

        logging.debug(f"Re-ranking {len(pairs)} pairs for query '{processed_query[:50]}...'")
        iterator = range(0, len(pairs), batch_size)

        for i in iterator:
            batch_pairs = pairs[i:i+batch_size]
            if not batch_pairs: continue

            try:
                encoded_input = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                ).to(self.device)

                outputs = self.model(**encoded_input)
                if outputs.logits.shape[1] > 1:
                     scores = outputs.logits[:, 1]
                     logging.warning(f"Cross-encoder {self.model.config.name_or_path} has >1 output logit. Using index 1. Verify this is correct.")
                else:
                    scores = outputs.logits.squeeze(-1)

                all_scores.extend(scores.cpu().numpy().tolist())

            except Exception as e:
                 logging.error(f"Error re-ranking batch for query '{processed_query[:50]}...' at index {i}: {e}")
                 all_scores.extend([-999.0] * len(batch_pairs))


        if len(all_scores) != len(doc_ids):
            logging.error(f"Score count ({len(all_scores)}) mismatch with doc ID count ({len(doc_ids)}) after re-ranking. Returning empty results.")
            return {}

        reranked_results = {doc_id: float(score) for doc_id, score in zip(doc_ids, all_scores)}
        logging.debug("Re-ranking complete for query.")
        return reranked_results