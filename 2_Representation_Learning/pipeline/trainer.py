import logging
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data_loader import load_queries, load_qrels, load_documents
import torch


def prepare_training_data(config, queries, qrels_df, documents_content, max_samples=-1):
    train_examples = []
    query_ids = qrels_df['query_id'].unique()
    all_doc_ids = list(documents_content.keys())

    logging.info("Preparing training data examples...")
    logging.info(f"First 5 queries: {list(queries.items())[:5]}")
    logging.info(f"Qrels DataFrame head:\n{qrels_df.head()}")
    logging.info(f"Unique query IDs in qrels: {qrels_df['query_id'].nunique()}")
    logging.info(f"Number of documents loaded: {len(documents_content)}")

    qrels_grouped = qrels_df.groupby('query_id')

    sample_count = 0
    for query_id in tqdm(query_ids, desc="Processing queries for training"):
        if query_id not in queries:
            logging.info(f"Skipping query {query_id} as text is missing.")
            continue

        query_text = queries[query_id]
        group = qrels_grouped.get_group(query_id)
        positive_docs = group[group['relevance_score'] > 0]['doc_id'].tolist()

        logging.info(f"Query ID: {query_id}, Positive Docs: {positive_docs}") # Added logging

        if not positive_docs:
            logging.info(f"No positive docs for query {query_id}. Skipping.")
            continue

        if config['st_loss'] == "TripletLoss":
            for pos_doc_id in positive_docs:
                if pos_doc_id not in documents_content:
                    logging.info(f"Positive doc {pos_doc_id} missing in content. Skipping.")
                    continue

                pos_doc_text = documents_content[pos_doc_id]
                neg_doc_id = random.choice(all_doc_ids)

                attempts = 0
                while (neg_doc_id in positive_docs or neg_doc_id not in documents_content) and attempts < 10:
                    neg_doc_id = random.choice(all_doc_ids)
                    attempts += 1

                if neg_doc_id in documents_content:
                    neg_doc_text = documents_content[neg_doc_id]
                    train_examples.append(InputExample(texts=[query_text, pos_doc_text, neg_doc_text]))
                    sample_count += 1
                else:
                    logging.info(f"No valid negative doc found for query {query_id} after 10 attempts.")

        elif config['st_loss'] == "MultipleNegativesRankingLoss":
            for pos_doc_id in positive_docs:
                if pos_doc_id not in documents_content:
                    logging.info(f"Positive doc {pos_doc_id} missing in content. Skipping.")
                    continue

                pos_doc_text = documents_content[pos_doc_id]
                train_examples.append(InputExample(texts=[query_text, pos_doc_text]))
                sample_count += 1

        else:
            logging.info(f"Unsupported loss '{config['st_loss']}'. Falling back to MultipleNegativesRankingLoss.")
            for pos_doc_id in positive_docs:
                if pos_doc_id not in documents_content:
                    continue
                pos_doc_text = documents_content[pos_doc_id]
                train_examples.append(InputExample(texts=[query_text, pos_doc_text]))
                sample_count += 1

        if max_samples > 0 and sample_count >= max_samples:
            logging.info(f"Reached maximum training samples limit: {max_samples}")
            break

    logging.info(f"Total training examples prepared: {len(train_examples)}")
    return train_examples

def fine_tune_model(config):
    if not config.get('do_finetune', False):
        logging.info("Fine-tuning disabled in config. Skipping training.")
        return config['model_name_or_path']
    device = config['device']

    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA not available. Falling back to CPU.")
    else:
        device = "cpu"

    # Validate loss type early
    valid_losses = {"MultipleNegativesRankingLoss", "TripletLoss"}
    if config['st_loss'] not in valid_losses:
        logging.warning(f"Unsupported loss function '{config['st_loss']}'. Defaulting to MultipleNegativesRankingLoss.")
        config['st_loss'] = "MultipleNegativesRankingLoss"

    logging.info("Starting fine-tuning process...")

    # Load data
    queries = load_queries(config)
    qrels_df = load_qrels(config)
    documents_content = load_documents(config)

    # Load base model
    model = SentenceTransformer(config['model_name_or_path'], device=config['device'])
    model.max_seq_length = config['max_seq_length']

    # Prepare training data
    train_samples = prepare_training_data(config, queries, qrels_df, documents_content, config['st_train_samples_max'])
    if not train_samples:
        logging.error("No training samples prepared. Aborting fine-tuning.")
        return config['model_name_or_path']

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=config['st_batch_size'])

    # Configure loss
    loss_name = config['st_loss']
    if loss_name == "MultipleNegativesRankingLoss":
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    elif loss_name == "TripletLoss":
        train_loss = losses.TripletLoss(
            model=model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=config.get('triplet_margin', 0.5)
        )

    # Train
    logging.info(f"Training model using {loss_name} for {config['st_num_epochs']} epochs...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config['st_num_epochs'],
        optimizer_params={'lr': config['st_learning_rate']},
        warmup_steps=config['st_warmup_steps'],
        output_path=config['st_model_output_path'],
        show_progress_bar=True,
        checkpoint_save_steps=5000,
        checkpoint_path=config['st_model_output_path'] + "_checkpoints"
    )

    logging.info(f"Fine-tuning complete. Model saved at {config['st_model_output_path']}")
    return config['st_model_output_path']


if __name__ == "__main__":
    from utils import parse_args, load_config

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

    args = parse_args()
    config = load_config(args.config)

    model_path = fine_tune_model(config)
    logging.info(f"Model training process completed. Final model path: {model_path}")
