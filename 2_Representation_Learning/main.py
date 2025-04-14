from pipeline.utils import parse_args, load_config
import pipeline.data_loader as data_loader
import pipeline.embedder as embedder
import pipeline.indexer as indexer
import pipeline.trainer as trainer
from pipeline.retriever import Retriever
import pipeline.evaluate as evaluate
import os
import logging
from tqdm import tqdm
import json


def load_base_data(config):
    logging.info("Loading queries and relevance judgments...")
    queries = data_loader.load_queries(config)
    qrels_df = data_loader.load_qrels(config)
    return queries, qrels_df


def handle_fine_tuning(config):
    model_path = config['model_name_or_path']

    if config.get('do_finetune', False):
        try:
            logging.info("Loading documents for fine-tuning...")
            documents_content_train = data_loader.load_documents(config)
            if documents_content_train:
                model_path = trainer.fine_tune_model(config)
            else:
                logging.error("Cannot fine-tune without documents.")
        except Exception as e:
            logging.error(f"Fine-tuning failed: {e}")
            logging.warning("Proceeding without fine-tuning due to error.")
            model_path = config['model_name_or_path']
    else:
        logging.info("Skipping fine-tuning based on config.")

    logging.info(f"Using model for embedding: {model_path}")
    return model_path, locals().get('documents_content_train', None)


def generate_embeddings_and_index(config, documents_content_train=None):
    index_path = os.path.join(config['output_dir'], config['doc_embeddings_file_name'])
    ids_path = os.path.join(config['output_dir'], config['doc_ids_file_name'])

    if os.path.exists(index_path) and os.path.exists(ids_path):
        logging.info("Embeddings and index found. Skipping generation.")
        return True

    logging.info("Generating document embeddings and building index...")
    if documents_content_train is None:
        logging.info("Loading documents for embedding...")
        documents_content_embed = data_loader.load_documents(config)
    else:
        documents_content_embed = documents_content_train

    if not documents_content_embed:
        logging.error("No document content loaded. Cannot generate embeddings.")
        return False

    try:
        text_embedder = embedder.TextEmbedder(
            config['model_name_or_path'],
            device=config['device'],
            max_length=config['max_seq_length']
        )

        doc_ids, doc_embeddings = text_embedder.generate_embeddings(
            documents_content_embed,
            batch_size=config['embedding_batch_size']
        )

        if doc_embeddings.size > 0:
            index = indexer.build_faiss_index(
                doc_embeddings,
                index_type=config['faiss_index_type'],
                normalize=config['normalize_embeddings']
            )
            if index:
                indexer.save_index(index, doc_ids, config)
                return True
            else:
                logging.error("Failed to build FAISS index.")
                return False
        else:
            logging.error("Embedding generation resulted in empty embeddings.")
            return False

    except Exception as e:
        logging.error(f"Embedding/Indexing failed: {e}")
        return False
    finally:
        # Free memory if possible
        if 'documents_content_embed' in locals():
            del documents_content_embed
        if 'doc_embeddings' in locals():
            del doc_embeddings
        if 'text_embedder' in locals():
            del text_embedder



def perform_retrieval(config, model_path, queries, qrels_df):  # Added config parameter
    logging.info("Starting retrieval for evaluation snapshots...")
    try:
        # (load index and fine-tuned/base model)
        retriever_instance = Retriever(config, model_path=model_path)
        if retriever_instance.index is None:
            raise RuntimeError("Retriever could not be initialized properly.")

        # Get snapshot definitions (mapping snapshot_id to its docs/queries)
        snapshot_info = data_loader.get_snapshot_data(qrels_df)
        if not snapshot_info:
            logging.error("Could not determine snapshot information from qrels. Cannot proceed with evaluation.")
            return False

        all_doc_ids_in_index = set(retriever_instance.doc_ids)

        for snapshot_id in config['evaluation_snapshots']:
            process_snapshot(
                config,
                snapshot_id,
                snapshot_info,
                queries,
                retriever_instance,
                all_doc_ids_in_index
            )

        return True

    except Exception as e:
        logging.error(f"Retrieval phase failed: {e}")
        return False

def process_snapshot(config, snapshot_id, snapshot_info, queries, retriever_instance, all_doc_ids_in_index):
    logging.info(f"--- Processing Snapshot: {snapshot_id} ---")
    if snapshot_id not in snapshot_info:
        logging.warning(f"Snapshot ID {snapshot_id} from config not found in derived snapshot info. Skipping.")
        return

    snapshot_queries = snapshot_info[snapshot_id]['queries']
    # Determine the set of documents *allowed* for retrieval in this snapshot
    allowed_doc_ids = snapshot_info[snapshot_id]['docs']

    # Make sure allowed doc IDs actually exist in the index
    allowed_doc_ids_in_index = allowed_doc_ids.intersection(all_doc_ids_in_index)
    if not allowed_doc_ids_in_index:
        logging.warning(f"No documents for snapshot {snapshot_id} found in the index. Skipping retrieval.")
        return

    run_file_path = os.path.join(
        config['output_dir'],  # Changed CONFIG to config
        config['run_file_name_template'].format(snapshot_id=snapshot_id)
    )

    logging.info(f"Generating run file: {run_file_path}")

    with open(run_file_path, 'w', encoding='utf-8') as f_run:
        processed_queries = 0
        for query_id in tqdm(snapshot_queries, desc=f"Retrieving for {snapshot_id}"):
            if query_id not in queries:
                logging.warning(f"Query ID {query_id} not found in loaded queries. Skipping.")
                continue

            query_text = queries[query_id]
            # Perform search, filtering by allowed docs for this snapshot
            ranked_results = retriever_instance.search(
                query_text,
                top_k=config['top_k'],
                allowed_doc_ids=allowed_doc_ids_in_index
            )

            if ranked_results:
                trec_lines = retriever_instance.format_trec_run(query_id, ranked_results)
                f_run.write(trec_lines)
                processed_queries += 1

        logging.info(f"Wrote rankings for {processed_queries} queries to {run_file_path}")



def run_evaluation_phase(config, qrels_df):
    logging.info("Starting evaluation using generated run files...")
    try:
        evaluation_results = evaluate.run_evaluation(config, qrels_df)
        print("\n--- Evaluation Summary ---")
        print(json.dumps(evaluation_results, indent=2))
        return True
    except Exception as e:
        logging.error(f"Evaluation phase failed: {e}")
        return False

def main():
    args = parse_args()
    config = load_config(args.config)

    logging.info(f"--- Starting LongEval Scientific Retrieval Pipeline with config: {args.config} ---")

    # --- 1) Load Base Data ---
    queries, qrels_df = load_base_data(config)

    # --- 2) Fine-tuning (Optional) ---
    model_path, documents_content_train = handle_fine_tuning(config)

    # --- 3) Document Embedding & Indexing ---
    if not generate_embeddings_and_index(config, documents_content_train):
        return  # Stop pipeline if embeddings/indexing failed

    # --- 4) Retrieval & Ranking for Evaluation Snapshots ---
    if not perform_retrieval(config, model_path, queries, qrels_df):
        return  # Stop pipeline if retrieval failed

    # --- 5) Run Evaluation ---
    run_evaluation_phase(config, qrels_df)

    logging.info("--- Pipeline Finished ---")


if __name__ == "__main__":
    main()
