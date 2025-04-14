import os
import glob
import json
import pandas as pd
from tqdm import tqdm
import logging

def norm_path(*parts):
    return os.path.normpath(os.path.join(*parts))

def load_queries(config):
    file_path = norm_path(config['data_dir'], config['queries_file_name'])
    try:
        queries_df = pd.read_csv(file_path, sep='\t', header=None, names=['query_id', 'query_text'], dtype={'query_id': str})
        queries = pd.Series(queries_df.query_text.values, index=queries_df.query_id).to_dict()
        logging.info(f"Loaded {len(queries)} queries from {file_path}")
        return queries
    except FileNotFoundError:
        logging.error(f"Queries file not found: {file_path}")
        return {}

def load_qrels(config):
    file_path = norm_path(config['data_dir'], config['qrels_file_name'])
    try:
        qrels_df = pd.read_csv(file_path, sep=' ', header=None,
                               names=['query_id', 'snapshot_id', 'doc_id', 'relevance_score'],
                               dtype={'query_id': str, 'snapshot_id': str, 'doc_id': str})
        logging.info(f"Loaded {len(qrels_df)} relevance judgments from {file_path}")
        return qrels_df
    except FileNotFoundError:
        logging.error(f"Qrels file not found: {file_path}")
        return pd.DataFrame()

def load_documents(config):
    """
    Loads document content iteratively from JSONL files.
    """
    documents_content = {}
    doc_dir = norm_path(config['data_dir'], config['documents_dir_name'])
    jsonl_files = glob.glob(norm_path(doc_dir, "*.jsonl"))
    separator = config.get('text_separator', ' ')

    if not jsonl_files:
        logging.warning(f"No .jsonl files found in {doc_dir}")
        return {}

    logging.info(f"Loading documents from {len(jsonl_files)} files in {doc_dir}...")
    for file_path in tqdm(jsonl_files, desc="Reading document files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        doc_data = json.loads(line)
                        doc_id = str(doc_data.get('id'))  # Ensure doc_id is string
                        if doc_id:
                            content_parts = [doc_data.get(field, '') or '' for field in config['doc_content_fields']]
                            non_empty_parts = [part for part in content_parts if part and part.strip()]
                            documents_content[doc_id] = separator.join(non_empty_parts)
                        if len(documents_content) >= 1000: break
                    except json.JSONDecodeError:
                        logging.warning(f"Skipping invalid JSON line in {file_path}: {line.strip()}")
                    except Exception as e_line:
                        logging.warning(f"Error processing line in {file_path}: {e_line} | Line: {line.strip()}")

        except Exception as e_file:
            logging.error(f"Error reading file {file_path}: {e_file}")

    logging.info(f"Loaded content for {len(documents_content)} documents.")
    return documents_content


def get_snapshot_data(qrels_df):
    """
    Determines documents and queries per snapshot from the Qrels DataFrame.
    Returns a dictionary: snapshot_id -> {'queries': set(query_id), 'docs': set(doc_id)}
    """
    snapshot_data = {}

    if qrels_df.empty:
        logging.warning("Qrels DataFrame is empty.")
        return snapshot_data

    for snapshot_id, group in qrels_df.groupby('snapshot_id'):
        snapshot_data[snapshot_id] = {
            'queries': set(group['query_id'].unique()),
            'docs': set(group['doc_id'].unique())
        }

    logging.info(f"Derived snapshot info for snapshots: {list(snapshot_data.keys())}")
    return snapshot_data

# TEST
if __name__ == "__main__":
    from utils import load_config
    config = load_config()
    queries = load_queries(config)
    qrels = load_qrels(config)
    documents = load_documents(config)
    snapshot_info = get_snapshot_data(qrels)
    print(f"First 5 queries: {list(queries.items())[:5]}")
    print(f"Qrels shape: {qrels.shape}")
    print(f"Documents loaded: {len(documents)}")
    print(f"Snapshot info keys: {snapshot_info.keys()}")