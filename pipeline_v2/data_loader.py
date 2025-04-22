import json
import os
import logging
from tqdm.auto import tqdm
import ir_measures
import config
import gc

def stream_document_dicts(doc_dir, content_field='fulltext'):
    logging.info(f"Streaming documents from: {doc_dir} using field '{content_field}'")
    doc_files_all = [f for f in os.listdir(doc_dir) if f.endswith('.jsonl')]

    # --- Demo Mode Check ---
    files_to_process = doc_files_all
    if config.IS_DEMO_MODE:
        limit = getattr(config, 'DEMO_FILES_LIMIT', 1)
        files_to_process = doc_files_all[:limit]
        logging.info(f"--- DEMO MODE ACTIVE: Streaming only {len(files_to_process)} file(s) ---")
    # --- End Demo Mode Check ---

    if not files_to_process:
        logging.error(f"No .jsonl files found or selected to process in {doc_dir}")
        return

    processed_ids = set()

    for filename in tqdm(files_to_process, desc="Streaming doc files"):
        filepath = os.path.join(doc_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        doc_id = doc.get("id")
                        if not doc_id:
                            logging.debug(f"Document missing ID in {filename}, skipping line.")
                            continue

                        title = doc.get('title', '') or ""
                        abstract_raw = doc.get('abstract', '') or ""
                        main_content_raw = doc.get(content_field, '') or ""

                        abstract = ""
                        if abstract_raw and isinstance(abstract_raw, str) and not abstract_raw.startswith(("http://", "https://")) and len(abstract_raw) > 10:
                             abstract = abstract_raw

                        main_content = ""
                        if main_content_raw and isinstance(main_content_raw, str) and not main_content_raw.startswith(("http://", "https://")) and len(main_content_raw) > 10:
                             main_content = main_content_raw

                        if content_field == 'abstract':
                             combined_text = f"{title} {abstract}".strip()
                        else:
                             combined_text = f"{title} {abstract} {main_content}".strip()

                        if combined_text:
                            yield {"id": doc_id, "text": combined_text}
                        else:
                            logging.debug(f"Document {doc_id} resulted in empty text after validation, skipping stream.")

                    except json.JSONDecodeError:
                        logging.warning(f"Skipping invalid JSON line in {filename}: {line.strip()}")
                    except Exception as e:
                        logging.error(f"Error processing line in {filename} during streaming: {e}")
        except Exception as e:
            logging.error(f"Error reading file {filepath} during streaming: {e}")

def load_documents_structure(doc_dir, content_field=config.CONTENT_FIELD):
    logging.info(f"Loading required document data from: {doc_dir} (using content field: '{content_field}')")
    documents = {}
    doc_ids_in_order = []

    doc_files_all = [f for f in os.listdir(doc_dir) if f.endswith('.jsonl')]
    total_files = len(doc_files_all)

    # --- Demo Mode Check ---
    files_to_scan = doc_files_all
    files_to_load = doc_files_all
    if config.IS_DEMO_MODE:
        limit = getattr(config, 'DEMO_FILES_LIMIT', 1)
        files_to_scan = doc_files_all[:limit]
        files_to_load = doc_files_all[:limit]
        logging.warning(f"--- DEMO MODE ACTIVE: Loading structure from only {len(files_to_load)} file(s) ---") # Warning for emphasis
    # --- End Demo Mode Check ---

    if not files_to_scan:
         logging.error(f"No .jsonl files found or selected to process in {doc_dir}")
         return {}, []

    logging.info("Scanning files to establish document order...")
    temp_ids = []
    processed_scan_ids = set()

    for filename in tqdm(files_to_scan, desc="Scanning doc files for IDs"):
         filepath = os.path.join(doc_dir, filename)
         try:
             with open(filepath, 'r', encoding='utf-8') as f:
                  for line_num, line in enumerate(f):
                      try:
                           doc = json.loads(line)
                           doc_id = doc.get("id")
                           if doc_id and doc_id not in processed_scan_ids:
                               temp_ids.append(str(doc_id))
                               processed_scan_ids.add(doc_id)
                           elif not doc_id:
                                logging.debug(f"Missing ID in {filename} line {line_num+1} during scan.")
                      except json.JSONDecodeError:
                           logging.debug(f"Invalid JSON in {filename} line {line_num+1} during scan.")
                           pass
                      except Exception:
                           pass
         except Exception as e_outer:
             logging.warning(f"Could not scan file {filename}: {e_outer}")
             pass

    doc_ids_in_order = temp_ids
    del processed_scan_ids
    del temp_ids
    gc.collect()
    logging.info(f"Established order for {len(doc_ids_in_order)} unique document IDs found (from processed files).")


    logging.info("Loading required document fields into memory...")
    loaded_count = 0
    docs_actually_loaded = set()
    for filename in tqdm(files_to_load, desc="Loading required fields"):
        filepath = os.path.join(doc_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        doc = json.loads(line)
                        doc_id = doc.get("id")
                        if not doc_id: continue

                        title = doc.get('title', '') or ""
                        abstract = doc.get('abstract', '') or ""
                        main_content = doc.get(content_field) or None

                        documents[doc_id] = {
                            'id': doc_id,
                            'title': title,
                            'abstract': abstract,
                            content_field: main_content
                        }

                        if doc_id not in docs_actually_loaded:
                             loaded_count += 1
                             docs_actually_loaded.add(doc_id)
                    except json.JSONDecodeError:
                         logging.warning(f"Skipping invalid JSON line {line_num+1} in {filename} during load.")
                    except Exception as e_inner_load:
                         logging.error(f"Error loading line {line_num+1} structure in {filename}: {e_inner_load}")
        except Exception as e_outer_load:
             logging.error(f"Error reading file {filepath} during structure loading: {e_outer_load}")

    final_doc_ids_in_order = [doc_id for doc_id in doc_ids_in_order if doc_id in documents]
    if len(final_doc_ids_in_order) != len(doc_ids_in_order):
         logging.info(f"Refined doc ID order list to {len(final_doc_ids_in_order)} based on loaded documents.")

    logging.info(f"Stored required fields for {len(documents)} unique document IDs ({loaded_count} added in this pass).")
    gc.collect()
    return documents, final_doc_ids_in_order


def load_queries(queries_file):
    logging.info(f"Loading queries from: {queries_file}")
    queries = {}
    try:
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    queries[parts[0]] = parts[1]
                else:
                    logging.warning(f"Skipping malformed query line: {line.strip()}")
    except FileNotFoundError:
        logging.error(f"Queries file not found: {queries_file}")
        return {}
    logging.info(f"Loaded {len(queries)} queries.")
    return queries

def load_qrels(qrels_file):
    logging.info(f"Loading qrels from: {qrels_file}")
    try:
        qrels = list(ir_measures.read_trec_qrels(qrels_file))
    except FileNotFoundError:
        logging.error(f"Qrels file not found: {qrels_file}")
        return []
    except Exception as e:
        logging.error(f"Error reading qrels file {qrels_file}: {e}")
        return []
    logging.info(f"Loaded {len(qrels)} relevance judgments.")
    return qrels