import os
import logging

def save_run(run, output_dir, filename, system_name):
    path = os.path.join(output_dir, filename)
    logging.info(f"Saving run for system '{system_name}' to: {path}")
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    # Expecting run format: {query_id: {doc_id: score}}
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for qid, doc_scores in run.items():
                # Sort documents by score (descending) for ranking
                sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
                if not sorted_docs:
                     logging.debug(f"Query {qid} has no results in the run.")
                     continue

                for i, (docid, score) in enumerate(sorted_docs):
                    rank = i + 1
                    # TREC format: query_id Q0 doc_id rank score system_name
                    f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score:.6f}\t{system_name}\n")
                    count += 1
        logging.info(f"Saved {count} lines to {path}.")
    except Exception as e:
        logging.error(f"Error saving run file {path}: {e}")