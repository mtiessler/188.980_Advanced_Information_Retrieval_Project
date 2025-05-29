import os
import logging

def save_run(run, output_dir, filename, system_name):
    """
    Save run results to a file in TREC format.

    Each line in the output file will have the format:
    query_id Q0 doc_id rank score system_name

    Args:
        run (dict): Dictionary of run results. 
            Format: {query_id: {doc_id: score, ...}, ...}
        output_dir (str): Directory where the run file will be saved.
        filename (str): Name of the output file.
        system_name (str): Name of the system (used in the output file for each line).

    """    
    path = os.path.join(output_dir, filename)
    logging.info(f"Saving run for system '{system_name}' to: {path}")
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
    count = 0   # Counter for the number of lines written
    
    # Expecting run format: {query_id: {doc_id: score}}
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for qid, doc_scores in run.items():
                # Sort documents by score (descending) for ranking
                sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
                if not sorted_docs:
                     logging.debug(f"Query {qid} has no results in the run.")
                     continue  # Skip queries with no results

                for i, (docid, score) in enumerate(sorted_docs):
                    rank = i + 1
                    # write in TREC format: query_id Q0 doc_id rank score system_name
                    f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score:.6f}\t{system_name}\n")
                    count += 1   # Increment line counter
        logging.info(f"Saved {count} lines to {path}.")
    except Exception as e:
        logging.error(f"Error saving run file {path}: {e}")
