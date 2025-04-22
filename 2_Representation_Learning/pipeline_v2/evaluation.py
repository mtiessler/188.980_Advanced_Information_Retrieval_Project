import logging
import ir_measures
from ir_measures import *

def evaluate_run(run_results, qrels, metrics_str, run_name):
    if not qrels:
         logging.warning("Qrels not loaded. Cannot perform evaluation.")
         return None

    if not run_results:
         logging.warning(f"Run results for '{run_name}' are empty. Cannot evaluate.")
         return None

    run_list = []
    for qid, doc_scores in run_results.items():
         # Handle cases where doc_scores might be empty for a query
         if not doc_scores:
              logging.debug(f"No documents found for query {qid} in run {run_name}.")
              continue
         for docid, score in doc_scores.items():
              run_list.append(ir_measures.ScoredDoc(query_id=str(qid), doc_id=str(docid), score=float(score)))

    if not run_list:
         logging.warning(f"Run list for '{run_name}' is empty after conversion (no valid scored documents found). Cannot evaluate.")
         return None

    logging.info(f"Evaluating run: {run_name} with metrics: {metrics_str}")
    try:
        eval_measures = []
        for m_str in metrics_str.split():
             try:
                  eval_measures.append(ir_measures.parse_measure(m_str))
             except Exception as e_parse:
                  logging.error(f"Could not parse measure '{m_str}': {e_parse}")
        if not eval_measures:
             logging.error(f"No valid evaluation measures found in '{metrics_str}'. Cannot evaluate.")
             return None

        results = ir_measures.calc_aggregate(eval_measures, qrels, run_list)

        logging.info(f"Evaluation Results ({run_name}):")
        print(f"--- Evaluation: {run_name} ---")
        if results:
            for measure, value in results.items():
                print(f"{measure}: {value:.4f}")
                logging.info(f"{measure}: {value:.4f}")
        else:
             print("No results calculated.")
             logging.warning("ir_measures returned empty results.")
        print("-----------------------------")
        return results

    except Exception as e:
         logging.error(f"Evaluation failed for run '{run_name}': {e}", exc_info=True)
         return None