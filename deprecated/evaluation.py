import logging
import ir_measures
from ir_measures import *

def evaluate_run(run_results, qrels, metrics_set, run_name):
    if not qrels:
         logging.warning(f"Qrels not loaded or empty for run '{run_name}'. Cannot evaluate.")
         return None

    if not run_results:
         logging.warning(f"Run results for '{run_name}' are empty. Cannot evaluate.")
         return None

    run_list = []
    for qid, doc_scores in run_results.items():
         if not doc_scores:
              logging.debug(f"No documents found for query {qid} in run {run_name}.")
              continue
         for docid, score in doc_scores.items():
              run_list.append(ir_measures.ScoredDoc(query_id=str(qid), doc_id=str(docid), score=float(score)))

    if not run_list:
         logging.warning(f"Run list for '{run_name}' is empty after conversion (no valid scored documents found). Cannot evaluate.")
         return None

    logging.info(f"Evaluating run: {run_name} with metrics: {metrics_set}")
    try:
        eval_measures = []
        if not isinstance(metrics_set, (list, set, tuple)):
             logging.error(f"metrics_set must be a list, set, or tuple, but got {type(metrics_set)}")
             return None

        for m_str in metrics_set:
             try:
                  eval_measures.append(ir_measures.parse_measure(m_str))
             except Exception as e_parse:
                  logging.error(f"Could not parse measure '{m_str}': {e_parse}")

        if not eval_measures:
             logging.error(f"No valid evaluation measures could be parsed from '{metrics_set}'. Cannot evaluate.")
             return None

        results = ir_measures.calc_aggregate(eval_measures, qrels, run_list)

        logging.info(f"Evaluation Results ({run_name}):")
        print(f"--- Evaluation: {run_name} ---")
        if results:
            for measure, value in results.items():
                logging.info(f"{str(measure)}: {value:.4f}")
        else:
             print("No results calculated.")
             logging.warning("ir_measures returned empty results.")
        print("-----------------------------")
        return results

    except Exception as e:
         logging.error(f"Evaluation failed for run '{run_name}': {e}", exc_info=True)
         return None