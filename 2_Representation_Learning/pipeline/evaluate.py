import logging
from ranx import Qrels, Run, evaluate
import pandas as pd
from .utils import load_config
from .data_loader import load_qrels
import os

def prepare_ranx_qrels(qrels_df, snapshot_id=None):
    """Prepares Qrels object for ranx from DataFrame, optionally filtered by snapshot."""
    if snapshot_id:
        qrels_df_filtered = qrels_df[qrels_df['snapshot_id'] == snapshot_id].copy()
    else:
        qrels_df_filtered = qrels_df.copy()

    # Ranx expects relevance >= 1 for relevant docs
    qrels_df_filtered['relevance_score'] = qrels_df_filtered['relevance_score'].apply(lambda x: max(0, int(x)))  # Example: Treat >0 as relevant, map score if graded

    # TODO (I need to check) Filter out non-relevant judgments if needed by metric (most handle 0s)
    # qrels_df_filtered = qrels_df_filtered[qrels_df_filtered['relevance_score'] > 0]

    # Rename for ranx library -> query_id, doc_id, score
    qrels_df_filtered = qrels_df_filtered[['query_id', 'doc_id', 'relevance_score']]
    qrels_dict = qrels_df_filtered.groupby('query_id').apply(lambda x: dict(zip(x['doc_id'], x['relevance_score']))).to_dict()

    return Qrels(qrels_dict)


def prepare_ranx_run(run_file_path):
    try:
        run_df = pd.read_csv(run_file_path, sep=' ', header=None,
                           names=['query_id', 'Q0', 'doc_id', 'rank', 'score', 'system_name'],
                           dtype={'query_id': str, 'doc_id': str, 'score': float})

        run_dict = run_df.groupby('query_id').apply(lambda x: dict(zip(x['doc_id'], x['score']))).to_dict()
        return Run(run_dict)
    except FileNotFoundError:
        logging.error(f"Run file not found: {run_file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading run file {run_file_path}: {e}")
        return None

def calculate_relative_ndcg_drop(results):
    rnd_results = {}
    snapshot_ids = sorted(results.keys())  # Ensure snapshots are in chronological order

    for i in range(1, len(snapshot_ids)):
        prev_snapshot_id = snapshot_ids[i - 1]
        curr_snapshot_id = snapshot_ids[i]

        prev_ndcg = results[prev_snapshot_id].get("ndcg@10", None)
        curr_ndcg = results[curr_snapshot_id].get("ndcg@10", None)

        if prev_ndcg is not None and curr_ndcg is not None:
            rnd = (curr_ndcg - prev_ndcg) / prev_ndcg if prev_ndcg != 0 else 0
            rnd_results[f"{prev_snapshot_id}_to_{curr_snapshot_id}"] = {"rnd@10": rnd}
        else:
            rnd_results[f"{prev_snapshot_id}_to_{curr_snapshot_id}"] = {"rnd@10": "N/A"}

    return rnd_results

def run_evaluation(config, qrels_df):
    """Runs evaluation for all specified snapshots and calculates RnD."""
    results = {}
    logging.info("Starting evaluation process for specified snapshots...")

    for snapshot_id in config['evaluation_snapshots']:
        logging.info(f"--- Evaluating Snapshot: {snapshot_id} ---")
        run_file_path = os.path.join(config['output_dir'],
                                     config['run_file_name_template'].format(snapshot_id=snapshot_id))

        qrels_ranx = prepare_ranx_qrels(qrels_df, snapshot_id=snapshot_id)
        run_ranx = prepare_ranx_run(run_file_path)

        if not qrels_ranx.get_query_ids() or run_ranx is None or not run_ranx.get_query_ids():
            logging.warning(f"Skipping evaluation for {snapshot_id} due to missing qrels or run data.")
            results[snapshot_id] = {}
            continue

        # Ensure qrels and run cover the same queries for fair evaluation
        common_query_ids = set(qrels_ranx.get_query_ids()) & set(run_ranx.get_query_ids())
        if not common_query_ids:
            logging.warning(f"No common queries between qrels and run for snapshot {snapshot_id}.")
            results[snapshot_id] = {}
            continue

        # Create new Qrels and Run objects with only the common queries
        qrels_dict = {qid: qrels_ranx.qrels[qid] for qid in common_query_ids if qid in qrels_ranx.qrels}
        run_dict = {qid: run_ranx.run[qid] for qid in common_query_ids if qid in run_ranx.run}

        qrels_filtered = Qrels(qrels_dict)
        run_filtered = Run(run_dict)

        logging.info(f"Evaluating on {len(common_query_ids)} common queries for snapshot {snapshot_id}.")

        try:
            snapshot_results = evaluate(qrels_filtered, run_filtered, config['eval_metrics'])
            results[snapshot_id] = snapshot_results
            logging.info(f"Results for {snapshot_id}: {snapshot_results}")
        except Exception as e:
            logging.error(f"Error during ranx evaluation for snapshot {snapshot_id}: {e}")
            results[snapshot_id] = {"error": str(e)}

    rnd_results = calculate_relative_ndcg_drop(results)
    results["rnd"] = rnd_results

    logging.info("Evaluation process finished.")
    return results

# TEST
if __name__ == "__main__":
    # This assumes qrels are loaded and run files exist in output_dir
    print("Running evaluation example...")
    config = load_config()
    qrels = load_qrels(config)
    if not qrels.empty:
        # IMPORTANT: The generated run files generated using the retriever exist ---
        # Example: Create dummy run files if they don't exist
        for snap_id in config['evaluation_snapshots']:
            dummy_run_path = os.path.join(config['output_dir'], config['run_file_name_template'].format(snapshot_id=snap_id))
            if not os.path.exists(dummy_run_path):
                print(f"Creating dummy run file: {dummy_run_path}")
                # Create a very basic dummy file
                snap_qrels = qrels[qrels['snapshot_id'] == snap_id]
                if not snap_qrels.empty:
                    with open(dummy_run_path, 'w') as f:
                        # Just write one line per query mentioned in qrels for this snapshot
                        processed_queries = set()
                        for _, row in snap_qrels.iterrows():
                            if row['query_id'] not in processed_queries:
                                f.write(f"{row['query_id']} Q0 {row['doc_id']} 1 0.99 DummyRun\n")
                                processed_queries.add(row['query_id'])
                else:
                    print(f"No qrels found for snapshot {snap_id}, cannot create dummy run file.")

        # Run evaluation
        evaluation_results = run_evaluation(config, qrels)
        print("\n--- Evaluation Summary ---")
        for snapshot, metrics in evaluation_results.items():
            if snapshot != "rnd":
                print(f"Snapshot: {snapshot}")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
        print("\n--- RnD Results ---")
        for rnd_metric, rnd_value in evaluation_results["rnd"].items():
            print(f"  {rnd_metric}: {rnd_value}")

    else:
        print("Qrels DataFrame is empty, skipping evaluation example.")