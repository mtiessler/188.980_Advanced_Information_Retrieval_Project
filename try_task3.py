import os
import json
from pathlib import Path
import collections
from typing import Dict, List, Tuple, Any
import sys
import time
import gzip
import shutil
import torch
import ir_measures
from tqdm import tqdm
from ragatouille import RAGPretrainedModel
from rank_bm25 import BM25Okapi
import re
import nltk

USE_GOOGLE_DRIVE = True

DATA_DIR_NAME = "longeval_sci_testing_2025_abstract"
RAGATOUILLE_COLBERT_MODEL_CACHE_SUBDIR = "colbert_model_cache"

BM25_K1 = 1.5
BM25_B = 0.75
BM25_REMOVE_STOPWORDS = True
BM25_ENABLE_STEMMING = True
BM25_TOP_K_CANDIDATES = 200

COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"
RUN_TAG = "BM25_ColBERTv2_ReRanker_v1"
MAX_SEQ_LENGTH = 512
MAX_AUTHORS_IN_INPUT = 3
K_RETRIEVAL = 100

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)

english_stopwords_global = []
if BM25_REMOVE_STOPWORDS:
    try:
        from nltk.corpus import stopwords as nltk_stopwords_import

        english_stopwords_global = nltk_stopwords_import.words('english')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords as nltk_stopwords_import_after_download

        english_stopwords_global = nltk_stopwords_import_after_download.words('english')

stemmer_global = None
if BM25_ENABLE_STEMMING:
    from nltk.stem import PorterStemmer

    stemmer_global = PorterStemmer()

if USE_GOOGLE_DRIVE:
    try:
        from google.colab import drive
    except ImportError:
        print(
            "ERROR: google.colab.drive module not found. Set USE_GOOGLE_DRIVE to False or run in a Colab environment.")
        sys.exit(1)

if USE_GOOGLE_DRIVE:
    GOOGLE_DRIVE_PROJECT_ROOT = Path("/content/drive/MyDrive/AIR_Project/")
    DATA_DIR = GOOGLE_DRIVE_PROJECT_ROOT / DATA_DIR_NAME
    OUTPUT_DIR = GOOGLE_DRIVE_PROJECT_ROOT / "colbert_pipeline_output" / "YOUR-SUBMISSION-BM25-ColBERT"
    COLBERT_MODEL_CACHE_PATH = GOOGLE_DRIVE_PROJECT_ROOT / "colbert_pipeline_output" / RAGATOUILLE_COLBERT_MODEL_CACHE_SUBDIR
else:
    SCRIPT_DIR = Path.cwd()
    DATA_DIR = SCRIPT_DIR / DATA_DIR_NAME
    OUTPUT_DIR = Path("./YOUR-SUBMISSION-BM25-ColBERT")
    COLBERT_MODEL_CACHE_PATH = OUTPUT_DIR / RAGATOUILLE_COLBERT_MODEL_CACHE_SUBDIR

QUERIES_FILE_2024_11 = DATA_DIR / "queries_2024-11_test.txt"
QUERIES_FILE_2025_01 = DATA_DIR / "queries_2025-01_test.txt"
DOCUMENTS_DIR = DATA_DIR / "documents"


def bm25_tokenizer(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    if BM25_REMOVE_STOPWORDS:
        tokens = [token for token in tokens if token not in english_stopwords_global]
    if BM25_ENABLE_STEMMING and stemmer_global:
        tokens = [stemmer_global.stem(token) for token in tokens]
    return tokens


def mount_drive_and_verify_paths(data_dir_path: Path, queries_file_2024_11_path: Path,
                                 queries_file_2025_01_path: Path, docs_dir_path: Path) -> bool:
    try:
        drive.mount('/content/drive', force_remount=True)
        print("INFO: Google Drive mounted successfully.")
    except Exception as e:
        print(f"ERROR: Failed to mount Google Drive: {e}")
        print("INFO: If error persists, try 'Runtime -> Factory reset runtime' in Colab.")
        return False
    return verify_common_paths("Google Drive", data_dir_path, queries_file_2024_11_path, queries_file_2025_01_path,
                               docs_dir_path)


def verify_local_paths(data_dir_path: Path, queries_file_2024_11_path: Path,
                       queries_file_2025_01_path: Path, docs_dir_path: Path) -> bool:
    print("INFO: Verifying local paths...")
    return verify_common_paths("Local", data_dir_path, queries_file_2024_11_path, queries_file_2025_01_path,
                               docs_dir_path)


def verify_common_paths(env_type: str, data_dir_path: Path, queries_file_2024_11_path: Path,
                        queries_file_2025_01_path: Path, docs_dir_path: Path) -> bool:
    paths_to_check = {
        f"{env_type} Dataset directory": data_dir_path,
        f"{env_type} Queries file 2024-11": queries_file_2024_11_path,
        f"{env_type} Queries file 2025-01": queries_file_2025_01_path,
        f"{env_type} Documents directory": docs_dir_path
    }
    all_exist = True
    for name, path_val in paths_to_check.items():
        display_path = path_val.resolve() if not str(path_val).startswith("/content/drive") else path_val
        is_dir_check = name.endswith("Documents directory") or name.endswith("Dataset directory")

        if is_dir_check and not path_val.is_dir():
            print(f"ERROR: {name} directory not found at: {display_path}")
            all_exist = False
        elif not is_dir_check and not path_val.exists():
            print(f"ERROR: {name} file not found at: {display_path}")
            all_exist = False

    if all_exist:
        print(f"INFO: All required {env_type.lower()} paths verified successfully.")
    else:
        print(
            f"ERROR: One or more required {env_type.lower()} paths are missing. Base data directory expected at: {data_dir_path}")
    return all_exist


def load_queries(file_path: Path) -> Dict[str, str]:
    queries = {}
    display_path = file_path.resolve() if not str(file_path).startswith("/content/drive") else file_path
    print(f"INFO: Attempting to load queries from {display_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    query_id, query_text = parts
                    queries[query_id] = query_text
                else:
                    print(
                        f"WARNING: Skipping malformed line #{i + 1} in queries file ({file_path.name}): {line.strip()}")
        print(f"INFO: Successfully loaded {len(queries)} queries from {file_path.name}.")
    except FileNotFoundError:
        print(f"ERROR: Queries file not found: {display_path}")
    except Exception as e:
        print(f"ERROR: Error loading queries from {display_path}: {e}")
    return queries


def load_and_prepare_documents(docs_dir_path: Path, batch_size_info_log: int = 10000) -> Tuple[
    Dict[str, str], List[str], List[str]]:
    doc_id_to_text_map = {}
    corpus_texts_for_bm25 = []
    corpus_doc_ids_for_bm25 = []

    display_docs_dir_path = docs_dir_path.resolve() if not str(docs_dir_path).startswith(
        "/content/drive") else docs_dir_path
    if not docs_dir_path.is_dir():
        print(f"ERROR: Documents directory not found: {display_docs_dir_path}")
        return {}, [], []

    jsonl_files = list(docs_dir_path.glob('*.jsonl'))
    if not jsonl_files:
        jsonl_files = list(docs_dir_path.glob('*.jsonl.gz'))
        if jsonl_files:
            print(f"INFO: Found .jsonl.gz files, will decompress on the fly.")
        else:
            print(f"WARNING: No .jsonl or .jsonl.gz files found in document directory: {display_docs_dir_path}")
            return {}, [], []

    print(f"INFO: Preparing documents from {len(jsonl_files)} files in {display_docs_dir_path}...")
    total_docs_processed_in_files = 0

    for file_idx, data_file in enumerate(tqdm(jsonl_files, desc="Loading document files", unit="file")):
        open_func = open
        if str(data_file).endswith(".gz"):
            open_func = gzip.open

        try:
            with open_func(data_file, 'rt', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    try:
                        doc_data = json.loads(line)
                        doc_id = str(doc_data.get("id"))
                        if not doc_id:
                            print(f"WARNING: Document in {data_file.name} line {line_idx + 1} has no ID. Skipping.")
                            continue

                        title = doc_data.get("title", "")
                        abstract = doc_data.get("abstract", "")

                        doc_parts = [title, abstract]
                        document_text_input = " [SEP] ".join(
                            part for part in doc_parts if part and part.strip()).strip()

                        if document_text_input:
                            if doc_id not in doc_id_to_text_map:
                                doc_id_to_text_map[doc_id] = document_text_input
                                corpus_texts_for_bm25.append(document_text_input)
                                corpus_doc_ids_for_bm25.append(doc_id)
                                total_docs_processed_in_files += 1
                                if total_docs_processed_in_files % batch_size_info_log == 0:
                                    print(
                                        f"INFO: Loaded and prepared {total_docs_processed_in_files} documents so far...")
                        else:
                            print(
                                f"WARNING: Document ID {doc_id} in {data_file.name} has no content after processing. Skipping.")

                    except json.JSONDecodeError:
                        print(f"WARNING: Skipping malformed JSON line in {data_file.name} (line {line_idx + 1})")
                        continue
                    except Exception as e_doc:
                        print(
                            f"WARNING: Error processing a document in {data_file.name} (line {line_idx + 1}): {e_doc}")
        except Exception as e_file:
            print(f"ERROR: Error reading or processing file {data_file}: {e_file}")

    if corpus_texts_for_bm25:
        print(f"INFO: Successfully loaded and prepared a total of {len(corpus_texts_for_bm25)} unique documents.")
    else:
        print("WARNING: No documents were loaded. The corpus is empty.")
    return doc_id_to_text_map, corpus_texts_for_bm25, corpus_doc_ids_for_bm25


def generate_trec_run_file(run_data: Dict[str, Dict[str, float]], output_file_path: Path, run_tag_for_file: str):
    display_output_file_path = output_file_path.resolve() if not str(output_file_path).startswith(
        "/content/drive") else output_file_path
    print(f"INFO: Generating TREC run file at {display_output_file_path} with tag '{run_tag_for_file}'...")
    lines_written = 0
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            for q_id, doc_scores in run_data.items():
                sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
                for rank, (doc_id, score) in enumerate(sorted_docs[:K_RETRIEVAL], 1):
                    f_out.write(f"{q_id} Q0 {doc_id} {rank} {score:.6f} {run_tag_for_file}\n")
                    lines_written += 1
        print(f"INFO: TREC run file saved to {display_output_file_path}, {lines_written} lines written.")
    except Exception as e:
        print(f"ERROR: Failed to write TREC run file to {display_output_file_path}: {e}")


def main_bm25_colbert_rerank_pipeline():
    print(f"INFO: Starting BM25 + ColBERT Re-ranking pipeline. Using Google Drive: {USE_GOOGLE_DRIVE}")
    display_data_dir = DATA_DIR.resolve() if not str(DATA_DIR).startswith("/content/drive") else DATA_DIR
    display_output_dir = OUTPUT_DIR.resolve() if not str(OUTPUT_DIR).startswith("/content/drive") else OUTPUT_DIR
    print(f"INFO: Data is expected in: {display_data_dir}")
    print(f"INFO: Output will be generated in: {display_output_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path_2024_11 = OUTPUT_DIR / "2024-11"
    path_2025_01 = OUTPUT_DIR / "2025-01"
    path_2024_11.mkdir(parents=True, exist_ok=True)
    path_2025_01.mkdir(parents=True, exist_ok=True)

    COLBERT_MODEL_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    print(f"INFO: ColBERT model cache path: {COLBERT_MODEL_CACHE_PATH}")

    if USE_GOOGLE_DRIVE:
        if 'drive' not in globals():
            print("CRITICAL: Google Drive module ('drive') not available. Halting pipeline.")
            return
        if not mount_drive_and_verify_paths(DATA_DIR, QUERIES_FILE_2024_11, QUERIES_FILE_2025_01, DOCUMENTS_DIR):
            print("CRITICAL: Google Drive Path verification or mount failed. Halting pipeline.")
            return
    else:
        if not verify_local_paths(DATA_DIR, QUERIES_FILE_2024_11, QUERIES_FILE_2025_01, DOCUMENTS_DIR):
            print("CRITICAL: Local Path verification failed. Halting pipeline.")
            return

    print("INFO: Loading and preparing all documents...")
    doc_id_to_text_map, corpus_texts_for_bm25, corpus_doc_ids_for_bm25 = load_and_prepare_documents(DOCUMENTS_DIR,
                                                                                                    batch_size_info_log=50000)

    if not corpus_texts_for_bm25:
        print("ERROR: No documents were loaded. Halting pipeline.")
        return
    print(f"INFO: Total documents loaded: {len(corpus_texts_for_bm25)}")

    print("INFO: Tokenizing corpus for BM25 (this might take a while)...")
    start_bm25_setup_time = time.time()
    tokenized_corpus_for_bm25 = [bm25_tokenizer(doc_text) for doc_text in
                                 tqdm(corpus_texts_for_bm25, desc="Tokenizing for BM25")]
    bm25_model = BM25Okapi(tokenized_corpus_for_bm25, k1=BM25_K1, b=BM25_B)
    end_bm25_setup_time = time.time()
    print(f"INFO: BM25 setup and tokenization completed in {end_bm25_setup_time - start_bm25_setup_time:.2f} seconds.")

    print(f"INFO: Initializing ColBERT model for re-ranking: {COLBERT_MODEL_NAME}")
    colbert_reranker = None
    try:
        colbert_reranker = RAGPretrainedModel.from_pretrained(
            COLBERT_MODEL_NAME,
            index_root=str(COLBERT_MODEL_CACHE_PATH)
        )
        if torch.cuda.is_available():
            device = torch.device("cuda")
            if hasattr(colbert_reranker, 'model') and hasattr(colbert_reranker.model, 'model') and isinstance(
                    colbert_reranker.model.model, torch.nn.Module):
                colbert_reranker.model.model.to(device)
                print("INFO: ColBERT re-ranker model moved to CUDA.")
            else:
                print(
                    "WARNING: Could not move ColBERT model to CUDA. Structure might have changed or model not loaded correctly.")
        else:
            print("INFO: CUDA not available, ColBERT model will run on CPU.")

    except Exception as e:
        print(f"ERROR: Failed to load ColBERT model for re-ranking: {e}")
        import traceback
        traceback.print_exc()
        return
    if colbert_reranker is None:
        print("ERROR: colbert_reranker is None after attempting to load. Halting.")
        return
    print(f"INFO: ColBERT model {COLBERT_MODEL_NAME} initialized for re-ranking.")

    query_sets = {
        "2024-11": {"file": QUERIES_FILE_2024_11, "output_path": path_2024_11,
                    "run_results": collections.defaultdict(dict)},
        "2025-01": {"file": QUERIES_FILE_2025_01, "output_path": path_2025_01,
                    "run_results": collections.defaultdict(dict)},
    }

    for setName, setData in query_sets.items():
        print(f"\nINFO: Processing query set: {setName}")
        queries = load_queries(setData["file"])
        if not queries:
            print(f"WARNING: No queries loaded for {setName}, skipping this set.")
            continue

        num_processed_queries = 0
        total_search_time_ms = 0

        for q_id, q_text in tqdm(queries.items(), desc=f"Processing {setName} queries", unit="query"):
            query_start_time = time.time()
            try:
                tokenized_query = bm25_tokenizer(q_text)
                doc_scores_bm25_all = bm25_model.get_scores(tokenized_query)

                scored_doc_indices = []
                for i in range(len(doc_scores_bm25_all)):
                    if doc_scores_bm25_all[i] > 0:
                        scored_doc_indices.append((doc_scores_bm25_all[i], i))

                scored_doc_indices.sort(key=lambda x: x[0], reverse=True)

                bm25_candidate_original_ids = []
                bm25_candidate_texts = []

                for score, doc_idx in scored_doc_indices[:BM25_TOP_K_CANDIDATES]:
                    original_doc_id = corpus_doc_ids_for_bm25[doc_idx]
                    bm25_candidate_original_ids.append(original_doc_id)
                    bm25_candidate_texts.append(doc_id_to_text_map[original_doc_id])

                if not bm25_candidate_texts:
                    setData["run_results"][str(q_id)] = {}
                    num_processed_queries += 1
                    continue

                effective_k_for_rerank = min(K_RETRIEVAL, len(bm25_candidate_texts))

                if effective_k_for_rerank == 0:
                    setData["run_results"][str(q_id)] = {}
                    num_processed_queries += 1
                    continue

                colbert_reranked_results = colbert_reranker.rerank(
                    query=q_text,
                    documents=bm25_candidate_texts,
                    k=effective_k_for_rerank
                )

                if colbert_reranked_results is None:
                    print(
                        f"WARNING: colbert_reranker.rerank returned None for QID {q_id} with k={effective_k_for_rerank} and {len(bm25_candidate_texts)} candidates. Skipping.")
                    setData["run_results"][str(q_id)] = {}
                    num_processed_queries += 1
                    continue

                for res in colbert_reranked_results:
                    original_doc_id_for_this_res = bm25_candidate_original_ids[res['result_index']]
                    setData["run_results"][str(q_id)][str(original_doc_id_for_this_res)] = float(res['score'])

                num_processed_queries += 1
                query_end_time = time.time()
                total_search_time_ms += (query_end_time - query_start_time) * 1000

                if num_processed_queries % 20 == 0 and num_processed_queries < len(queries):
                    avg_time_per_query = total_search_time_ms / num_processed_queries if num_processed_queries > 0 else 0
                    print(
                        f"INFO: Processed {num_processed_queries}/{len(queries)} queries for {setName}. Avg time/query: {avg_time_per_query:.2f} ms.")

            except Exception as e:
                print(f"ERROR: Error processing query ID {q_id} ('{q_text[:50]}...') for {setName}: {e}")
                import traceback
                traceback.print_exc()

        avg_time_per_query_final = total_search_time_ms / num_processed_queries if num_processed_queries > 0 else 0
        print(
            f"INFO: Finished processing for {setName}. Processed {num_processed_queries} queries. Avg time/query: {avg_time_per_query_final:.2f} ms.")

        run_file_path = setData["output_path"] / "run.txt"
        generate_trec_run_file(setData["run_results"], run_file_path, RUN_TAG)

        gzipped_run_file_path = setData["output_path"] / "run.txt.gz"
        display_set_output_path = setData['output_path'].resolve() if not str(setData['output_path']).startswith(
            "/content/drive") else setData['output_path']
        print(f"INFO: Compressing {run_file_path.name} to {gzipped_run_file_path.name} in {display_set_output_path}...")
        try:
            with open(run_file_path, 'rb') as f_in, gzip.open(gzipped_run_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(run_file_path)
            print(f"INFO: Successfully compressed and removed original run file for {setName}.")
        except Exception as e:
            print(f"ERROR: Error during compression or deletion for {setName} run file: {e}")

    print(f"\nINFO: Two-stage pipeline finished. Submission files generated in: {display_output_dir}")
    print(
        f"INFO: Remember to create the 'ir-metadata.yml' file in '{display_output_dir}' manually if required for submission.")


if __name__ == '__main__':
    if USE_GOOGLE_DRIVE:
        if 'google.colab.drive' not in sys.modules and 'drive' not in globals():
            print("ERROR: USE_GOOGLE_DRIVE is True, but the 'google.colab.drive' module is not available.")
            print("INFO: Please ensure you are in a Colab environment or set USE_GOOGLE_DRIVE to False.")
            sys.exit(1)

    if not USE_GOOGLE_DRIVE:
        if not (Path.cwd() / DATA_DIR_NAME).exists():
            print(
                f"WARNING: The data directory '{DATA_DIR_NAME}' was not found in the current working directory: {Path.cwd()}")
            print(
                f"WARNING: Please ensure the script is run from the directory containing '{DATA_DIR_NAME}' or adjust DATA_DIR_NAME path if it's elsewhere.")

    if torch.cuda.is_available():
        print(f"INFO: CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available. ColBERT re-ranking will be very slow on CPU.")

    pipeline_start_time = time.time()
    main_bm25_colbert_rerank_pipeline()
    pipeline_end_time = time.time()
    print(f"INFO: Total pipeline execution time: {pipeline_end_time - pipeline_start_time:.2f} seconds.")