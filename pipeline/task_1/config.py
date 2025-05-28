import torch
import os

# --- Run Mode ---
IS_DEMO_MODE = False # Set to False FOR FULL RUN
DEMO_FILES_LIMIT = 1 # Ignored when IS_DEMO_MODE is False

# --- Paths (For Colab) ---
#DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/AIR_Project"

# --- Paths (For Jupyter Hub) ---
DRIVE_PROJECT_ROOT = "/home/jovyan/AIR/Offline_188.980_Advanced_Information_Retrieval_Project"

#for TRAIN Dataset
DRIVE_PROJECT_ROOT_DATASET = "/home/jovyan/shared/188.980-AIR-2025S/LongEval-Sci"
#for TEST Dataset
#DRIVE_PROJECT_ROOT_DATASET = "/home/jovyan/sharedTEST/188.980-AIR-2025S/LongEval-Sci"
#for Collab
#DRIVE_PROJECT_ROOT_DATASET = "/content/drive/MyDrive/AIR_Project"

# -- Paths (For Local) ---
# DRIVE_PROJECT_ROOT = ""
# DRIVE_PROJECT_ROOT_DATASET = "../LongEval-Sci"
# --- Core Paths ---

#for TRAIN Dataset
BASE_DIR = os.path.join(DRIVE_PROJECT_ROOT_DATASET, "longeval_sci_training_2025_abstract")
#BASE_DIR = os.path.join(DRIVE_PROJECT_ROOT_DATASET, "longeval_sci_training_2025_fulltext")

#for TEST Dataset
#BASE_DIR = os.path.join(DRIVE_PROJECT_ROOT_DATASET, "longeval_sci_testing_2025_abstract")
#BASE_DIR = os.path.join(DRIVE_PROJECT_ROOT_DATASET, "longeval_sci_testing_2025_fulltext")

PROJECT_CODE_DIR = os.path.join(DRIVE_PROJECT_ROOT, "pipeline/task_1")
OUTPUT_DIR = os.path.join(PROJECT_CODE_DIR, "longeval_runs")
CACHE_DIR = os.path.join(PROJECT_CODE_DIR, "cache")

DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
QUERIES_FILE = os.path.join(BASE_DIR, "queries.txt") #queries_2024-11_test
QRELS_FILE = os.path.join(BASE_DIR, "qrels.txt")

# --- Paths FOR SUBMISSION (For Jupyter Hub) ---
# DRIVE_PROJECT_ROOT_DATASET_SUBMISSON = "/home/jovyan/sharedTEST/188.980-AIR-2025S/LongEval-Sci"

OUTPUT_DIR_SUBMISSON = os.path.join(PROJECT_CODE_DIR, "submissions")
QUERIES_FILE_SUBMISSON_T1 = os.path.join(BASE_DIR, "queries_2024-11_test.txt")
QUERIES_FILE_SUBMISSON_T2 = os.path.join(BASE_DIR, "queries_2025-01_test.txt")


# --- Added for rank_bm25 token caching ---
BM25_TOKEN_CHUNK_SIZE = 5000 # 50000
BM25_TOKEN_CACHE_FILE = os.path.join(CACHE_DIR, f"bm25_tokens_cache_{os.path.basename(BASE_DIR)}.pkl")

# --- Device ---
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Names (we need to explore alternatives) ---
#CROSS_ENCODER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# --- Preprocessing ---
REMOVE_STOPWORDS = True
ENABLE_STEMMING = True

# --- BM25 Parameters ---
# Best results abstract-dataset: k1=1.0, b=0.7
# Best results fulltext-dataset: k1=0.2, b=0.8
# inital value: k1=0.95, b=0.75
BM25_K1 = 0.2
BM25_B = 0.8
BM25_TOP_K = 1000

RERANK_BATCH_SIZE = 32
FINAL_TOP_K = 100

EVALUATION_MEASURES = {"nDCG@10", "MAP@100", "MRR@10"}

CONTENT_FIELD = 'abstract' # Select field for document text ('abstract' or 'fullText')
