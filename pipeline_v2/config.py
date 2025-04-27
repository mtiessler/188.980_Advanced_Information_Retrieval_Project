import torch
import os

# --- Run Mode ---
IS_DEMO_MODE = False # Set to False FOR FULL RUN
DEMO_FILES_LIMIT = 1 # Ignored when IS_DEMO_MODE is False

# --- Drive Path (For Colab) ---
# DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/AIR_Project"

# --- Paths (For Jupyter Hub) ---
DRIVE_PROJECT_ROOT = "/home/jovyan/AIR/188.980_Advanced_Information_Retrieval_Project"
DRIVE_PROJECT_ROOT_DATASET = "/home/jovyan/shared/188.980-AIR-2025S/LongEval-Sci"

# --- Core Paths ---
BASE_DIR = os.path.join(DRIVE_PROJECT_ROOT_DATASET, "longeval_sci_training_2025_abstract")
PROJECT_CODE_DIR = os.path.join(DRIVE_PROJECT_ROOT, "pipeline_v2")
OUTPUT_DIR = os.path.join(PROJECT_CODE_DIR, "longeval_runs")
CACHE_DIR = os.path.join(PROJECT_CODE_DIR, "cache")

DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
QUERIES_FILE = os.path.join(BASE_DIR, "queries.txt")
QRELS_FILE = os.path.join(BASE_DIR, "qrels.txt")

# --- Added for rank_bm25 token caching ---
BM25_TOKEN_CHUNK_SIZE = 50000
BM25_TOKEN_CACHE_FILE = os.path.join(CACHE_DIR, f"bm25_tokens_cache_{os.path.basename(BASE_DIR)}.pkl")

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Names (we need to explore alternatives) ---
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# --- Preprocessing ---
REMOVE_STOPWORDS = True
ENABLE_STEMMING = True

# --- BM25 Parameters ---
BM25_K1 = 1.2 # 1.2-1.9
BM25_B = 0.7
BM25_TOP_K = 1000

# --- Dense Retrieval / FAISS ---
EMBEDDING_BATCH_SIZE = 64 # Adjust based on Colab GPU memory if needed
FAISS_TOP_K = 1000 # Candidates for re-ranking
FORCE_REGENERATE_EMBEDDINGS = False # Set to True if you want to force regeneration even if cache exists for the full run

# --- Re-ranking ---
RERANK_BATCH_SIZE = 32
FINAL_TOP_K = 100

# --- Evaluation ---
EVALUATION_MEASURE = "nDCG@10"

# --- Data Loading ---

CONTENT_FIELD = 'abstract' # Select field for document text ('abstract' or 'fulltext')