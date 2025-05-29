
# LongEval 2025 SciRetrieval Pipeline (AIR Course Project)

## Overview

This project implements and compares three different information retrieval pipelines for the LongEval 2025 Task 2: SciRetrieval challenge, focusing on scientific article retrieval using the CORE dataset. The goal is to evaluate different approaches as required by the TU Wien Advanced Information Retrieval (188.980) course (SS 2025).

The implemented pipelines are:
1.  **Traditional IR Model:** BM25 using the `rank_bm25` library with optimizations for memory usage (token caching).
2.  **Representation Learning Model:** Dense retrieval using SentenceBERT embeddings (`sentence-transformers/all-MiniLM-L6-v2`) indexed and searched with FAISS.
3.  **Neural Re-ranking Model:** A hybrid approach using BM25 and/or FAISS for initial candidate retrieval, followed by re-ranking using a pre-trained Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

## Dataset

This project uses the **LongEval 2025 CORE Retrieval Train Collection**. It consists of queries, scientific documents (provided in JSONL format with fields like `id`, `title`, `abstract`, and optionally `fulltext`), and relevance judgments (`qrels.txt` derived from click models).

The pipeline can be configured to use either the abstract-only version or the full-text version of the documents.

## Implemented Pipelines

1.  **BM25 (rank_bm25 + Caching):**
    * Uses the `rank_bm25` library.
    * Implements NLTK-based preprocessing (tokenization, optional stopword removal, optional Snowball stemming via `PyStemmer`).
    * To handle large dataset memory constraints, it tokenizes documents in chunks, caches tokens to disk (`.pkl`), loads all tokens into memory, and then initializes the `BM25Okapi` index.
    * Performs standard BM25 search.

2.  **Dense Retrieval (SentenceBERT + FAISS):**
    * Uses `sentence-transformers/all-MiniLM-L6-v2` (or other configured model) to generate dense embeddings for documents and queries.
    * Builds a FAISS index (`IndexFlatIP`) for efficient nearest-neighbor search based on cosine similarity (inner product on normalized vectors).
    * Caches document embeddings (`.npy`) and the FAISS index (`.index`) for faster subsequent runs.

3.  **Hybrid Re-ranking (BM25/FAISS + Cross-Encoder):**
    * Retrieves an initial set of candidate documents using both the BM25 model (Top-K) and the FAISS index (Top-K).
    * Combines the candidate sets.
    * Uses a pre-trained Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2` or other configured model) to re-score the candidate query-document pairs based on deeper semantic interaction.
    * Ranks the final results based on the Cross-Encoder scores.

## Project Structure

```
.
├── cache/                    # Directory for cached embeddings, indexes, tokens (auto-created)
├── config.py               # Main configuration (paths, models, params, run mode)
├── data_loader.py          # Functions for loading data (structure, stream, queries, qrels)
├── evaluation.py           # Evaluation function using ir_measures
├── longeval_runs/          # Directory for output run files (auto-created)
├── main.py                 # Main execution script
├── pipeline.py             # Orchestrator class (RetrievalPipeline)
├── preprocessing.py        # Text preprocessing class (Preprocessor)
├── requirements.txt        # Python dependencies
├── retrievers/
│   ├── __init__.py
│   ├── bm25_rank_retriever.py # BM25 implementation using rank_bm25 + caching
│   └── dense_retriever.py    # BERT Embedder and FAISS Index classes
├── rerankers/
│   ├── __init__.py
│   └── bert_reranker.py      # BERT Cross-Encoder implementation
└── utils.py                # Utility functions (e.g., saving runs)
```

## Setup

### Prerequisites

* Python 3.9+
* Access to the LongEval 2025 SciRetrieval dataset.
* `pip` for installing packages.
* (Optional but Recommended) A virtual environment (`venv`, `conda`).
* (Optional but Recommended for Neural Stages) An NVIDIA GPU with CUDA installed compatible with PyTorch and FAISS-GPU.

### Installation

1.  **Clone or Download:** Get the project code.
    ```bash
    # Example if using git
    # git clone <your-repo-url>
    # cd <your-repo-folder>
    ```
2.  **(Optional) Create Virtual Environment:**
    ```bash
    python -m venv env
    source env/bin/activate  # Linux/macOS
    # env\Scripts\activate  # Windows
    ```
3.  **Install Dependencies:** Modify `requirements.txt` to include either `faiss-cpu` or `faiss-gpu` based on your hardware, then install:
    ```bash
    pip install -r requirements.txt
    ```
4.  **NLTK Data:** The `preprocessing.py` script attempts to automatically download required NLTK data (`punkt`, `stopwords`, `punkt_tab`) on first run if not found. Alternatively, you can download them manually:
    ```bash
    python -m nltk.downloader punkt stopwords punkt_tab
    ```

### Data Placement

1.  Unzip the LongEval dataset (e.g., `longeval_sci_training_2025_abstract`).
2.  Place the dataset directory in a location accessible to the project.
3.  **Crucially, update the paths** in `config.py` (`BASE_DIR`, etc.) to point to the correct location of the dataset directory. See Configuration section.

## Configuration (`config.py`)

This file controls all major aspects of the pipeline. Edit this file before running:

* **Run Mode:**
    * `IS_DEMO_MODE`: Set `True` for a quick test run on a limited number of document files (see `DEMO_FILES_LIMIT`), `False` for the full run.
    * `DEMO_FILES_LIMIT`: Number of `.jsonl` files to process in demo mode.
* **Paths:**
    * `DRIVE_PROJECT_ROOT`: (For Colab primarily) Set the root path on Drive where the project and data reside.
    * `BASE_DIR`: **Must** point to the root of your dataset directory (e.g., `.../longeval_sci_training_2025_abstract`).
    * `PROJECT_CODE_DIR`: Path to the directory containing `main.py`.
    * `OUTPUT_DIR`, `CACHE_DIR`: Locations for saving run files and caches (indexes, embeddings). **Ensure these are writable and persistent (e.g., on Google Drive for Colab).**
* **Models:**
    * `EMBEDDING_MODEL_NAME`: Hugging Face model name for sentence embeddings.
    * `CROSS_ENCODER_MODEL_NAME`: Hugging Face model name for the re-ranker.
* **Preprocessing:**
    * `REMOVE_STOPWORDS`: `True` or `False` for BM25 preprocessing.
    * `ENABLE_STEMMING`: `True` or `False` for BM25 preprocessing (uses PyStemmer/Snowball).
* **Parameters:** Adjust `k1`, `b`, `*_TOP_K`, `*_BATCH_SIZE`, `EVALUATION_MEASURE` etc. as needed.
* **Device:** `DEVICE` automatically selects `cuda` if available, else `cpu`.

## Execution

### Local Execution

1.  Ensure Prerequisites, Installation, Data Placement, and Configuration are done.
2.  Activate your virtual environment (if used).
3.  Navigate to the project's code directory (where `main.py` is located) in your terminal.
4.  Run the main script:
    ```bash
    python main.py
    ```
5.  Monitor the console output for progress and logs. Expect long runtimes for the full dataset, especially during indexing and embedding generation.

### Google Colab Execution

1.  **Upload:** Upload your **entire project code folder** and the **dataset folder** to your Google Drive (e.g., into a folder named `AIR_Project`).
2.  **Create Notebook & Set Runtime:** Open a Colab notebook, go to `Runtime` -> `Change runtime type`, select `Python 3`, and choose a `GPU` Hardware accelerator (T4 recommended). *(Colab Pro/Pro+ highly recommended for RAM/runtime limits)*.
3.  **Mount Drive:** Run this cell:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    # Follow auth prompts
    ```
4.  **Configure Paths:** **Edit `config.py` on your Google Drive.** Update `DRIVE_PROJECT_ROOT`, `BASE_DIR`, `PROJECT_CODE_DIR`, `OUTPUT_DIR`, `CACHE_DIR` to use absolute paths starting with `/content/drive/MyDrive/...` pointing to your uploaded folders. Verify `DEVICE = "cuda"`.
5.  **Configure Run Mode:** Edit `config.py` on Drive to set `IS_DEMO_MODE` (`True`/`False`), `REMOVE_STOPWORDS`, `ENABLE_STEMMING`, etc.
6.  **Install Dependencies:** Run this cell:
    ```python
    # Change to your project code directory on Drive
    %cd /content/drive/MyDrive/AIR_Project/pipeline_v2/
    # Install (ensure requirements.txt lists faiss-gpu)
    !pip install -r requirements.txt
    ```
7.  **Run Pipeline:** Run this cell:
    ```python
    # Ensure you are in the project directory (%cd ...)
    !python main.py
    ```
8.  **Monitor & Wait:** Watch cell output. Be prepared for **long runtimes (hours)** for the full dataset. Keep the tab active (free tier) or rely on background execution (Pro+). Monitor RAM/GPU usage.

## Output

* **Run Files:** Standard TREC-format run files (`run_BM25Rank_Baseline.txt`, `run_BERT_Dense_FAISS.txt`, `run_Hybrid_ReRank_BM25Rank.txt`) will be saved in the directory specified by `OUTPUT_DIR`.
* **Cache Files:** Intermediate files are saved in `CACHE_DIR` to speed up subsequent runs:
    * BM25 Tokens: `bm25_tokens_cache_....pkl`
    * Document Embeddings: `doc_embeddings_....npy`, `doc_embeddings_ids_....txt`
    * FAISS Index: `faiss_....index`, `faiss_doc_ids_....txt`
* **Logs:** Detailed execution logs are printed to the console/Colab output.

## Baseline Results (Example from Logs)

*(Note: These results were obtained on the abstract dataset without fine-tuning)*

* BM25 (rank_bm25): `nDCG@10: 0.1474`
* Dense Retrieval (all-MiniLM-L6-v2): `nDCG@10: 0.0483`
* Hybrid Re-rank (ms-marco-MiniLM-L-6-v2): `nDCG@10: 0.1325`

These results highlight the effectiveness of the traditional BM25 baseline and the challenge of applying general-purpose neural models zero-shot to this domain.

## Course Context

This project was developed as part of the Advanced Information Retrieval (188.980) course at TU Wien during the Summer Semester 2025. 
It aims to fulfill the exercise requirements by implementing traditional, representation-learning, and neural re-ranking retrieval models.
