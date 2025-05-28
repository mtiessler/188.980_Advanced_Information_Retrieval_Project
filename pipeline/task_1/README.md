
# Traditional IR - BM25 with LongEval 2025 SciRetrieval Dataset

## Overview

This "Traditional IR" sub-project implements an information retrieval pipeline for the LongEval 2025 Task 2: SciRetrieval challenge, focusing on scientific article retrieval using the CORE dataset. 

For this approach, BM25 using the `rank_bm25` library with optimizations for memory usage (token caching) is implemented. 

## Dataset

This project uses the **LongEval 2025 CORE Retrieval Train Collection**. It consists of queries, scientific documents (provided in JSONL format with fields like `id`, `title`, `abstract`, and optionally `fulltext`), and relevance judgments (`qrels.txt` derived from click models).

The pipeline can be configured to use either the abstract-only version or the full-text version of the documents. Also a demo-mode is available which can use only a 

## Implemented Pipeline

BM25 (rank_bm25 + Caching):

* Uses the `rank_bm25` library.
* Implements NLTK-based preprocessing (tokenization, optional stopword removal, optional Snowball stemming via `PyStemmer`).
* To handle large dataset memory constraints, it tokenizes documents in chunks, caches tokens to disk (`.pkl`), loads all tokens into memory, and then initializes the `BM25Okapi` index.
* Performs standard BM25 search.


## Project Structure

```
.
├── cache/                  # Directory for cached embeddings, indexes, tokens (auto-created)
├── BM25_main.py            # Main execution script for training data,
├── BM25_main_submit.py     # Main execution script for submission, dataset
├── BM25_pipline_exec.ipynb # Notebook for executing the whole process,
├── config.py               # Main configuration (paths, models, params, run mode)
├── data_loader.py          # Functions for loading data (structure, stream, queries, qrels)
├── evaluation.py           # Evaluation function using ir_measures
├── gridsearch_main.py      # BM25 Parameter tuning gridsearch,
├── ir-metadata.yml         # YAML file template for submission,
├── longeval_runs/          # Directory for output run files (auto-created)
├── pipeline.py             # Orchestrator class (RetrievalPipeline)
├── preprocessing.py        # Text preprocessing class (Preprocessor)
├── requirements.txt        # Python dependencies
├── retrievers/
│   ├── __init__.py
│   ├── bm25_rank_retriever.py # BM25 implementation using implementation
├── submissions/            # Directory for submission files for CLEF competition (auto-created),
└── utils.py                # Utility functions (e.g., saving runs)
```

## Setup

### Prerequisites

* Python 3.9+
* Access to the LongEval 2025 SciRetrieval datasets.
* `pip` for installing packages.
* (Optional) A virtual environment (`venv`, `conda`).

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

* **Preprocessing:**
    * `REMOVE_STOPWORDS`: `True` or `False` for BM25 preprocessing.
    * `ENABLE_STEMMING`: `True` or `False` for BM25 preprocessing (uses PyStemmer/Snowball).
* **Parameters:** Adjust `k1`, `b`, `*_TOP_K`, `*_BATCH_SIZE`, `EVALUATION_MEASURE` etc. as needed.
* **Device:** `DEVICE` automatically selects `cuda` if available, else `cpu`.

## Execution

### Local Execution or Jupyter Hub Execution

1.  Ensure Prerequisites, Installation, Data Placement, and Configuration in `config.py` are done.
2.  Activate your virtual environment (if used).
3.  Open the notebook `BM25_pipline_exec.ipynb` and follow additional instructions step-by-step.
4.  Monitor the output for progress and logs. Expect long runtimes for the full dataset, especially during preprocessing. 

### Submission
The notebook also includes the preparing steps for submitting the results to the CLEF competition. This excludes the evaluation of the ranking results. 

## Output

* **Run Files:** Standard TREC-format run files (`run_BM25_*.txt`) will be saved in the directory specified by `OUTPUT_DIR`.
* **Cache Files:** Intermediate files are saved in `CACHE_DIR` to speed up subsequent runs:
    * BM25 Tokens: `bm25_tokens_cache_....pkl`
* **Logs:** Detailed execution logs are printed to the console output and saved to `*.log` files.


## Additional Content

### BM25 Parameter k1, b

For finding usefull values for the BM25 parameters k1 and b a semi automated grid-search was implemented (`gridsearch_main.py`). Because of long runtimes, maximum 3x3 values per grid search were configured. Following the most promissing direction of the result, the next grid was defined. Details on the results are documented in our project report.

### More Details

For more detailed information, please read project report paper. 





