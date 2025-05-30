
# LongEval 2025 SciRetrieval Pipeline (AIR Course Project)

## Overview

This project implements and compares three different information retrieval pipelines for the LongEval 2025 Task 2: SciRetrieval challenge, focusing on scientific article retrieval using the CORE dataset. The goal is to evaluate different approaches as required by the TU Wien Advanced Information Retrieval (188.980) course (SS 2025).

The implemented pipelines are:
1.  **Traditional IR Model:** BM25 using the `rank_bm25` library with optimizations for memory usage (token caching).
2.  **Representation Learning Model:** Dense retrieval using a fine-tuned SciBERT (`allenai/scibert_scivocab_uncased`) and a pre-trained ColBERTv2 (`colbert-ir/colbertv2.0`)
3.  **Neural Re-ranking Model:** A hybrid two-stage re-ranking architecture, aiming to combine the strengths of lexical recall from
BM25 with the semantic precision of ColBERTv2 (`colbert-ir/colbertv2.0`)

## Dataset

This project uses the **LongEval 2025 CORE Retrieval Train Collection**. It consists of queries, scientific documents (provided in JSONL format with fields like `id`, `title`, `abstract`, and optionally `fulltext`), and relevance judgments (`qrels.txt` derived from click models).


## Implemented Pipelines

1.  **Task1 - BM25 (rank_bm25 + Caching):**
    * Uses the `rank_bm25` library.
    * Implements NLTK-based preprocessing (tokenization, optional stopword removal, optional Snowball stemming via `PyStemmer`).
    * To handle large dataset memory constraints, it tokenizes documents in chunks, caches tokens to disk (`.pkl`), loads all tokens into memory, and then initializes the `BM25Okapi` index.
    * Performs standard BM25 search.

2.  **Task2 - Dense Retrieval (SciBERT):**
    * Fine-tuned as a regression-based relevance classifier using the `allenai/scibert_scivocab_uncased model`.
    * Inputs are constructed by concatenating the query and document abstract with [SEP] tokens.
    * Sequences are tokenized with the SciBERT tokenizer, padded/truncated to 512 tokens (BERT limit).
    * Relevance judgments (qrels) were split 90/10 into training and evaluation sets using stratified sampling by query ID for fine-tuning

3.  **Task2 - Dense Retrieval (ColBERT):**
    * Uses the `colbert-ir/colbertv2.0` model via the RAGatouille library for late interaction semantic retrieval.
    * Indexes token-level embeddings of abstracts, with structured input: title [SEP] authors [SEP] abstract.
    * Queries are independently encoded, and retrieval uses `MaxSim` token-level similarity for ranking.

4.  **Task3 - Hybrid Re-ranking (BM25 + ColBERTv2):**
    * Retrieves an initial set of candidate documents using both the BM25 model (Top-K) and the FAISS index (Top-K).
    * Combines the candidate sets.
    * Uses a pre-trained model (`colbert-ir/colbertv2.0`) to re-score the candidate query-document pairs based on deeper semantic interaction.
    * Ranks the final results based on the ColBERT scores.

## Project Structure

```
.
pipeline/
├── task_1/               # Lexical BM25 pipeline 
│   ├── BM25_main.py      # Main execution script for training data
│   ├── BM25_main_submit.py # Main execution script for submission, dataset
│   ├── data_loader.py    # Loads and preprocesses datasets
│   ├── config.py         # Main configuration (paths, models, params, run mode)
│   ├── evaluation.py     # Evaluation function using ir_measures
│   ├── gridsearch_main.py# BM25 Parameter tuning gridsearch
│   ├── preprocessing.py  # Text preprocessing class (Preprocessor)
│   ├── pipeline.py       # Orchestrator class (RetrievalPipeline)
│   ├── utils.py          # Utility functions
│   ├── FinalSubmit/      # Final submissions done for BM25
│   │   └── BM25_v001_k1_0p95_b_0p75
│   │   ├── BM25_v002_k1_0p2_b_0p8
│   │   └── BM25_v003_k1_1p0_b_0p7
│   ├── longeval_runs/    # Directory for output run files (auto-created)
│   ├── retrievers/
│   │   ├── __init__.py
│   │   └── bm25_rank_retriever.py        # BM25 retrieval class using rank_bm25
│   ├── submissions/      # Directory for submission files for CLEF competition (auto-created)
│   ├── README.md
│   ├── requirements.txt  # Python dependencies
│   ├── ir-metadata.yml   # YAML file template for submission
│   │
├── task_2/               # Neural models (SciBERT, ColBERTv2)
│   ├── BERT_v1.ipynb     # SciBERT fine-tuning notebook
│   ├── ColBERT_v2.ipynb  # RAGatouille-ColBERTv2 notebook
│   │
├── task_3/               # Hybrid: BM25 + ColBERTv2 re-ranking
│   ├── evaluation.ipynb
│   ├── exec_and_uploader.ipynb  # Execution for submission files
├── __init__.py
deprecated/           # Initial pipeline attempts, experiments done
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
3.  **Install Dependencies for Task1:** Modify `requirements.txt` to include either `faiss-cpu` or `faiss-gpu` based on your hardware, then install:
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

## Configuration (`task_1\config.py`)

This file controls all major aspects of the pipeline. Edit this file before running:

* **Run Mode:**
    * `IS_DEMO_MODE`: Set `True` for a quick test run on a limited number of document files (see `DEMO_FILES_LIMIT`), `False` for the full run.
    * `DEMO_FILES_LIMIT`: Number of `.jsonl` files to process in demo mode.
* **Paths:**
    * `DRIVE_PROJECT_ROOT`: (For Colab primarily) Set the root path on Drive where the project and data reside.
    * `BASE_DIR`: **Must** point to the root of your dataset directory (e.g., `.../longeval_sci_training_2025_abstract`).
    * `PROJECT_CODE_DIR`: Path to the directory containing `BM25_main.py`.
    * `OUTPUT_DIR`, `CACHE_DIR`: Locations for saving run files and caches (indexes, embeddings). **Ensure these are writable and persistent (e.g., on Google Drive for Colab).**
* **Preprocessing:**
    * `REMOVE_STOPWORDS`: `True` or `False` for BM25 preprocessing.
    * `ENABLE_STEMMING`: `True` or `False` for BM25 preprocessing (uses PyStemmer/Snowball).
* **Parameters:** Adjust `k1`, `b`, `*_TOP_K`, `*_BATCH_SIZE`, `EVALUATION_MEASURE` etc. as needed.
* **Device:** `DEVICE` automatically selects `cuda` if available, else `cpu`.

## Execution

### Task 1 - Traditional IR Model

1.  Ensure Prerequisites, Installation, Data Placement, and Configuration in `task_1/config.py` are done.
2.  Open the notebook task_1/BM25_pipline_exec.ipynb and follow additional instructions step-by-step to generate ir metrics or additionally generate TIRA submission. 
3.  Monitor the output for progress and logs. Expect long runtimes for the full dataset, especially during preprocessing. 

* for some additional information see task_1\ReadMe.

### Task 2 

1. Run notebook task_2/BERT_v1.ipynb for SciBERT implementation
2. Run notebook task_2/ColBERT_v2.ipynb for ColBERTv2 implementation

* All the dependencies are installed and the configurations set in the begining of the notebooks

### Task 3 
1. Run notebook task_3/evaluation.ipynb to execute model and generate ir metrics
2. Run notebook task_3/exec_and_uploader.ipynb to execute model and generate TIRA submission


## Output

* **Run Files:** Standard TREC-format run files (`run_BM25.txt`, `CLEF-ColBERT-Abstract-Final-v1.txt`, `CLEF-Bert-Run-FP16.txt`, `run.txt`)
* **Cache Files:** Intermediate files are saved in `CACHE_DIR` to speed up subsequent runs:
    * BM25 Tokens: `bm25_tokens_cache_....pkl`
    * ColBERT Model Cache, if used by RAGatouille: `colbert_model_cache`
* **Logs:** Detailed execution logs are printed to the console/Colab output.

## Final Results for Abstract Dataset

* BM25 (k1 = 1.0, b = 0.7): `nDCG@10: 0.1778`
* Dense Retrieval - SciBERT (Fine-tuned): `nDCG@10: 0.1896`
* Dense Retrieval - ColBERTv2 (End-to-End, RAGatouille): `nDCG@10: 0.6363`
* Hybrid Re-rank - BM25 (k1 = 1.5, b = 0.75) + ColBERTv2 Re-ranker: `nDCG@10: 0.1316`

## Group members contribution:
* Adrian Bergler: Supported BM25 implementation & tuning; Experimented with BM25F, the integration of the developed BM25 into Hybrid Re-ranking and some other Task 3 tuning; Report
* Christine Hubinger: Lead BM25 implementation (Task1), preprocessing pipeline, and grid search tuning; TIRA submission setup; groupmeeting organization; Report
* Dmytro Pashchenko: Report and documentation
* Margarida Maria Agostinho Gonçalves: Experimented with initial dense retrieval; Report
* Max Tiessler: Developed and trained SciBERT (Task2); Implemented Hybrid Re-ranking (Task3: BM25 + ColBERT); Developed initial pipeline set-up; TIRA submission setup; Report

## Course Context

This project was developed as part of the Advanced Information Retrieval (188.980) course at TU Wien during the Summer Semester 2025. 
It aims to fulfill the exercise requirements by implementing traditional, representation-learning, and neural re-ranking retrieval models.
