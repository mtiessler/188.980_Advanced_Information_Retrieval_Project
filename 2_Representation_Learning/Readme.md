**1. Directory Structure**


```
longeval-retrieval-project/
│
├── data/                               <-- Create this directory
│   └── longeval_sci_training_2025_abstract/  <-- Place UNZIPPED data here
│       ├── documents/
│       │   ├── documents_000001.jsonl
│       │   └── ... (other document files)
│       ├── qrels.txt
│       └── queries.txt
│   # Or instead: longeval_sci_training_2025_fulltext/
│
├── output/                             <-- Create this (or script will)
│   ├── doc_embeddings.index            <-- Will be generated
│   ├── doc_ids.pkl                     <-- Will be generated
│   ├── fine_tuned_scibert/             <-- Will be generated if fine-tuning
│   │   └── ... (model files)           <-- Will be generated if fine-tuning
│   ├── run_2024-11.txt                 <-- Will be generated (example)
│   └── run_2024-12.txt                 <-- Will be generated (example)
│
├── abstract_config.yaml                <-- The abstract dataset config file
├── requirements.txt                    <-- Python libraries needed
├── utils.py                            <-- Utility functions (config loader)
├── data_loader.py                      <-- Data loading code
├── embedder.py                         <-- Embedding generation code
├── indexer.py                          <-- FAISS indexing code
├── trainer.py                          <-- Fine-tuning code (optional)
├── retriever.py                        <-- Retrieval/search code
├── evaluate.py                         <-- Evaluation code
└── main.py                             <-- Main pipeline script
```

**2. How to Configure Everything (`config.yaml`)**

Edit the `config.yaml` file to control the pipeline's behavior.

* **`data_dir`**: Set this to the *exact path* where you placed the unzipped data folder (e.g., `./data/longeval_sci_training_2025_abstract`). Make sure the `queries_file_name`, `qrels_file_name`, and `documents_dir_name` match the names inside that folder.
* **`output_dir`**: Specifies where generated files (index, model, results) will be saved. `./output/` is usually fine.
* **`model_name_or_path`**: Initially, set this to the base model (`allenai/scibert_scivocab_uncased`). If you successfully fine-tune, you can later change this to the path where the fine-tuned model was saved (e.g., `output/fine_tuned_scibert`).
* **`max_seq_length`**: Keep at 512 for standard SciBERT.
* **`device`**: Set to `"cuda"` if you have a compatible GPU and PyTorch/FAISS installed, otherwise `"cpu"` (will be very slow).
* **`doc_content_fields`**: Use `["title", "abstract"]` for the abstract dataset. If attempting full-text, change to `["title", "abstract", "fulltext"]` (and ensure `embedder.py` handles long text appropriately, e.g., with chunking logic).
* **`embedding_batch_size`**: Adjust based on GPU memory. Larger batches are faster but use more memory (e.g., 32, 64, 128).
* **`faiss_index_type`**: `"IndexFlatIP"` + `normalize_embeddings: true` is standard for cosine similarity search with BERT embeddings.
* **`do_finetune`**: Set to `true` to run the fine-tuning step, `false` to skip it.
* **`st_model_output_path`**: Where the fine-tuned model gets saved if `do_finetune` is true.
* **Fine-tuning Params (`st_*`)**: Control batch size, epochs, learning rate for fine-tuning. **`st_train_samples_max`**: Set this to a small number (e.g., `1000`) for initial *testing* of the fine-tuning code, then set to `-1` to use all data for actual training.
* **`top_k`**: How many results to retrieve per query.
* **`evaluation_snapshots`**: **Crucial:** This list **must contain the exact snapshot IDs** (like `"2024-11"`, `"2024-12"`) that exist in your `qrels.txt` file and that you want to generate run files for and evaluate. The placeholder logic in `data_loader.py` currently derives these from the qrels file, but double-check this matches the task requirements.

**3. How to Start Testing (Step-by-Step)**

It's best to test components first.

* **Step 0: Setup**
    1.  Create the directory structure shown above.
    2.  Place the Python files inside.
    3.  Place the *unzipped* data folder inside the `data/` directory (or adjust `data_dir` in `config.yaml`).
    4.  Create a Python virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```
    5.  Install requirements: `pip install -r requirements.txt`.
    6.  Basic Config Check: Run `python utils.py`. It should print "Configuration loaded..." without errors.

* **Step 1: Test Data Loading**
    1.  Run `python data_loader.py`.
    2.  Check the output: Does it report loading queries, qrels, and documents? Does the number of documents look reasonable? Does it find snapshot IDs?
    3.  Fix any `FileNotFoundError` by correcting paths in `config.yaml`. Debug any JSON parsing errors (might indicate corrupt data lines).

* **Step 2: Test Embedding & Indexing (Small Scale)**
    1.  **Modify:** Temporarily change `data_loader.py` to load only a small number of documents (e.g., add `if len(documents_content) >= 1000: break` inside the loading loop) OR modify `main.py` to pass a subset of `documents_content` to the embedder.
    2.  **Configure:** Ensure `do_finetune: false` in `config.yaml`.
    3.  **Run:** Execute `python main.py`. Focus on the logging output related to "Generating document embeddings..." and "Building FAISS index...".
    4.  **Check:** Did it complete without crashing? Check the `output/` folder for `doc_embeddings.index` and `doc_ids.pkl`. Monitor GPU memory/usage if using CUDA.
    5.  **Debug:** Address errors in `embedder.py` (model loading, tokenization, embedding) or `indexer.py` (FAISS issues).
    6.  **Full Scale:** Once working on a small scale, remove the modification from step 2.1 and run `python main.py` again. This will generate embeddings for *all* documents and will take significant time.

* **Step 3: Test Retrieval**
    1.  **Prerequisite:** Ensure `doc_embeddings.index` and `doc_ids.pkl` exist in `output/` from the previous step.
    2.  **Run:** You can test retrieval in isolation by running `python retriever.py`. It includes example usage in its `if __name__ == "__main__":` block.
    3.  **Check:** Does it load the index? Does it perform a search for the example query? Does the output format look like the TREC run format?
    4.  **Integrate:** Run `python main.py` again (ensure embedding step is skipped as files exist). Focus on the "Retrieving for snapshot..." logs. Check if `run_*.txt` files are created in `output/`.
    5.  **Debug:** Issues might be in loading the index, query embedding, the FAISS search call, or the filtering logic (`allowed_doc_ids`).

* **Step 4: Test Fine-tuning (Small Scale)**
    1.  **Configure:** Set `do_finetune: true`. **Crucially, set `st_train_samples_max: 1000` (or similar small number) and `st_num_epochs: 1`**.
    2.  **Run:** Execute `python main.py`.
    3.  **Check:** Monitor the training progress bar (if using `sentence-transformers`). Does the loss decrease? Does it complete without errors? Check `output/fine_tuned_scibert/` for saved model files.
    4.  **Debug:** Errors often occur in `prepare_training_data` (check positive/negative sampling logic) or during the `model.fit()` call (GPU memory errors, input format issues).
    5.  **Full Scale:** Once the mechanics work, set `st_train_samples_max: -1` and increase `st_num_epochs` for actual training.

* **Step 5: Test Evaluation**
    1.  **Prerequisite:** Ensure `run_*.txt` files exist in `output/` for the snapshots listed in `config.yaml -> evaluation_snapshots`.
    2.  **Run:** Execute `python main.py`. Focus on the "Starting evaluation..." logs.
    3.  **Check:** Does it print evaluation scores (nDCG, MAP, etc.) for each snapshot?
    4.  **Debug:** Errors might be in loading the run files (`evaluate.py -> prepare_ranx_run`), preparing the qrels (`prepare_ranx_qrels`), or the `ranx.evaluate` call itself (mismatched query IDs between run and qrels). Ensure the snapshot IDs in `config.yaml` exactly match those in your `qrels.txt`.

Start with Step 0 and proceed sequentially.