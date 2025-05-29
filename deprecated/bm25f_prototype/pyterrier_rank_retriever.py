import logging
import os
import pyterrier as pt

"""
This file contains an experimental rank retriever implementation for BM25F using PyTerrier.
It is intended to be used instead of our BM25RankRetriever (and hence should be moved to the same package in case
experimentation with BM25F is continued).
It might contain (several) bugs, since it was not fully run due to the environment issues caused by PyTerrier requiring 
access to a JDK, that made us decide to stop further experimentation considering the time constraints.
"""

# Requires python-terrier>=0.13.0 (add to requirements.txt or similar if this class is used)
class PyTerrierRankRetriever:
    def __init__(
        self,
        preprocessor,
        index_path,
        field_weights=None,    # e.g. {"title": 2.0, "abstract": 1.0, "fulltext": 0.8}  (BM25F)
        num_results=1000,
    ):
        if not pt.started():           # start JVM
            pt.init()
        self.pre = preprocessor
        self.index_path = index_path
        self.field_weights = field_weights or {}
        self.num_results = num_results
        self._retriever = None         # will hold pt.terrier.Retriever
        logging.info("Initialized PyTerrierRankRetriever")

    def _iter_docs(self, doc_it):
        """
        Convert loader’s per-doc dict into the shape Terrier expects.
        Copy only the fields mentioned in self.field_weights to avoid storing unused text.
        """
        wanted = list(self.field_weights.keys())  # e.g. ['title','abstract','fulltext']
        for d in doc_it:
            if not d or "id" not in d:
                continue
            doc = {"docno": d["id"]}
            for fld in wanted:
                # call your existing preprocessor for normalisation
                doc[fld] = self.pre.clean(d.get(fld, ""))
            yield doc

    def index(self, doc_iterator_factory, force_reindex=False):
        """
        Build or load a Terrier index on disk.
        """
        data_props = os.path.join(self.index_path, "data.properties")
        if not force_reindex and os.path.exists(data_props):
            logging.info("Opening existing Terrier index")
            self.index_ref = pt.IndexRef.of(data_props)
        else:
            logging.info("Building Terrier index …")
            os.makedirs(self.index_path, exist_ok=True)

            # ---- 1) pick indexer
            idxr = pt.IterDictIndexer(
                self.index_path,
                meta=["docno"],                # keep document ids
                threads=os.cpu_count(),
                fields=bool(self.field_weights)  # BM25F needs per-field stats
            )
            # Configure Terrier to record fields to allow usage of BM25F
            if self.field_weights:
                idxr.setProperties(indexing_fields="true")

            # ---- 2) stream docs in
            index_ref = idxr.index(self._iter_docs(doc_iterator_factory()))
            self.index_ref = index_ref
            logging.info("Terrier index built")

        # ---- 3) configure Retriever
        controls = {}
        if self.field_weights:
            # Terrier numbers fields in indexing order 0,1,…
            for i, (_, w) in enumerate(self.field_weights.items()):
                controls[f"w.{i}"] = w       # weight
                controls[f"c.{i}"] = 0.3     # normalisation, tune later
            wmodel = "BM25F"
        else:
            wmodel = "BM25"

        self._retriever = pt.terrier.Retriever(
            self.index_ref,
            wmodel=wmodel,
            controls=controls,
            num_results=self.num_results
        )
        logging.info("Retriever ready")
        return True

    def load_or_build_index(self, doc_iterator_factory, force_reindex=False):
        return self.index(doc_iterator_factory, force_reindex)

    def search(self, query_text, k=10):
        if self._retriever is None:
            logging.error("Index not built/loaded.")
            return {}

        # Terrier will tokenise queries itself; you can still pre-clean
        query = self.pre.clean(query_text)
        # use the Retriever directly for a single query -> DataFrame
        hits = self._retriever.transform(
            pt.new.query_df([(1, query)], ["qid", "query"])
        )

        topk = hits.head(k)
        return dict(zip(topk["docno"], topk["score"]))
