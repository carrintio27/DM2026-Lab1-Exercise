from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool("build_dtm_tool")
    def build_dtm_tool(text_column: str = "text", ngram_range: List[int] = [1, 1]) -> dict:
        """
        Builds a Document-Term Matrix (DTM) from session.dataframe using CountVectorizer.

        Args:
            text_column: Name of text column in session.dataframe.
            ngram_range: Lower and upper boundary of n-grams as a list [min_n, max_n].

        Returns:
            Dict containing result_id, vocabulary_size, n_documents, non_zero, total_elements, and sparsity_pct.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Call load_dataset_tool first."}

        df = session.dataframe
        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in dataset."}

        # Validate ngram_range parameter strictly (must be list/tuple of 2 integers)
        if not isinstance(ngram_range, (list, tuple)) or len(ngram_range) != 2 or not isinstance(ngram_range[0], int) or not isinstance(ngram_range[1], int):
            return {"error": "ngram_range must be a list of two integers, e.g. [1, 1]."}

        min_n, max_n = ngram_range[0], ngram_range[1]
        if min_n <= 0 or max_n <= 0 or min_n > max_n:
            return {"error": f"Invalid ngram_range [{min_n}, {max_n}]: min_n must be > 0 and <= max_n."}

        ng_tuple = (min_n, max_n)

        # Handle missing text cleanly
        text_data = df[text_column].fillna("")

        vectorizer = CountVectorizer(ngram_range=ng_tuple)
        dtm = vectorizer.fit_transform(text_data)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state per contract
        session.feature_matrix = dtm
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        label_col = "category_name" if "category_name" in df.columns else ("label" if "label" in df.columns else None)
        if label_col:
            session.set_labels(df[label_col].values)

        n_rows, n_cols = dtm.shape
        non_zero = int(dtm.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "vocabulary_size": n_cols,
            "n_documents": n_rows,
            "vocabulary": feature_names,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": sparsity_pct
        }

        session.store_result("build_dtm_tool", {"text_column": text_column, "ngram_range": ngram_range}, summary)
        return summary

    @tool("term_frequency_tool")
    def term_frequency_tool() -> dict:
        """
        Calculates the total frequency for each term across all documents in session.feature_matrix.

        Returns:
            Dict containing result_id, total_terms, and term_frequencies dictionary.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Call build_dtm_tool first."}

        dtm = session.feature_matrix
        feature_names = session.feature_names

        # Sum along columns (axis=0) to get total counts per term
        term_sums = dtm.sum(axis=0).A1  # Convert 1xN matrix to 1D array
        term_frequencies = {term: int(count) for term, count in zip(feature_names, term_sums)}

        result_id = session.next_result_id("term_freq")
        summary = {
            "result_id": result_id,
            "total_terms": len(feature_names),
            "term_frequencies": term_frequencies
        }

        session.store_result("term_frequency_tool", {}, summary)
        return summary

    @tool("dtm_heatmap_tool")
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> dict:
        """
        Generates a heatmap visualization and returns a positional slice of the raw Document-Term Matrix.

        Args:
            n_terms: Maximum number of vocabulary terms (columns) to slice from the DTM (default 20).
            n_documents: Maximum number of documents (rows) to slice from the DTM (default 20).

        Returns:
            Dict containing result_id, included_terms, n_documents, and sliced_matrix (2D list).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Call build_dtm_tool first."}

        dtm = session.feature_matrix
        feature_names = session.feature_names

        if n_terms <= 0 or n_documents <= 0:
            return {"error": "n_terms and n_documents must be greater than 0."}

        max_docs, max_terms = dtm.shape
        actual_docs = min(n_documents, max_docs)
        actual_terms = min(n_terms, max_terms)

        sliced_sparse = dtm[0:actual_docs, 0:actual_terms]
        dense_array = sliced_sparse.toarray()
        sliced_matrix = dense_array.tolist()
        included_terms = feature_names[0:actual_terms]

        # Generate Seaborn heatmap figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            dense_array,
            annot=True,
            fmt="d",
            xticklabels=included_terms,
            yticklabels=[f"Doc {i}" for i in range(actual_docs)],
            cmap="YlGnBu",
            ax=ax
        )
        ax.set_title(f"DTM Heatmap ({actual_docs} Docs x {actual_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        fig.tight_layout()

        # Assign figure directly to session.pending_figure per plotting requirement
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents": actual_docs,
            "n_terms": actual_terms,
            "included_terms": included_terms,
            "sliced_matrix": sliced_matrix
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
