import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_index1: int, doc_index2: int) -> dict:
        """Computes the cosine similarity between two documents in the document-term matrix.

        Args:
            doc_index1: Zero-based row index of the first document.
            doc_index2: Zero-based row index of the second document.

        Returns:
            Dict summary containing doc_index1, doc_index2, cosine_similarity, and result_id.
        """
        if session.feature_matrix is None:
            return {"error": "No document-term matrix found in session. Run build_dtm_tool first."}

        n_docs = session.feature_matrix.shape[0]
        if not (0 <= doc_index1 < n_docs):
            return {"error": f"doc_index1 {doc_index1} out of bounds (matrix has {n_docs} documents)."}
        if not (0 <= doc_index2 < n_docs):
            return {"error": f"doc_index2 {doc_index2} out of bounds (matrix has {n_docs} documents)."}

        v1 = session.feature_matrix[doc_index1:doc_index1+1]
        v2 = session.feature_matrix[doc_index2:doc_index2+1]

        norm1 = np.linalg.norm(v1.toarray()) if hasattr(v1, "toarray") else np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2.toarray()) if hasattr(v2, "toarray") else np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            sim_score = 0.0
        else:
            sim = cosine_similarity(v1, v2)
            sim_score = float(sim[0][0])

        text1 = ""
        text2 = ""
        if session.dataframe is not None and "text" in session.dataframe.columns:
            text1 = str(session.dataframe.iloc[doc_index1]["text"])
            text2 = str(session.dataframe.iloc[doc_index2]["text"])

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc_index1": doc_index1,
            "doc_index2": doc_index2,
            "cosine_similarity": sim_score,
            "text1_preview": text1,
            "text2_preview": text2
        }

        report_df = pd.DataFrame([{
            "doc_index1": doc_index1,
            "doc_index2": doc_index2,
            "cosine_similarity": sim_score
        }])

        session.store_result(
            tool_name="cosine_similarity_tool",
            args={"doc_index1": doc_index1, "doc_index2": doc_index2},
            summary=summary,
            full_report=report_df
        )

        return summary

    @tool
    def feature_correlation_matrix_tool(top_k: int = 20) -> dict:
        """Computes a feature-vs-feature correlation matrix over the top terms by variance and plots a heatmap.

        Args:
            top_k: Number of highest-variance terms to evaluate (default 20).

        Returns:
            Dict summary containing top_terms, correlation_matrix, top_k, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found in session. Run build_dtm_tool first."}

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
        feature_names = list(session.feature_names)

        # Compute term variance (ddof=0)
        variances = np.var(X, axis=0, ddof=0)

        # Deterministic sorting: highest variance first; if tied, alphabetical by feature name
        sorted_indices = sorted(range(len(variances)), key=lambda i: (-variances[i], feature_names[i]))
        k = min(top_k, len(feature_names))
        top_indices = sorted_indices[:k]

        top_terms = [feature_names[i] for i in top_indices]
        top_X = X[:, top_indices]

        # Compute Pearson correlation matrix across terms (columns)
        raw_corr = np.corrcoef(top_X, rowvar=False)
        corr_matrix = np.nan_to_num(raw_corr, nan=0.0)

        # Build Seaborn heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".4f",
            xticklabels=top_terms,
            yticklabels=top_terms,
            cmap="coolwarm",
            ax=ax,
            vmin=-1,
            vmax=1
        )
        ax.set_title(f"Feature Correlation Matrix (Top {k} Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig

        corr_df = pd.DataFrame(corr_matrix, index=top_terms, columns=top_terms)

        result_id = session.next_result_id("feature_correlation")
        summary = {
            "result_id": result_id,
            "top_k": k,
            "top_terms": top_terms,
            "correlation_matrix": corr_matrix.tolist()
        }

        session.store_result(
            tool_name="feature_correlation_matrix_tool",
            args={"top_k": top_k},
            summary=summary,
            full_report=corr_df
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
