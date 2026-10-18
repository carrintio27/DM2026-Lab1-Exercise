import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.15) -> dict:
        """Flags near-constant terms by variance for removal from the DTM.

        Args:
            threshold: Minimum variance required to keep a term (default 0.15).

        Returns:
            Dict summary containing kept_terms, removed_terms, total_terms,
            kept_count, varianc_per_term, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found in session. Run build_dtm_tool first."}

        # Convert sparse matrix to dense array or compute variance per column
        matrix = session.feature_matrix
        if hasattr(matrix, "toarray"):
            dense_matrix = matrix.toarray()
            variances = np.var(dense_matrix, axis=0, ddof=0)
        else:
            variances = np.var(matrix, axis=0, ddof=0)

        feature_names = session.feature_names
        variances_dict = {term: float(var) for term, var in zip(feature_names, variances)}

        # Create DataFrame report: term as 1st col, variance as 2nd col
        report_df = pd.DataFrame({
            "feature": feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        kept_df = report_df[report_df["variance"] >= threshold]
        removed_df = report_df[report_df["variance"] < threshold]

        kept_terms = kept_df["feature"].tolist()
        removed_terms = removed_df["feature"].tolist()

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_count": len(kept_terms),
            "removed_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "variances": variances_dict,
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=report_df
        )

        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.5) -> dict:
        """Filters terms by signed Pearson correlation with a binary target class indicator.

        Args:
            target_class: The target category name to evaluate correlation against.
            threshold: Minimum Pearson correlation coefficient required to keep a term (default 0.5).

        Returns:
            Dict summary containing target_class, threshold, kept_terms, removed_terms,
            correlations per term, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found in session. Run build_dtm_tool first."}
        if session.labels is None:
            return {"error": "No labels found in session. Run build_dtm_tool first."}

        matrix = session.feature_matrix
        X = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        y = (np.asarray(session.labels) == target_class).astype(float)

        if np.std(y) == 0:
            return {"error": f"Target class '{target_class}' has zero variance in labels."}

        correlations = []
        std_y = np.std(y)
        mean_y = np.mean(y)

        for col_idx in range(X.shape[1]):
            x_col = X[:, col_idx].astype(float)
            std_x = np.std(x_col)
            if std_x == 0:
                correlations.append(0.0)
            else:
                r = np.mean((x_col - np.mean(x_col)) * (y - mean_y)) / (std_x * std_y)
                correlations.append(float(r))

        feature_names = session.feature_names
        correlations_dict = {term: float(r) for term, r in zip(feature_names, correlations)}

        report_df = pd.DataFrame({
            "feature": feature_names,
            "pearson_r": correlations
        }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        kept_df = report_df[report_df["pearson_r"] >= threshold]
        removed_df = report_df[report_df["pearson_r"] < threshold]

        kept_terms = kept_df["feature"].tolist()
        removed_terms = removed_df["feature"].tolist()

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_count": len(kept_terms),
            "removed_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "correlations": correlations_dict,
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=report_df
        )

        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.5) -> dict:
        """Filters terms by signed Spearman rank correlation with a binary target class indicator.

        Args:
            target_class: The target category name to evaluate correlation against.
            threshold: Minimum Spearman correlation coefficient required to keep a term (default 0.5).

        Returns:
            Dict summary containing target_class, threshold, kept_terms, removed_terms,
            correlations per term, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found in session. Run build_dtm_tool first."}
        if session.labels is None:
            return {"error": "No labels found in session. Run build_dtm_tool first."}

        matrix = session.feature_matrix
        X = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        y = (np.asarray(session.labels) == target_class).astype(float)

        if np.std(y) == 0:
            return {"error": f"Target class '{target_class}' has zero variance in labels."}

        correlations = []
        for col_idx in range(X.shape[1]):
            x_col = X[:, col_idx].astype(float)
            if np.std(x_col) == 0:
                correlations.append(0.0)
            else:
                rho, _ = spearmanr(x_col, y)
                if np.isnan(rho):
                    correlations.append(0.0)
                else:
                    correlations.append(float(rho))

        feature_names = session.feature_names
        correlations_dict = {term: float(r) for term, r in zip(feature_names, correlations)}

        report_df = pd.DataFrame({
            "feature": feature_names,
            "spearman_r": correlations
        }).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        kept_df = report_df[report_df["spearman_r"] >= threshold]
        removed_df = report_df[report_df["spearman_r"] < threshold]

        kept_terms = kept_df["feature"].tolist()
        removed_terms = removed_df["feature"].tolist()

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_count": len(kept_terms),
            "removed_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "correlations": correlations_dict,
        }

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=report_df
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
