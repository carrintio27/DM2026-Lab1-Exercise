import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def binarize_labels_tool() -> dict:
        """One-hot encodes / binarizes the dataset category labels.

        Returns:
            Dict summary containing classes, n_samples, shape, one_hot_matrix, and result_id.
        """
        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].tolist()

        if labels is None or len(labels) == 0:
            return {"error": "No labels found in session. Load a dataset first."}

        # Explicitly check for missing label values using pd.isna(labels).any()
        if pd.isna(labels).any():
            return {"error": "Cannot binarize labels: dataset contains missing label values."}

        lb = LabelBinarizer()
        binarized = lb.fit_transform(labels)
        classes = lb.classes_.tolist()

        # Store in session artifacts
        session.artifacts["binarized_labels"] = binarized
        session.artifacts["label_binarizer"] = lb

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "classes": classes,
            "n_samples": len(labels),
            "shape": list(binarized.shape),
            "one_hot_matrix": binarized.tolist()
        }

        report_df = pd.DataFrame(
            binarized,
            columns=classes if binarized.shape[1] == len(classes) else [classes[-1]]
        )

        session.store_result(
            tool_name="binarize_labels_tool",
            args={},
            summary=summary,
            full_report=report_df
        )

        return summary

    return [binarize_labels_tool]
