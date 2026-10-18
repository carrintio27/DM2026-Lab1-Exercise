import os
import math
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

# Ensure necessary NLTK tokenizer resources are downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def _sanitize_for_json(obj):
    """Recursively converts all missing value forms and NumPy types to JSON-safe Python primitives."""
    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return [_sanitize_for_json(x) for x in obj.tolist()]
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj) if not math.isnan(obj) else None
    return obj


def make_tools(session):
    @tool("tokenize_tool")
    def tokenize_tool(text_column: str = "text") -> dict:
        """
        Tokenizes document text into unigram token lists using nltk.word_tokenize.

        Args:
            text_column: Name of the column in session.dataframe containing raw text strings.

        Returns:
            Dict containing result_id, document count, total tokens, unique tokens, and sample tokens.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Call load_dataset_tool first."}

        df = session.dataframe
        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in dataset."}

        tokens_list = []
        for text in df[text_column]:
            if pd.isna(text) or not str(text).strip():
                tokens_list.append([])
            else:
                tokens = nltk.word_tokenize(str(text).lower())
                tokens_list.append(tokens)

        # Mutate session dataframe and artifacts as required
        df["unigrams"] = tokens_list
        session.artifacts["tokens"] = tokens_list

        total_tokens = sum(len(doc_tokens) for doc_tokens in tokens_list)
        unique_tokens = len(set(token for doc_tokens in tokens_list for token in doc_tokens))

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": len(df),
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "sample_doc_0_tokens": tokens_list[0] if tokens_list else []
        }

        session.store_result("tokenize_tool", {"text_column": text_column}, summary)
        return summary

    @tool("check_missing_tool")
    def check_missing_tool(text_column: str = "text") -> dict:
        """
        Checks for missing, null, or empty text values in the working dataset.

        Args:
            text_column: Column name in session.dataframe to check for missing text.

        Returns:
            Dict containing result_id, total documents, missing_count, and missing_indices.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Call load_dataset_tool first."}

        df = session.dataframe
        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in dataset."}

        missing_indices = []
        for idx, text in enumerate(df[text_column]):
            if pd.isna(text) or not str(text).strip():
                missing_indices.append(idx)

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_documents": len(df),
            "missing_count": len(missing_indices),
            "missing_indices": missing_indices
        }

        session.store_result("check_missing_tool", {"text_column": text_column}, summary)
        return summary

    @tool("check_duplicates_tool")
    def check_duplicates_tool(text_column: str = "text", drop: bool = False) -> dict:
        """
        Checks for duplicate documents in the dataset and optionally drops all copies using keep=False.

        Args:
            text_column: Column name in session.dataframe containing text strings.
            drop: If True, drops all copies of duplicated rows (keep=False) from session.dataframe.

        Returns:
            Dict containing result_id, duplicate_count, dropped status, and n_remaining_documents.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Call load_dataset_tool first."}

        df = session.dataframe
        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in dataset."}

        # Count extra duplicated rows under keep='first'
        duplicate_count = int(df.duplicated(subset=[text_column], keep="first").sum())

        if drop and duplicate_count > 0:
            # Must use keep=False per specification to drop all copies of duplicate rows
            session.dataframe = df.drop_duplicates(subset=[text_column], keep=False).reset_index(drop=True)

        n_remaining = len(session.dataframe)
        result_id = session.next_result_id("check_duplicates")

        summary = {
            "result_id": result_id,
            "duplicate_count": duplicate_count,
            "dropped": drop,
            "n_remaining_documents": n_remaining
        }

        session.store_result("check_duplicates_tool", {"text_column": text_column, "drop": drop}, summary)
        return summary

    @tool("list_files_tool")
    def list_files_tool(subdirectory: str = "") -> dict:
        """
        Lists subdirectories and files in the specified directory relative to the workspace root.

        Args:
            subdirectory: Subdirectory path relative to workspace root (defaults to workspace root if empty).

        Returns:
            Dict containing result_id, subdirectory, directories, files, n_directories, and n_files.
        """
        base_dir = os.path.abspath(".")
        target_dir = os.path.abspath(os.path.join(base_dir, subdirectory)) if subdirectory else base_dir

        # Path traversal guard using os.path.commonpath
        try:
            if os.path.commonpath([base_dir, target_dir]) != base_dir:
                return {"error": "Access denied: Path is outside the workspace."}
        except ValueError:
            # Raised if paths are on different drives on Windows
            return {"error": "Access denied: Path is outside the workspace."}

        if not os.path.exists(target_dir):
            return {"error": f"Directory '{subdirectory}' does not exist."}

        if not os.path.isdir(target_dir):
            return {"error": f"Path '{subdirectory}' is not a directory."}

        entries = os.listdir(target_dir)
        directories = sorted([e for e in entries if os.path.isdir(os.path.join(target_dir, e))])
        files = sorted([e for e in entries if os.path.isfile(os.path.join(target_dir, e))])

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
            "n_directories": len(directories),
            "n_files": len(files)
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool("sample_data_tool")
    def sample_data_tool(n: int = 5, random_state: int = 42) -> dict:
        """
        Subsamples the working dataset using pandas DataFrame.sample.

        Args:
            n: Number of items to sample from the dataset.
            random_state: Seed for random number generator to ensure reproducibility.

        Returns:
            Dict containing result_id, sample_size, original_size, and sampled_indices.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Call load_dataset_tool first."}

        df = session.dataframe
        original_size = len(df)

        if n <= 0:
            return {"error": f"Sample size n ({n}) must be greater than 0."}

        if n > original_size:
            return {"error": f"Sample size n ({n}) cannot exceed total dataset size ({original_size})."}

        sampled_df = df.sample(n=n, random_state=random_state)
        sampled_indices = list(sampled_df.index)

        # Update session.dataframe with the sampled subset
        session.dataframe = sampled_df.reset_index(drop=True)

        result_id = session.next_result_id("sample")
        summary = {
            "result_id": result_id,
            "sample_size": len(session.dataframe),
            "original_size": original_size,
            "sampled_indices": sampled_indices
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool("describe_data_tool")
    def describe_data_tool(text_column: str = "text", label_column: str = "category_name") -> dict:
        """
        Computes text length summary statistics overall and per category, and generates a box plot.

        Args:
            text_column: Column name in session.dataframe containing text strings.
            label_column: Column name in session.dataframe containing category labels.

        Returns:
            Dict containing result_id, text_lengths, overall_stats, and by_category_stats.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Call load_dataset_tool first."}

        df = session.dataframe
        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in dataset."}

        # Calculate character text length per document
        text_lengths = df[text_column].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        df["text_length"] = text_lengths

        overall_stats = df["text_length"].describe().to_dict()

        # Group by category_name (or label_column fallback)
        by_category_stats = {}
        target_col = label_column if label_column in df.columns else ("label" if "label" in df.columns else None)

        if target_col:
            grouped = df.groupby(target_col)["text_length"].describe()
            by_category_stats = grouped.to_dict(orient="index")

        # Generate box plot
        fig, ax = plt.subplots(figsize=(6, 4))
        if target_col:
            sns.boxplot(data=df, x=target_col, y="text_length", ax=ax)
            ax.set_xlabel("Category")
        else:
            sns.boxplot(y=df["text_length"], ax=ax)

        ax.set_ylabel("Text Length (chars)")
        ax.set_title("Document Text Length Distribution")
        fig.tight_layout()

        # Assign figure to session.pending_figure directly (do NOT call plt.close)
        session.pending_figure = fig

        result_id = session.next_result_id("describe")
        summary = {
            "result_id": result_id,
            "text_lengths": list(text_lengths),
            "overall_stats": overall_stats,
            "by_category_stats": by_category_stats
        }

        session.store_result("describe_data_tool", {"text_column": text_column, "label_column": label_column}, summary)
        return summary

    @tool("inspect_data_tool")
    def inspect_data_tool(n: int = 5) -> dict:
        """
        Peeks at the top n rows of session.dataframe, returning dataset dimensions, column names, data types, and row preview without modifying session state.

        Args:
            n: Number of top rows to preview (default 5).

        Returns:
            Dict containing result_id, total_rows, total_columns, columns, column_dtypes, and head.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Call load_dataset_tool first."}

        df = session.dataframe

        if n <= 0:
            return {"error": "Number of preview rows n must be greater than 0."}

        shape = df.shape
        head_records = df.head(n).to_dict(orient="records")
        sanitized_head = _sanitize_for_json(head_records)

        result_id = session.next_result_id("inspect")
        summary = {
            "result_id": result_id,
            "total_rows": int(shape[0]),
            "total_columns": int(shape[1]),
            "columns": list(df.columns),
            "column_dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
            "preview_rows_count": len(sanitized_head),
            "head": sanitized_head
        }

        session.store_result("inspect_data_tool", {"n": n}, summary)
        return summary

    return [
        tokenize_tool,
        check_missing_tool,
        check_duplicates_tool,
        list_files_tool,
        sample_data_tool,
        describe_data_tool,
        inspect_data_tool
    ]
