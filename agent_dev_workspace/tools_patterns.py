import os
import tempfile
from typing import Literal, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# PAMI imports
from PAMI.extras.convert import DF2DB
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth

def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None
    ) -> dict:
        """Mines frequent word patterns for a specific document category using PAMI algorithms.

        Args:
            category_name: Target category to mine patterns from.
            filtering_method: Method to filter vocabulary ("variance", "tfidf", or "term_frequency").
            algorithm: Pattern mining algorithm ("fpgrowth", "topk", or "maxfpgrowth").
            min_sup: Minimum support threshold (required positive integer for fpgrowth/maxfpgrowth).
            k: Number of top patterns (required positive integer for topk).

        Returns:
            Dict summary containing category_name, filtering_method, algorithm, patterns, and result_id.
        """
        # Validate arguments early: positive integer checks
        if algorithm in ["fpgrowth", "maxfpgrowth"]:
            if min_sup is None or not isinstance(min_sup, int) or min_sup <= 0:
                return {"error": f"min_sup must be a positive integer for algorithm '{algorithm}'."}
        if algorithm == "topk":
            if k is None or not isinstance(k, int) or k <= 0:
                return {"error": "k must be a positive integer for algorithm 'topk'."}

        if session.dataframe is None:
            return {"error": "No dataframe found in session. Load a dataset first."}

        # Step 1: Filter to category and build fresh local DTM with stop_words="english"
        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        if cat_df.empty:
            return {"error": f"No documents found for category '{category_name}'."}

        texts = cat_df["text"].fillna("").tolist()

        try:
            vectorizer = CountVectorizer(stop_words="english")
            X_sparse = vectorizer.fit_transform(texts)
            feature_names = np.array(vectorizer.get_feature_names_out())
        except ValueError:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived stopword removal"
            }
            session.store_result("mine_patterns_tool", {}, summary)
            return summary

        if len(feature_names) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived stopword removal"
            }
            session.store_result("mine_patterns_tool", {}, summary)
            return summary

        X = X_sparse.toarray()

        # Step 2: Apply vocabulary filter
        if filtering_method == "variance":
            scores = np.var(X, axis=0, ddof=0)
            low, high = np.percentile(scores, 5), np.percentile(scores, 95)
            kept_mask = (scores > low) & (scores < high)
        elif filtering_method == "term_frequency":
            scores = np.sum(X, axis=0)
            low, high = np.percentile(scores, 5), np.percentile(scores, 95)
            kept_mask = (scores > low) & (scores < high)
        elif filtering_method == "tfidf":
            tfidf_transformer = TfidfTransformer()
            X_tfidf = tfidf_transformer.fit_transform(X).toarray()
            scores = np.mean(X_tfidf, axis=0)
            cutoff = np.percentile(scores, 20)
            kept_mask = scores >= cutoff
        else:
            return {"error": f"Unknown filtering method: '{filtering_method}'."}

        kept_terms = feature_names[kept_mask]

        # Step 3: Handle 0 terms kept
        if len(kept_terms) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {}, summary)
            return summary

        # Step 4: Convert to Transactional DB via PAMI DF2DB
        kept_indices = np.where(kept_mask)[0]
        binary_df = pd.DataFrame(X[:, kept_indices] >= 1, columns=kept_terms)
        
        # Drop empty transactions (rows with all False / 0 terms)
        binary_df = binary_df[binary_df.any(axis=1)]

        if binary_df.empty:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no non-empty transactions survived"
            }
            session.store_result("mine_patterns_tool", {}, summary)
            return summary

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = os.path.join(tmp_dir, "transactions.db")
            out_file = os.path.join(tmp_dir, "output.txt")

            converter = DF2DB.DF2DB(binary_df, "dense")
            converter.convert2TransactionalDatabase(db_file)

            if algorithm == "fpgrowth":
                miner = FPGrowth.FPGrowth(db_file, minSup=min_sup)
            elif algorithm == "topk":
                miner = FAE.FAE(db_file, k=k)
            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowth.MaxFPGrowth(db_file, minSup=min_sup)
            else:
                return {"error": f"Unknown algorithm: '{algorithm}'."}

            miner.mine()
            miner.save(out_file)

            # Parse mined patterns
            patterns = []
            if os.path.exists(out_file):
                with open(out_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split(":")
                            if len(parts) >= 2:
                                items = parts[0].strip().split("\t")
                                if len(items) == 1 and " " in items[0]:
                                    items = items[0].split()
                                sup = float(parts[1].strip())
                                patterns.append({"pattern": items, "support": sup})

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "pattern_count": len(patterns),
            "patterns": patterns
        }

        report_df = pd.DataFrame(patterns) if patterns else pd.DataFrame(columns=["pattern", "support"])

        session.store_result(
            tool_name="mine_patterns_tool",
            args={
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k
            },
            summary=summary,
            full_report=report_df
        )

        return summary

    return [mine_patterns_tool]
