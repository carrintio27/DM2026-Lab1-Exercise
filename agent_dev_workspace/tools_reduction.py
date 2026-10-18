from typing import Literal, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15
    ) -> dict:
        """Reduces the dimensionality of the document-term matrix (e.g. using PCA, t-SNE, or UMAP)
        and plots a scatter plot of the reduced 2D coordinates colored by category.

        Args:
            method: Dimensionality reduction algorithm ("pca", "tsne", or "umap").
            n_components: Number of target dimensions (default 2).
            random_state: Seed for reproducible output (default 42).
            perplexity: Perplexity parameter for t-SNE (default 30.0).
            n_neighbors: Number of neighbors for UMAP (default 15).

        Returns:
            Dict summary containing method, n_components, coordinates, explained_variance_ratio
            (for PCA), and result_id.
        """
        if session.feature_matrix is None:
            return {"error": "No document-term matrix found in session. Run build_dtm_tool first."}

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
        X = X.astype(np.float64)

        explained_variance_ratio = None

        if method == "pca":
            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(X)
            explained_variance_ratio = model.explained_variance_ratio_.tolist()
        elif method == "tsne":
            model = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                init="random",
                learning_rate=200.0,
                method="exact"
            )
            coords = model.fit_transform(X)
        elif method == "umap":
            model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
            coords = model.fit_transform(X)
        else:
            return {"error": f"Unsupported method: '{method}'. Choose 'pca', 'tsne', or 'umap'."}

        coords_list = coords.tolist()

        # Build scatter plot and set session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = session.labels if session.labels is not None else ["Unassigned"] * X.shape[0]

        df_plot = pd.DataFrame({
            "Dim 1": coords[:, 0],
            "Dim 2": coords[:, 1] if n_components > 1 else np.zeros(X.shape[0]),
            "Category": labels
        })

        sns.scatterplot(data=df_plot, x="Dim 1", y="Dim 2", hue="Category", ax=ax, s=100)
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.grid(True, linestyle="--", alpha=0.5)

        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "coordinates": coords_list,
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={
                "method": method,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors
            },
            summary=summary,
            full_report=df_plot
        )

        return summary

    return [reduce_dimensions_tool]
