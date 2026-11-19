"""Deterministic dimensionality reduction and aligned results."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness


def project(matrix, metadata: pd.DataFrame, random_state: int = 42):
    n = matrix.shape[0]
    if n < 4:
        raise ValueError("At least four goals are required for projection and trustworthiness.")
    pca = PCA(n_components=2, random_state=random_state)
    pc = pca.fit_transform(matrix)
    perplexity = min(30.0, max(2.0, (n - 1) / 3.0))
    perplexity = min(perplexity, n - 1.0)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state,
                init="pca", learning_rate="auto", max_iter=1000)
    ts = tsne.fit_transform(matrix)
    k = min(10, max(1, (n - 1) // 2))
    pca_trust = trustworthiness(matrix, pc, n_neighbors=k)
    tsne_trust = trustworthiness(matrix, ts, n_neighbors=k)
    out = metadata.copy().reset_index(drop=True)
    out["PC1"], out["PC2"] = pc[:, 0], pc[:, 1]
    out["TSNE1"], out["TSNE2"] = ts[:, 0], ts[:, 1]
    metrics = {"pca_variance": pca.explained_variance_ratio_.tolist(),
               "pca_variance_total": float(pca.explained_variance_ratio_.sum()),
               "pca_trustworthiness": float(pca_trust), "tsne_trustworthiness": float(tsne_trust),
               "trustworthiness_neighbors": k, "tsne_perplexity": perplexity}
    return out, metrics
