"""Deterministic dimensionality reduction and aligned results."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SUBSPACE_GROUPS = {
    "Match context": {"categorical": ["Competition", "Venue", "Opponent"], "numeric": []},
    "Goal context": {"categorical": ["Type", "Playing_Position"], "numeric": ["minute_num"]},
    "Career context": {"categorical": ["Club"], "numeric": ["season_start"]},
}


def subspace_projection_matrix(data: pd.DataFrame):
    """Return 3x3 PCA projections for pairwise context subspaces, aligned by goal_id.

    Each context block is separately scaled/one-hot encoded. A cell projects the
    union of its row and column blocks (diagonal cells use one block). This is a
    small, fixed-matrix adaptation, not an implementation of the paper's tree.
    """
    encoded = {}
    for group, spec in SUBSPACE_GROUPS.items():
        parts = []
        if spec["numeric"]:
            values = data[spec["numeric"]].apply(pd.to_numeric, errors="coerce")
            values = values.fillna(values.median()).fillna(0)
            parts.append(StandardScaler().fit_transform(values))
        if spec["categorical"]:
            values = data[spec["categorical"]].fillna("Unknown").astype(str)
            parts.append(OneHotEncoder(handle_unknown="ignore", min_frequency=5,
                                       sparse_output=False).fit_transform(values))
        encoded[group] = np.column_stack(parts)
    result = {}
    names = list(SUBSPACE_GROUPS)
    for row_i, row_group in enumerate(names):
        for col_i, col_group in enumerate(names):
            key = (row_group, col_group)
            blocks = [encoded[row_group]] if row_i == col_i else [encoded[row_group], encoded[col_group]]
            matrix = np.column_stack(blocks)
            if matrix.shape[0] < 2:
                # A single filtered record has no estimable variance direction.
                projection = np.zeros((matrix.shape[0], 2))
            else:
                projection = PCA(n_components=2, random_state=42).fit_transform(matrix)
            result[key] = pd.DataFrame({"goal_id": data["goal_id"].to_numpy(),
                                        "PC1": projection[:, 0], "PC2": projection[:, 1]})
    return result


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
