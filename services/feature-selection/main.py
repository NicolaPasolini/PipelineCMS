from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from typing import Dict, List, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64, io

app = FastAPI(title="Feature Selection Service")

class FeatureSelectionParams(BaseModel):
    df_tall:      Dict[str, List[Any]]
    df_dis:       Dict[str, List[Any]]
    num_clusters: int = 6

@app.post("/feature-selection")
def select_features(params: FeatureSelectionParams):
    try:
        dfT_ALL      = pd.DataFrame(params.df_tall)
        dfDIS_interp = pd.DataFrame(params.df_dis)

        # ── Linkage su dati trasposti: ward + euclidean ──────────────
        # Identico al notebook: linkage(dfT_ALL.iloc[:, 1:].T, method='ward', metric='euclidean')
        linkage_data = linkage(
            dfT_ALL.iloc[:, 1:].values.T,
            method="ward",
            metric="euclidean"
        )

        # ── Flat clustering ──────────────────────────────────────────
        clusters = fcluster(linkage_data, params.num_clusters, criterion="maxclust")

        # ── Rappresentante: PRIMO elemento del cluster (come notebook) 
        selected_variables = []
        cluster_map = {}
        for cluster_id in range(1, max(clusters) + 1):
            cluster_indices = np.where(clusters == cluster_id)[0]
            members = list(dfT_ALL.iloc[:, 1:].columns[cluster_indices])
            representative = dfT_ALL.iloc[:, 1:].columns[cluster_indices[0]]
            selected_variables.append(representative)
            cluster_map[str(cluster_id)] = {
                "members": members,
                "representative": representative
            }

        # ── dfT_CLUSTERED: TIME + sensori selezionati ────────────────
        dfT_CLUSTERED = pd.concat(
            [dfT_ALL.iloc[:, 0], dfT_ALL[selected_variables]], axis=1
        )

        # ── Dendrogramma ─────────────────────────────────────────────
        feature_names = list(dfT_ALL.iloc[:, 1:].columns)
        fig, ax = plt.subplots(figsize=(13, 5))
        dendrogram(linkage_data, labels=feature_names, ax=ax, leaf_rotation=90)
        ax.set_title("Hierarchical Clustering Dendrogram")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Cluster Distance")
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        dendro_b64 = base64.b64encode(buf.getvalue()).decode()

        return {
            "df_clustered":      dfT_CLUSTERED.to_dict(orient="list"),
            "df_dis":            dfDIS_interp.to_dict(orient="list"),
            "df_tall":           dfT_ALL.to_dict(orient="list"),
            "selected_features": selected_variables,
            "cluster_map":       cluster_map,
            "dendrogram_png":    dendro_b64,
            "info": {
                "rows":               len(dfT_CLUSTERED),
                "cols_clustered":     len(dfT_CLUSTERED.columns),
                "columns_clustered":  list(dfT_CLUSTERED.columns),
                "num_clusters":       params.num_clusters,
                "selected_features":  selected_variables,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "feature-selection"}