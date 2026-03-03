from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Dict, List, Any

app = FastAPI(title="Feature Selection Service")


class FeatureSelectionParams(BaseModel):
    df_tall:      Dict[str, List[Any]]
    d_target:     Dict[str, List[Any]]
    target_col:   str = "D3"
    num_clusters: int = 6


@app.post("/select_features")
def select_features(params: FeatureSelectionParams):
    try:
        # ── Ricostruisci DataFrame ────────────────────────────────────────
        df_tall  = pd.DataFrame(params.df_tall)
        d_target = pd.DataFrame(params.d_target)

        # ── Isola solo le colonne temperature (T1..T28, escludi TIME) ────
        temp_cols = [c for c in df_tall.columns if c != "TIME"]
        df_temps  = df_tall[temp_cols]

        # ── HIERARCHICAL CLUSTERING su matrice di correlazione ────────────
        # Stessa logica del notebook: distanza = 1 - |correlazione|
        corr_matrix  = df_temps.corr().abs()
        distance_mat = 1 - corr_matrix

        # Linkage su upper triangle della matrice di distanza
        condensed = distance_mat.values[
            np.triu_indices(len(temp_cols), k=1)
        ]
        Z = linkage(condensed, method="average")

        # Assegna cluster
        cluster_labels = fcluster(Z, t=params.num_clusters, criterion="maxclust")

        # ── Seleziona feature rappresentativa per ogni cluster ────────────
        # Criterio: feature con correlazione media più alta col resto del cluster
        selected_features = []
        for cluster_id in range(1, params.num_clusters + 1):
            cluster_mask  = cluster_labels == cluster_id
            cluster_feats = [f for f, m in zip(temp_cols, cluster_mask) if m]

            if len(cluster_feats) == 1:
                selected_features.append(cluster_feats[0])
            else:
                # Seleziona quella con correlazione media più alta nel cluster
                sub_corr = corr_matrix.loc[cluster_feats, cluster_feats]
                mean_corr = sub_corr.mean()
                best_feat = mean_corr.idxmax()
                selected_features.append(best_feat)

        # ── Costruisci df clustered (TIME + feature selezionate) ──────────
        df_clustered = df_tall[["TIME"] + selected_features].copy()

        return {
            "df_clustered":       df_clustered.to_dict(orient="list"),
            "d_target":           d_target.to_dict(orient="list"),
            "target_col":         params.target_col,
            "selected_features":  selected_features,
            "cluster_labels":     cluster_labels.tolist(),
            "all_features":       temp_cols,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
