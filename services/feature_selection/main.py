from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, List, Any, Literal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64, io

app = FastAPI(title="Feature Service")

# Sensori disponibili per tipo di test
TEST_TYPE_SENSORS = {
    "Z":    [f"T{i}" for i in range(1, 29)],
    "Y":    [f"T{i}" for i in range(1, 29)],
    "X":    [f"T{i}" for i in range(1, 29)],
    "ETVE": [f"T{i}" for i in range(13, 29)],  # solo interni
}

class FeatureSelectionParams(BaseModel):
    df_tall:        Dict[str, List[Any]]
    df_target:      Dict[str, List[Any]]
    target_col:     str   = "D3"
    num_clusters:   int   = 6
    vif_threshold:  float = 10.0
    test_type:      Literal["Z", "Y", "X", "ETVE"] = "Z"

@app.post("/select-features")
def select_features(params: FeatureSelectionParams):
    try:
        df_tall   = pd.DataFrame(params.df_tall)
        df_target = pd.DataFrame(params.df_target)

        # ── Sensori candidati per test_type ──────────────────────────
        candidates = [
            c for c in TEST_TYPE_SENSORS[params.test_type]
            if c in df_tall.columns
        ]
        if len(candidates) < 2:
            raise ValueError(f"Troppo pochi sensori disponibili per test_type={params.test_type}")

        df_temps = df_tall[candidates].copy()

        # ── HCA: clustering gerarchico su matrice correlazione ───────
        corr_mat     = df_temps.corr().abs()
        dist_mat     = 1 - corr_mat
        n            = len(candidates)
        condensed    = dist_mat.values[np.triu_indices(n, k=1)]
        Z            = linkage(condensed, method="average")
        num_clusters = min(params.num_clusters, n)
        labels       = fcluster(Z, t=num_clusters, criterion="maxclust")

        # Un rappresentante per cluster (quello con correlazione media massima)
        selected = []
        cluster_map = {}
        for cid in range(1, num_clusters + 1):
            members = [f for f, m in zip(candidates, labels) if m == cid]
            if not members:
                continue
            if len(members) == 1:
                rep = members[0]
            else:
                sub_corr = corr_mat.loc[members, members]
                rep = sub_corr.mean().idxmax()
            selected.append(rep)
            cluster_map[str(cid)] = {"members": members, "representative": rep}

        # ── VIF check: rimozione iterativa multicollinearità ─────────
        vif_df      = df_temps[selected].copy()
        removed_vif = []
        vif_final   = {}

        while vif_df.shape[1] >= 2:
            X = vif_df.values
            vifs = {
                col: variance_inflation_factor(X, i)
                for i, col in enumerate(vif_df.columns)
            }
            max_feat = max(vifs, key=vifs.get)
            if vifs[max_feat] > params.vif_threshold:
                vif_df.drop(columns=[max_feat], inplace=True)
                removed_vif.append(max_feat)
            else:
                vif_final = {k: round(v, 3) for k, v in vifs.items()}
                break

        final_features = list(vif_df.columns)

        # ── Dendrogramma → PNG base64 ────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 5))
        dendrogram(Z, labels=candidates, ax=ax, leaf_rotation=90, color_threshold=0.5)
        ax.set_title(f"HCA Dendrogram — test_type: {params.test_type}", fontsize=13)
        ax.set_xlabel("Sensore")
        ax.set_ylabel("Distanza (1 − correlazione)")
        ax.axhline(y=0.5, color="red", linestyle="--", linewidth=0.8, label="soglia taglio")
        ax.legend()
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        dendro_b64 = base64.b64encode(buf.getvalue()).decode()

        # ── df_clustered: solo TIME + feature finali ─────────────────
        df_clustered = pd.concat(
            [df_tall["TIME"], df_tall[final_features]], axis=1
        )

        return {
            "df_clustered":      df_clustered.to_dict(orient="list"),
            "df_target":         df_target.to_dict(orient="list"),
            "target_col":        params.target_col,
            "selected_features": final_features,
            "removed_by_vif":    removed_vif,
            "vif_scores":        vif_final,
            "cluster_map":       cluster_map,
            "test_type":         params.test_type,
            "dendrogram_png":    dendro_b64,
            "info": {
                "n_candidates":      len(candidates),
                "n_clusters":        num_clusters,
                "n_after_hca":       len(selected),
                "n_after_vif":       len(final_features),
                "removed_by_vif":    removed_vif,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "feature-selection"}
