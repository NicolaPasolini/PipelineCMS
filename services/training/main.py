from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from typing import Dict, List, Any

app = FastAPI(title="Training Service")

class TrainParams(BaseModel):
    df_clustered: Dict[str, List[Any]]   # TIME + feature selezionate (per MLRA)
    df_tall:      Dict[str, List[Any]]   # TIME + T1-T28 (per LASSO)
    df_dis:       Dict[str, List[Any]]   # TIME + D1-D5
    displ:        int   = 3              # indice colonna displacement in df_dis (3 = D3)
    k_folds:      int   = 10
    random_state: int   = 42

@app.post("/train")
def train(params: TrainParams):
    try:
        dfT_CLUSTERED = pd.DataFrame(params.df_clustered)
        dfT_ALL       = pd.DataFrame(params.df_tall)
        dfDIS_interp  = pd.DataFrame(params.df_dis)

        # ── Target: dfDIS_interp.iloc[:, displ] (indice colonna, non nome) ──
        dfDIS_select_MLRA = dfDIS_interp.iloc[:, params.displ]
        target_col_name   = dfDIS_interp.columns[params.displ]

        # ── KFold: shuffle=True, random_state=42 (identico al notebook) ─────
        kf = KFold(n_splits=params.k_folds, random_state=params.random_state, shuffle=True)

        # ════════════════════════════════════════════════════════════════
        # MLRA — usa dfT_CLUSTERED (feature selezionate da HCA)
        # ════════════════════════════════════════════════════════════════
        reg             = LinearRegression()
        estimators_coef = []
        estimators_intercept = []

        for train_idx, test_idx in kf.split(dfT_CLUSTERED):
            X_train = dfT_CLUSTERED.iloc[train_idx, 1:]
            X_test  = dfT_CLUSTERED.iloc[test_idx,  1:]
            y_train = dfDIS_select_MLRA.iloc[train_idx]
            reg.fit(X_train, y_train)
            estimators_coef.append(reg.coef_.tolist())
            estimators_intercept.append(float(reg.intercept_))

        avg_coef      = np.mean(estimators_coef, axis=0)
        avg_intercept = float(np.mean(estimators_intercept))

        # Predizione finale su tutto il dataset con coef medi
        dfDi_EST = np.dot(dfT_CLUSTERED.iloc[:, 1:].values, avg_coef) + avg_intercept

        # RMSE notebook: sqrt(mean(|neg_MAE scores|))  — replicato identico
        reg_scores = cross_val_score(
            reg, dfT_CLUSTERED.iloc[:, 1:], dfDIS_select_MLRA,
            scoring="neg_mean_absolute_error", cv=kf, n_jobs=-1
        )
        rmse_mlra = float(np.sqrt(np.mean(np.absolute(reg_scores))))

        # Pearson MLRA
        corr_mlra = stats.pearsonr(dfDi_EST, dfDIS_select_MLRA.values)

        # ════════════════════════════════════════════════════════════════
        # LASSO — Grid Search alpha su dfT_ALL (28 features)
        # ════════════════════════════════════════════════════════════════
        parameters  = {"alpha": np.logspace(-4, 0, 30)}
        lasso_grid  = GridSearchCV(Lasso(), parameters, cv=kf)
        lasso_grid.fit(dfT_ALL.iloc[:, 1:], dfDIS_select_MLRA)
        best_alpha  = float(lasso_grid.best_params_["alpha"])

        lasso_alpha    = Lasso(alpha=best_alpha)
        lasso_coefs    = []
        lasso_intercepts = []

        for train_idx, test_idx in kf.split(dfT_ALL):
            X_train = dfT_ALL.iloc[train_idx, 1:]
            X_test  = dfT_ALL.iloc[test_idx,  1:]
            y_train = dfDIS_select_MLRA.iloc[train_idx]
            lasso_alpha.fit(X_train, y_train)
            lasso_coefs.append(lasso_alpha.coef_.tolist())
            lasso_intercepts.append(float(lasso_alpha.intercept_))

        lasso_avg_coef      = np.mean(lasso_coefs, axis=0)
        lasso_avg_intercept = float(np.mean(lasso_intercepts))

        dfDi_EST_lasso = np.dot(dfT_ALL.iloc[:, 1:].values, lasso_avg_coef) + lasso_avg_intercept

        # RMSE LASSO
        scores_lasso = cross_val_score(
            lasso_alpha, dfT_ALL.iloc[:, 1:], dfDIS_select_MLRA,
            scoring="neg_mean_absolute_error", cv=kf, n_jobs=-1
        )
        rmse_lasso = float(np.sqrt(np.mean(np.absolute(scores_lasso))))

        # Pearson LASSO
        corr_lasso = stats.pearsonr(dfDi_EST_lasso, dfDIS_select_MLRA.values)

        lasso_nonzero = int(np.sum(lasso_avg_coef != 0))

        return {
            "mlra": {
                "avg_coef":       avg_coef.tolist(),
                "avg_intercept":  avg_intercept,
                "rmse":           rmse_mlra,
                "pearson_r":      float(corr_mlra[0]),
                "pearson_p":      float(corr_mlra[1]),
                "features":       list(dfT_CLUSTERED.columns[1:]),
                "y_pred":         dfDi_EST.tolist(),
            },
            "lasso": {
                "avg_coef":       lasso_avg_coef.tolist(),
                "avg_intercept":  lasso_avg_intercept,
                "best_alpha":     best_alpha,
                "rmse":           rmse_lasso,
                "pearson_r":      float(corr_lasso[0]),
                "pearson_p":      float(corr_lasso[1]),
                "nonzero_coef":   lasso_nonzero,
                "features":       list(dfT_ALL.columns[1:]),
                "y_pred":         dfDi_EST_lasso.tolist(),
            },
            "target": {
                "col_name":  target_col_name,
                "col_index": params.displ,
                "y_true":    dfDIS_select_MLRA.tolist(),
                "time":      dfDIS_interp["TIME"].tolist(),
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "training"}