from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Any, Optional

app = FastAPI(title="Prediction Service")


class PredictionParams(BaseModel):
    df_clustered:     Dict[str, List[Any]]
    d_target:         Dict[str, List[Any]]
    target_col:       str = "D3"
    selected_features: List[str]
    model:            str = "LASSO"   # "LASSO" oppure "MLRA"
    skip_cv_lasso:    bool = False
    n_folds:          int = 5


@app.post("/predict")
def predict(params: PredictionParams):
    try:
        # ── Ricostruisci DataFrame ────────────────────────────────────────
        df_clustered = pd.DataFrame(params.df_clustered)
        d_target     = pd.DataFrame(params.d_target)

        # ── Prepara X e y ────────────────────────────────────────────────
        X = df_clustered[params.selected_features].values
        y = d_target[params.target_col].values

        # ── Split train/test (80/20) ──────────────────────────────────────
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        kf = KFold(n_splits=params.n_folds, shuffle=False)

        # ── LASSO ─────────────────────────────────────────────────────────
        if params.model == "LASSO":

            if not params.skip_cv_lasso:
                # GridSearchCV per trovare alpha ottimale
                param_grid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}
                lasso      = Lasso(max_iter=10000)
                grid_search = GridSearchCV(
                    lasso, param_grid,
                    cv=kf, scoring="neg_mean_squared_error"
                )
                grid_search.fit(X_train, y_train)
                best_alpha = grid_search.best_params_["alpha"]
            else:
                best_alpha = 0.01   # valore di default se si skippa CV

            model = Lasso(alpha=best_alpha, max_iter=10000)
            model.fit(X_train, y_train)

            y_pred      = model.predict(X_test)
            coefficients = dict(zip(params.selected_features, model.coef_.tolist()))
            extra        = {"best_alpha": best_alpha, "coefficients": coefficients}

        # ── MLRA (Multiple Linear Regression) ────────────────────────────
        elif params.model == "MLRA":
            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred       = model.predict(X_test)
            coefficients = dict(zip(params.selected_features, model.coef_.tolist()))
            extra        = {"intercept": model.intercept_, "coefficients": coefficients}

        else:
            raise ValueError(f"Modello non supportato: {params.model}. Usa 'LASSO' o 'MLRA'.")

        # ── Metriche ──────────────────────────────────────────────────────
        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)

        # ── Cross-validation score (su tutto il dataset) ──────────────────
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
        cv_rmse   = np.sqrt(-cv_scores).tolist()

        return {
            "model":       params.model,
            "target_col":  params.target_col,
            "metrics": {
                "MAE":     round(mae,  6),
                "MSE":     round(mse,  6),
                "RMSE":    round(rmse, 6),
                "R2":      round(r2,   6),
            },
            "cv_rmse":         cv_rmse,
            "cv_rmse_mean":    round(float(np.mean(cv_rmse)), 6),
            "predictions":     y_pred.tolist(),
            "y_true":          y_test.tolist(),
            **extra,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
