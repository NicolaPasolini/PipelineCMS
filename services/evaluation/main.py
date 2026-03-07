from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64, io
from typing import Dict, List, Any, Optional

app = FastAPI(title="Evaluation Service")


class EvaluationParams(BaseModel):
    mlra: Dict[str, Any]
    lasso: Dict[str, Any]
    target: Dict[str, Any]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


@app.post("/evaluate")
def evaluate(params: EvaluationParams):
    try:
        y_true = np.array(params.target["y_true"], dtype=float)
        y_pred_mlra = np.array(params.mlra["y_pred"], dtype=float)
        y_pred_lasso = np.array(params.lasso["y_pred"], dtype=float)
        time = params.target.get("time", list(range(len(y_true))))
        target_col = params.target.get("col_name", "target")

        metrics_mlra = compute_metrics(y_true, y_pred_mlra)
        metrics_lasso = compute_metrics(y_true, y_pred_lasso)

        # ── Riduzione errore % rispetto a baseline (std y_true) ──────────
        baseline = float(np.std(y_true))
        reduction_mlra  = round((1 - metrics_mlra["RMSE"]  / baseline) * 100, 2) if baseline > 0 else None
        reduction_lasso = round((1 - metrics_lasso["RMSE"] / baseline) * 100, 2) if baseline > 0 else None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, y_pred, label, metrics in [
            (axes[0], y_pred_mlra,  "MLRA",  metrics_mlra),
            (axes[1], y_pred_lasso, "LASSO", metrics_lasso),
        ]:
            ax.plot(time, y_pred, label=label, linewidth=1.2)
            ax.plot(time, y_true, label=target_col, linewidth=1.2, linestyle="--")
            ax.set_title(
                f"{target_col} vs Estimated {label}\n"
                f"RMSE={metrics['RMSE']:.4f} R²={metrics['R2']:.4f} MAE={metrics['MAE']:.4f}"
            )
            ax.set_xlabel("TIME")
            ax.set_ylabel("Displacement")
            ax.legend()

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        plot_b64 = base64.b64encode(buf.getvalue()).decode()

        suggested = "LASSO" if metrics_lasso["RMSE"] < metrics_mlra["RMSE"] else "MLRA"

        return {
            "metrics": {
                "MLRA":  metrics_mlra,
                "LASSO": metrics_lasso,
            },
            "reduction_vs_baseline": {
                "MLRA":  reduction_mlra,
                "LASSO": reduction_lasso,
                "baseline_std": round(baseline, 6),
                "note": "% riduzione RMSE rispetto a std(y_true). >70% = buono, >50% = accettabile"
            },
            "suggested_model": suggested,
            "comparison_plot": plot_b64,
            "info": {
                "target_col": target_col,
                "n_samples": int(len(y_true)),
                "mlra_features": params.mlra.get("features", []),
                "lasso_nonzero": params.lasso.get("nonzero_coef", None),
                "lasso_alpha": params.lasso.get("best_alpha", None),
            }
        }

    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Campo mancante nel payload: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "evaluation"}
