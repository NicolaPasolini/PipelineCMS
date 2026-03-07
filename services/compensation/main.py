from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Dict, List, Any

app = FastAPI(title="Compensation Service")


class CompensationParams(BaseModel):
    # da /evaluate
    suggested_model: str                  # "LASSO" o "MLRA"
    # da /training
    lasso: Dict[str, Any]                 # avg_coef, avg_intercept, features, ...
    mlra: Dict[str, Any]                  # avg_coef, avg_intercept, features, ...
    # da /select_features
    df_clustered: Dict[str, List[Any]]    # TIME + 6 feature (per MLRA)
    df_tall: Dict[str, List[Any]]         # TIME + T1-T28   (per LASSO)


@app.post("/compensate")
def compensate(params: CompensationParams):
    try:
        if params.suggested_model == "LASSO":
            df = pd.DataFrame(params.df_tall)
            features = params.lasso.get("features", [])
            avg_coef = np.array(params.lasso.get("avg_coef", []))
            intercept = float(params.lasso.get("avg_intercept", 0.0))

        elif params.suggested_model == "MLRA":
            df = pd.DataFrame(params.df_clustered)
            features = params.mlra.get("features", [])
            avg_coef = np.array(params.mlra.get("avg_coef", []))
            intercept = float(params.mlra.get("avg_intercept", 0.0))

        else:
            raise HTTPException(status_code=400, detail=f"Modello non supportato: {params.suggested_model}")

        missing = [f for f in features if f not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Feature mancanti nel df: {missing}")

        X = df[features].values
        d_estimated = X @ avg_coef + intercept
        compensation_offsets = -d_estimated

        result = {
            "model_used": params.suggested_model,
            "features_used": features,
            "n_samples": len(d_estimated),
            "displacement_predicted_um": d_estimated.tolist(),
            "compensation_offset_um": compensation_offsets.tolist(),
            "summary": {
                "mean_displacement_um": round(float(np.mean(d_estimated)), 6),
                "max_displacement_um": round(float(np.max(np.abs(d_estimated))), 6),
                "mean_compensation_um": round(float(np.mean(compensation_offsets)), 6),
                "std_compensation_um": round(float(np.std(compensation_offsets)), 6),
            }
        }

        if "TIME" in df.columns:
            result["timestamps"] = df["TIME"].tolist()

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "compensation"}
