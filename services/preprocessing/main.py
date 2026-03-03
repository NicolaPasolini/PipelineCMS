from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, List, Any

app = FastAPI(title="Preprocessing Service")

class PreprocessParams(BaseModel):
    df_text:          Dict[str, List[Any]]
    df_tint:          Dict[str, List[Any]]
    df_dis:           Dict[str, List[Any]]
    displacement_col: str = "D3"

@app.post("/preprocess")
def preprocess(params: PreprocessParams):
    try:
        df_text = pd.DataFrame(params.df_text)
        df_tint = pd.DataFrame(params.df_tint)
        df_dis  = pd.DataFrame(params.df_dis)

        # ── Cast e ordinamento temporale ─────────────────────────────
        for df in [df_text, df_tint, df_dis]:
            df["TIME"] = df["TIME"].astype(np.float64)
            df.sort_values("TIME", inplace=True)
            df.reset_index(drop=True, inplace=True)

        # ── Intervallo temporale comune ──────────────────────────────
        t_min = max(df_text["TIME"].min(), df_tint["TIME"].min(), df_dis["TIME"].min())
        t_max = min(df_text["TIME"].max(), df_tint["TIME"].max(), df_dis["TIME"].max())

        # df_tint diventa l'asse temporale di riferimento
        mask  = (df_tint["TIME"] >= t_min) & (df_tint["TIME"] <= t_max)
        df_tint   = df_tint[mask].reset_index(drop=True)
        t_target  = df_tint["TIME"].values

        # ── Interpolazione temperature esterne su t_target ───────────
        ext_cols = [c for c in df_text.columns if c != "TIME"]
        f_ext = interp1d(
            df_text["TIME"].values,
            df_text[ext_cols].values.T,
            kind="linear", bounds_error=False,
            fill_value=(df_text[ext_cols].values[0],
                        df_text[ext_cols].values[-1])
        )
        df_text_i = pd.DataFrame(f_ext(t_target).T, columns=ext_cols)
        df_text_i.insert(0, "TIME", t_target)

        # ── Interpolazione displacement su t_target ──────────────────
        dis_cols = [c for c in df_dis.columns if c != "TIME"]
        f_dis = interp1d(
            df_dis["TIME"].values,
            df_dis[dis_cols].values.T,
            kind="linear", bounds_error=False,
            fill_value=(df_dis[dis_cols].values[0],
                        df_dis[dis_cols].values[-1])
        )
        df_dis_i = pd.DataFrame(f_dis(t_target).T, columns=dis_cols)
        df_dis_i.insert(0, "TIME", t_target)

        # ── Rimozione righe con NaN ──────────────────────────────────
        valid = (
            ~df_text_i.isnull().any(axis=1) &
            ~df_tint.isnull().any(axis=1)   &
            ~df_dis_i.isnull().any(axis=1)
        )
        df_text_i = df_text_i[valid].reset_index(drop=True)
        df_tint   = df_tint[valid].reset_index(drop=True)
        df_dis_i  = df_dis_i[valid].reset_index(drop=True)

        # ── ΔT: scaling rispetto alla prima lettura ──────────────────
        time_col    = df_text_i["TIME"].copy()
        df_text_rel = df_text_i.drop("TIME", axis=1) - df_text_i.drop("TIME", axis=1).iloc[0]
        df_tint_rel = df_tint.drop("TIME", axis=1)   - df_tint.drop("TIME", axis=1).iloc[0]
        df_dis_rel  = df_dis_i.drop("TIME", axis=1)  - df_dis_i.drop("TIME", axis=1).iloc[0]

        # ── Merge features (T1-T28) e target ────────────────────────
        df_tall = pd.concat([time_col, df_text_rel, df_tint_rel], axis=1)
        df_target = pd.DataFrame({
            "TIME": time_col,
            params.displacement_col: df_dis_rel[params.displacement_col]
        })

        # ── Pulizia residui ──────────────────────────────────────────
        df_tall   = df_tall.replace([np.inf, -np.inf], np.nan).fillna(0)
        df_target = df_target.replace([np.inf, -np.inf], np.nan).fillna(0)

        return {
            "df_tall":    df_tall.to_dict(orient="list"),
            "df_target":  df_target.to_dict(orient="list"),
            "target_col": params.displacement_col,
            "info": {
                "rows":           len(df_tall),
                "feature_cols":   [c for c in df_tall.columns if c != "TIME"],
                "t_min":          float(t_min),
                "t_max":          float(t_max),
                "displacement_col": params.displacement_col,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "preprocessing"}
