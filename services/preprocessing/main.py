from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, List, Any

app = FastAPI(title="Preprocessing Service")


class PreprocessParams(BaseModel):
    df_text: Dict[str, List[Any]]
    df_tint: Dict[str, List[Any]]
    df_dis:  Dict[str, List[Any]]
    displacement_col: str = "D3"


@app.post("/preprocess")
def preprocess(params: PreprocessParams):
    try:
        df_text = pd.DataFrame(params.df_text)
        df_tint = pd.DataFrame(params.df_tint)
        df_dis  = pd.DataFrame(params.df_dis)

        # ── Converti TIME in float64 esplicito ───────────────────────────
        df_text["TIME"] = df_text["TIME"].astype(np.float64)
        df_tint["TIME"] = df_tint["TIME"].astype(np.float64)
        df_dis["TIME"]  = df_dis["TIME"].astype(np.float64)

        # ── Ordina per TIME (sicurezza) ──────────────────────────────────
        df_text = df_text.sort_values("TIME").reset_index(drop=True)
        df_tint = df_tint.sort_values("TIME").reset_index(drop=True)
        df_dis  = df_dis.sort_values("TIME").reset_index(drop=True)

        # ── Trova range temporale comune ─────────────────────────────────
        t_min = max(df_text["TIME"].min(), df_tint["TIME"].min(), df_dis["TIME"].min())
        t_max = min(df_text["TIME"].max(), df_tint["TIME"].max(), df_dis["TIME"].max())

        # Taglia df_tint al range comune (sarà il nuovo asse temporale)
        mask_tint = (df_tint["TIME"] >= t_min) & (df_tint["TIME"] <= t_max)
        df_tint   = df_tint[mask_tint].reset_index(drop=True)
        time_target = df_tint["TIME"].values

        # ── Interpola temperature esterne su time_target ─────────────────
        temp_et_cols = [c for c in df_text.columns if c != "TIME"]
        interp_et = interp1d(
            df_text["TIME"].values,
            df_text[temp_et_cols].values.T,
            kind="linear",
            bounds_error=False,
            fill_value=(
                df_text[temp_et_cols].values[0],
                df_text[temp_et_cols].values[-1]
            )
        )
        et_interp = interp_et(time_target).T
        df_text_interp = pd.DataFrame(et_interp, columns=temp_et_cols)
        df_text_interp.insert(0, "TIME", time_target)

        # ── Interpola displacement su time_target ────────────────────────
        dis_cols = [c for c in df_dis.columns if c != "TIME"]
        interp_dis = interp1d(
            df_dis["TIME"].values,
            df_dis[dis_cols].values.T,
            kind="linear",
            bounds_error=False,
            fill_value=(
                df_dis[dis_cols].values[0],
                df_dis[dis_cols].values[-1]
            )
        )
        dis_interp = interp_dis(time_target).T
        df_dis_interp = pd.DataFrame(dis_interp, columns=dis_cols)
        df_dis_interp.insert(0, "TIME", time_target)

        # ── Rimuovi righe con NaN ────────────────────────────────────────
        valid = (
            ~df_text_interp.isnull().any(axis=1) &
            ~df_tint.isnull().any(axis=1) &
            ~df_dis_interp.isnull().any(axis=1)
        )
        df_text_interp = df_text_interp[valid].reset_index(drop=True)
        df_tint        = df_tint[valid].reset_index(drop=True)
        df_dis_interp  = df_dis_interp[valid].reset_index(drop=True)

        # ── Valori relativi (sottrai prima riga) ─────────────────────────
        time_col = df_text_interp["TIME"].copy()
        df_text_rel = df_text_interp.drop("TIME", axis=1) - df_text_interp.drop("TIME", axis=1).iloc[0]
        df_tint_rel = df_tint.drop("TIME", axis=1)       - df_tint.drop("TIME", axis=1).iloc[0]
        df_dis_rel  = df_dis_interp.drop("TIME", axis=1) - df_dis_interp.drop("TIME", axis=1).iloc[0]

        # ── Merge temperature T1..T28 ────────────────────────────────────
        df_tall = pd.concat(
            [time_col, df_text_rel, df_tint_rel],
            axis=1
        )

        # ── Target displacement ──────────────────────────────────────────
        d_target = pd.DataFrame({
            "TIME": time_col,
            params.displacement_col: df_dis_rel[params.displacement_col]
        })

        # ── Sostituisci inf e nan residui con 0 ─────────────────────────
        df_tall  = df_tall.replace([np.inf, -np.inf], np.nan).fillna(0)
        d_target = d_target.replace([np.inf, -np.inf], np.nan).fillna(0)

        return {
            "df_tall":    df_tall.to_dict(orient="list"),
            "d_target":   d_target.to_dict(orient="list"),
            "target_col": params.displacement_col,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
