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

@app.post("/preprocess")
def preprocess(params: PreprocessParams):
    try:
        dfT_ext = pd.DataFrame(params.df_text)
        dfT_int = pd.DataFrame(params.df_tint)
        dfDIS   = pd.DataFrame(params.df_dis)

        for df in [dfT_ext, dfT_int, dfDIS]:
            df["TIME"] = df["TIME"].astype(np.float64)

        # ── dfT_int è l'asse temporale master ───────────────────────
        time_target = dfT_int["TIME"]

        # ── Interpolazione con extrapolate (identico al notebook) ────
        interp_ext = interp1d(
            dfT_ext["TIME"], dfT_ext.iloc[:, 1:].values.T,
            kind="linear", fill_value="extrapolate"
        )
        interp_dis = interp1d(
            dfDIS["TIME"], dfDIS.iloc[:, 1:].values.T,
            kind="linear", fill_value="extrapolate"
        )

        temp_target_ET  = interp_ext(time_target)
        dis_target_DIS  = interp_dis(time_target)

        dfT_ext_interp = pd.concat(
            [pd.DataFrame(time_target).reset_index(drop=True),
             pd.DataFrame(temp_target_ET.T, columns=dfT_ext.columns[1:])],
            axis=1
        )
        dfDIS_interp = pd.concat(
            [pd.DataFrame(time_target).reset_index(drop=True),
             pd.DataFrame(dis_target_DIS.T, columns=dfDIS.columns[1:])],
            axis=1
        )
        dfT_ext_interp.columns = dfT_ext.columns
        dfDIS_interp.columns   = dfDIS.columns

        # ── Rimozione NaN solo sui frame interpolati ─────────────────
        valid_ext = ~np.isnan(dfT_ext_interp.values).any(axis=1)
        valid_dis = ~np.isnan(dfDIS_interp.values).any(axis=1)
        valid = valid_ext & valid_dis

        dfT_ext_interp = dfT_ext_interp[valid].reset_index(drop=True)
        dfT_int        = dfT_int[valid].reset_index(drop=True)
        dfDIS_interp   = dfDIS_interp[valid].reset_index(drop=True)

        # ── Valori relativi: sottrai prima riga (incluso TIME) ────────
        dfT_ext_interp = dfT_ext_interp - dfT_ext_interp.iloc[0, :]
        dfT_int        = dfT_int        - dfT_int.iloc[0, :]
        dfDIS_interp   = dfDIS_interp   - dfDIS_interp.iloc[0, :]

        # ── Merge T1-T28 in dfT_ALL ──────────────────────────────────
        dfT_ALL = pd.concat(
            [dfT_ext_interp, dfT_int.iloc[:, 1:]], axis=1
        )

        return {
            "df_tall":   dfT_ALL.to_dict(orient="list"),
            "df_dis":    dfDIS_interp.to_dict(orient="list"),
            "info": {
                "rows":        len(dfT_ALL),
                "cols_tall":   len(dfT_ALL.columns),
                "cols_dis":    len(dfDIS_interp.columns),
                "T1_first":    round(float(dfT_ALL["T1"].iloc[0]), 6),
                "D3_first":    round(float(dfDIS_interp["D3"].iloc[0]), 6),
                "D3_min":      round(float(dfDIS_interp["D3"].min()), 4),
                "D3_max":      round(float(dfDIS_interp["D3"].max()), 4),
                "columns_tall": list(dfT_ALL.columns),
                "columns_dis":  list(dfDIS_interp.columns),
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "preprocessing"}
