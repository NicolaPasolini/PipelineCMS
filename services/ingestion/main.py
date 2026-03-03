from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Ingestion Service")

DATA_DIR       = "/app/data"
EXT_TEMP_FILES = ["TE1.csv", "TE2.csv", "TE3.csv", "TE4.csv"]
INT_TEMP_FILE  = "TI.csv"
DISP_FILE      = "Displacements.csv"

class IngestParams(BaseModel):
    normalize_time: bool = True

@app.post("/ingest")
def ingest(params: IngestParams):
    try:
        # ── Temperature esterne (T1-T12) ─────────────────────────────
        col_groups = [
            ["TIME", "T1",  "T2",  "T3"],
            ["TIME", "T4",  "T5",  "T6"],
            ["TIME", "T7",  "T8",  "T9"],
            ["TIME", "T10", "T11", "T12"],
        ]
        dfs_ext = []
        for fname, cols in zip(EXT_TEMP_FILES, col_groups):
            df = pd.read_csv(
                os.path.join(DATA_DIR, fname),
                sep=";", decimal=",", header=0
            )
            df = df.dropna(axis=1, how="all").iloc[:, :4]
            df.columns = cols
            for c in cols[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            dfs_ext.append(df)

        df_text = pd.concat(
            [dfs_ext[0]] + [d.iloc[:, 1:] for d in dfs_ext[1:]],
            axis=1
        )
        df_text["TIME"] = (
            pd.to_datetime(df_text["TIME"], format="%d/%m/%Y %H.%M.%S")
            .astype("int64") // 10**9
        )

        # ── Temperature interne (T13-T28) ────────────────────────────
        df_tint = pd.read_csv(
            os.path.join(DATA_DIR, INT_TEMP_FILE),
            sep=";", decimal=",", header=0
        )
        df_tint  = df_tint.dropna(axis=1, how="all")
        date_col = df_tint.iloc[:, 0].str.strip()
        time_col = df_tint.iloc[:, 1].str.strip()
        new_dt   = (
            pd.to_datetime(date_col + " " + time_col, format="%d/%m/%Y %H:%M:%S.%f")
            .astype("int64") // 10**9
        )
        data_cols = df_tint.iloc[:, 2:].copy()
        for c in data_cols.columns:
            data_cols[c] = pd.to_numeric(data_cols[c], errors="coerce")
        df_tint = pd.concat(
            [pd.DataFrame({"TIME": new_dt}), data_cols.reset_index(drop=True)],
            axis=1
        )
        df_tint.columns = ["TIME"] + [f"T{i}" for i in range(13, 29)]

        # ── Displacements (D1-D5) ────────────────────────────────────
        df_dis = pd.read_csv(
            os.path.join(DATA_DIR, DISP_FILE),
            sep=";", header=None
        )
        time_col_dis = df_dis.iloc[:, 0]
        d1 = df_dis.iloc[:, 1].astype(float) / 1000
        d2 = df_dis.iloc[:, 2].astype(float) / 1000
        d5 = df_dis.iloc[:, 4].astype(float) / 1000
        col3 = df_dis.iloc[:, 3].astype(str).str.strip()
        d3 = col3.str[-8:].astype(float) / 1000
        d4 = col3.str[:8].astype(float)  / 1000
        df_dis = pd.DataFrame({
            "TIME": time_col_dis,
            "D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5
        })
        df_dis["TIME"] = (
            pd.to_datetime(df_dis["TIME"], format="%d/%m/%Y %H:%M:%S")
            .astype("int64") // 10**9
        )

        # ── Normalizzazione TIME → [0, 1] ────────────────────────────
        if params.normalize_time:
            for df in [df_text, df_tint, df_dis]:
                t_min = df["TIME"].min()
                t_max = df["TIME"].max()
                df["TIME"] = (df["TIME"] - t_min) / (t_max - t_min)

        return {
            "df_text": df_text.to_dict(orient="list"),
            "df_tint": df_tint.to_dict(orient="list"),
            "df_dis":  df_dis.to_dict(orient="list"),
            "info": {
                "rows_text": len(df_text),
                "rows_tint": len(df_tint),
                "rows_dis":  len(df_dis),
                "columns_text": list(df_text.columns),
                "columns_tint": list(df_tint.columns),
                "columns_dis":  list(df_dis.columns),
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "ingestion"}
