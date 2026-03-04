from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Optional
import os

app = FastAPI(title="Ingestion Service")

DATA_DIR = os.getenv("DATA_DIR", "/app/data")

class IngestParams(BaseModel):
    normalize_time: bool = False

@app.post("/ingest")
def ingest(params: IngestParams):
    try:
        # ── EXTERNAL TEMPERATURES ────────────────────────────────────
        dfT_ext_1 = pd.read_csv(f"{DATA_DIR}/TE1.csv", sep=";", decimal=",")
        dfT_ext_2 = pd.read_csv(f"{DATA_DIR}/TE2.csv", sep=";", decimal=",")
        dfT_ext_3 = pd.read_csv(f"{DATA_DIR}/TE3.csv", sep=";", decimal=",")
        dfT_ext_4 = pd.read_csv(f"{DATA_DIR}/TE4.csv", sep=";", decimal=",")
        # Remove last column
        for df in [dfT_ext_1, dfT_ext_2, dfT_ext_3, dfT_ext_4]:
            df.drop(df.columns[4], axis=1, inplace=True)

        dfT_ext_1.columns = ["TIME", "T1",  "T2",  "T3"]
        dfT_ext_2.columns = ["TIME", "T4",  "T5",  "T6"]
        dfT_ext_3.columns = ["TIME", "T7",  "T8",  "T9"]
        dfT_ext_4.columns = ["TIME", "T10", "T11", "T12"]

        dfT_ext = pd.concat([dfT_ext_1, dfT_ext_2.iloc[:,1:], dfT_ext_3.iloc[:,1:], dfT_ext_4.iloc[:,1:]], axis=1)

        # Timestamp: formato con punti nell'orario  "%d/%m/%Y %H.%M.%S"
        dfT_ext["TIME"] = pd.to_datetime(dfT_ext["TIME"], format="%d/%m/%Y %H.%M.%S")
        dfT_ext["TIME"] = dfT_ext["TIME"].astype(np.int64) / 10**9

        # ── INTERNAL TEMPERATURES ────────────────────────────────────
        dfT_int = pd.read_csv(f"{DATA_DIR}/TI.csv", sep=";", decimal=",")
        dfT_int.drop(dfT_int.columns[dfT_int.shape[1]-1], axis=1, inplace=True)

        # Merge prime due colonne per comporre datetime
        col0 = dfT_int.iloc[:, 0].str.strip()
        col1 = dfT_int.iloc[:, 1].str.strip()
        dt_col = pd.to_datetime(col0 + " " + col1, format="%d/%m/%Y %H:%M:%S.%f").astype(np.int64) / 10**9

        dfT_int = pd.concat([pd.DataFrame(dt_col), dfT_int.iloc[:, 2:]], axis=1)
        dfT_int.columns = ["TIME"] + [f"T{i}" for i in range(13, 29)]

        # ── DISPLACEMENTS ────────────────────────────────────────────
        dfDIS = pd.read_csv(f"{DATA_DIR}/Displacements.csv", sep=";", header=None)
        dfDIS = dfDIS.drop(dfDIS.columns[5:], axis=1)

        # Parsing colonna 3: lstrip('+') poi split('+')
        dfDIS.iloc[:, 3] = dfDIS.iloc[:, 3].str.lstrip("+")
        dfMIDDLE = pd.DataFrame(dfDIS.iloc[:, 3].str.split("+", expand=True))
        dfDIS = pd.concat([dfDIS.iloc[:, 0], dfDIS.iloc[:, 1], dfDIS.iloc[:, 2], dfMIDDLE, dfDIS.iloc[:, 4]], axis=1)
        dfDIS.columns = ["TIME", "D1", "D2", "D3", "D4", "D5"]

        # Conversione: float * 1000 (µm)
        dis_cols = dfDIS.columns[1:]
        dfDIS[dis_cols] = dfDIS[dis_cols].astype(float) * 1000

        # Timestamp displacement: formato "%d/%m/%Y %H:%M:%S"
        dfDIS["TIME"] = pd.to_datetime(dfDIS["TIME"], format="%d/%m/%Y %H:%M:%S")
        dfDIS["TIME"] = dfDIS["TIME"].astype(np.int64) / 10**9

        # ── Normalizzazione TIME opzionale ───────────────────────────
        if params.normalize_time:
            scaler = MinMaxScaler()
            for df in [dfT_ext, dfT_int, dfDIS]:
                df["TIME"] = scaler.fit_transform(df[["TIME"]])

        return {
            "df_text": dfT_ext.to_dict(orient="list"),
            "df_tint": dfT_int.to_dict(orient="list"),
            "df_dis":  dfDIS.to_dict(orient="list"),
            "info": {
                "rows_text":    len(dfT_ext),
                "rows_tint":    len(dfT_int),
                "rows_dis":     len(dfDIS),
                "cols_text":    len(dfT_ext.columns),
                "cols_tint":    len(dfT_int.columns),
                "cols_dis":     len(dfDIS.columns),
                "columns_text": list(dfT_ext.columns),
                "columns_tint": list(dfT_int.columns),
                "columns_dis":  list(dfDIS.columns),
                "T1_min": round(float(dfT_ext["T1"].min()), 2),
                "T1_max": round(float(dfT_ext["T1"].max()), 2),
                "D3_min": round(float(dfDIS["D3"].min()), 2),
                "D3_max": round(float(dfDIS["D3"].max()), 2),
    }
}


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "ingestion"}