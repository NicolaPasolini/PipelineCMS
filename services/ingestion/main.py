from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Ingestion Service")

DATA_DIR = "/app/data"

EXT_TEMP_FILES = ["TE1.csv", "TE2.csv", "TE3.csv", "TE4.csv"]
INT_TEMP_FILE  = "TI.csv"
DISP_FILE      = "Displacements.csv"


class IngestParams(BaseModel):
    normalize_time: bool = False


@app.post("/ingest")
def ingest(params: IngestParams):
    try:
        # ── TEMPERATURE ESTERNE ──────────────────────────────────────────
        # Struttura: "Data Ora;TEMP 1 (°C);TEMP 2 (°C);TEMP 3 (°C);"
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
            # Rimuovi colonna vuota finale (trailing ;)
            df = df.dropna(axis=1, how="all")
            df = df.iloc[:, :4]
            df.columns = cols
            # Converti valori temperatura in float
            for c in cols[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            dfs_ext.append(df)

        df_text = pd.concat(
            [dfs_ext[0]] + [d.iloc[:, 1:] for d in dfs_ext[1:]],
            axis=1
        )

        # Converti TIME → timestamp Unix
        df_text["TIME"] = pd.to_datetime(
            df_text["TIME"], format="%d/%m/%Y %H.%M.%S"
        ).astype(int) // 10**9

        if params.normalize_time:
            df_text["TIME"] = (df_text["TIME"] - df_text["TIME"].min()) / \
                              (df_text["TIME"].max() - df_text["TIME"].min())

        # ── TEMPERATURE INTERNE ──────────────────────────────────────────
        # Struttura: "DATA; TIME; DB290.xxx LREAL; ..." separatore ";"
        df_tint = pd.read_csv(
            os.path.join(DATA_DIR, INT_TEMP_FILE),
            sep=";", decimal=",", header=0
        )
        # Rimuovi colonna vuota finale
        df_tint = df_tint.dropna(axis=1, how="all")

        # Colonna 0 = DATA, colonna 1 = TIME (con spazio iniziale)
        date_col = df_tint.iloc[:, 0].str.strip()
        time_col = df_tint.iloc[:, 1].str.strip()
        new_dt = pd.to_datetime(
            date_col + " " + time_col,
            format="%d/%m/%Y %H:%M:%S.%f"
        ).astype(int) // 10**9

        # Colonne 2..17 = 16 temperature interne
        data_cols = df_tint.iloc[:, 2:]
        for c in data_cols.columns:
            data_cols[c] = pd.to_numeric(data_cols[c], errors="coerce")

        df_tint = pd.concat(
            [pd.DataFrame({"TIME": new_dt}), data_cols.reset_index(drop=True)],
            axis=1
        )
        df_tint.columns = ["TIME"] + [f"T{i}" for i in range(13, 29)]

        if params.normalize_time:
            df_tint["TIME"] = (df_tint["TIME"] - df_tint["TIME"].min()) / \
                              (df_tint["TIME"].max() - df_tint["TIME"].min())

        # ── DISPLACEMENT ─────────────────────────────────────────────────
        # Struttura: "13/12/2022 08:54:57;-000.002;+000.000;-000.001+000.000;+000.000;..."
        # Col 0: TIME, Col 1: D1, Col 2: D2, Col 3: D3+D4 appiccicati, Col 4: D5
        df_dis = pd.read_csv(
            os.path.join(DATA_DIR, DISP_FILE),
            sep=";", header=None
        )

        time_col  = df_dis.iloc[:, 0]
        d1        = df_dis.iloc[:, 1].astype(float) * 1000
        d2        = df_dis.iloc[:, 2].astype(float) * 1000
        d5        = df_dis.iloc[:, 4].astype(float) * 1000

        # Colonna 3 contiene due valori appiccicati tipo "-000.001+000.000"
        # Split su "+" o "-" tenendo il segno: usa regex
        col3 = df_dis.iloc[:, 3].astype(str).str.strip()

        # Trova il secondo numero: è sempre lungo 8 char con segno (+/-)
        # Formato: ±000.000±000.000 → prendi ultimi 8 caratteri come D4
        d3 = col3.str[:8].astype(float) * 1000
        d4 = col3.str[8:].astype(float) * 1000

        df_dis = pd.DataFrame({
            "TIME": time_col,
            "D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5
        })

        df_dis["TIME"] = pd.to_datetime(
            df_dis["TIME"], format="%d/%m/%Y %H:%M:%S"
        ).astype(int) // 10**9

        if params.normalize_time:
            df_dis["TIME"] = (df_dis["TIME"] - df_dis["TIME"].min()) / \
                             (df_dis["TIME"].max() - df_dis["TIME"].min())

        return {
            "df_text": df_text.to_dict(orient="list"),
            "df_tint": df_tint.to_dict(orient="list"),
            "df_dis":  df_dis.to_dict(orient="list"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
