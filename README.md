# Thermal Compensation Pipeline — n8n Orchestration

Pipeline di orchestrazione per la stima e compensazione della deformazione termica di un centro di lavoro CNC a 5 assi (CMS VM-30K). Trasforma una pipeline Python monolitica in un'architettura a microservizi orchestrata tramite [n8n](https://n8n.io), con integrazione di task Human-in-the-Loop (HITL) e supporto cognitivo LLM (Gemini 2.5 Flash).

---

## Architettura

```text
[Trigger]
    │
    ▼
[Ingestion :8001]       ← carica CSV sensori e spostamenti
    │
    ▼
[Preprocessing :8002]   ← interpolazione, valori relativi, merge T1-T28
    │
    ▼
[Feature Selection :8003] ← clustering gerarchico HCA, selezione rappresentanti
    │
    ▼
[HITL 1 — Form]         ← operatore verifica stato macchina e sensori
    │
    ├── nessun_problema ─────────────────────────────────────────┐
    │                                                             │
    ├── sensore_guasto → [sensor removal] → [Feature Selection 2] │
    │                                                             │
    └── macchina_anomala → [STOP + log]                [Body5] → [Training :8004]
                                                                  │
                                                                  ▼
                                                          [Evaluation :8005]
                                                                  │
                                                                  ▼
                                                      [LLM Gemini 2.5 Flash] ← Generazione Report
                                                                  │
                                                                  ▼
                                                          [HITL 2 — Form] ← Operatore approva/rifiuta offset
                                                                  │
                                                                  ├── Rifiuta → [STOP]
                                                                  │
                                                                  └── Approva → [Compensation :8006]
```

---

## Microservizi

| Servizio | Porta | Endpoint | Descrizione |
|---|---|---|---|
| `ingestion` | 8001 | `POST /ingest` | Carica TE1-TE4, TI, Displacements.csv e converte i timestamp in Unix epoch |
| `preprocessing` | 8002 | `POST /preprocess` | Interpola i DataFrame su asse temporale comune, calcola valori relativi, merge T1-T28 |
| `feature-selection` | 8003 | `POST /feature-selection` | Clustering gerarchico ward+euclidean su T trasposta, ritorna cluster_map e df_clustered |
| `training` | 8004 | `POST /train` | Allena MLRA (su df_clustered) e LASSO con GridSearch (su df_tall), KFold CV |
| `evaluation` | 8005 | `POST /evaluate` | Calcola MAE, MSE, RMSE, R², riduzione % vs baseline, genera plot comparativo |
| `compensation` | 8006 | `POST /compensate` | Applica il modello validato per calcolare gli offset di compensazione in µm |

Tutti i servizi espongono anche `GET /health`.  
Rete Docker interna: `pipeline` (bridge).

---

## Avvio

```bash
docker-compose up --build
```

n8n disponibile su `http://localhost:5678`.

---

## API Reference

### `POST /ingest`

Legge i file CSV dalla directory `DATA_DIR` (default `/app/data`).

**File attesi:**
| File | Sensori | Formato timestamp |
|---|---|---|
| `TE1.csv` – `TE4.csv` | T1–T12 (esterni) | `%d/%m/%Y %H.%M.%S` |
| `TI.csv` | T13–T28 (interni) | `%d/%m/%Y %H:%M:%S.%f` |
| `Displacements.csv` | D1–D5 (µm) | `%d/%m/%Y %H:%M:%S` |

**Request body:**
```json
{ "normalize_time": false }
```

**Response:**
```json
{
  "df_text": { "TIME": [...], "T1": [...], ..., "T12": [...] },
  "df_tint": { "TIME": [...], "T13": [...], ..., "T28": [...] },
  "df_dis":  { "TIME": [...], "D1": [...], ..., "D5": [...] },
  "info": { "rows_text": 460, "rows_tint": 460, "rows_dis": 460, ... }
}
```

---

### `POST /preprocess`

Interpola `df_text` e `df_dis` sull'asse temporale di `df_tint` (master).  
Rimuove NaN, normalizza in valori relativi alla prima riga (incluso TIME), merge T1-T28 → `df_tall`.

**Request body:**
```json
{
  "df_text": { "TIME": [...], "T1": [...], ..., "T12": [...] },
  "df_tint": { "TIME": [...], "T13": [...], ..., "T28": [...] },
  "df_dis":  { "TIME": [...], "D1": [...], ..., "D5": [...] }
}
```

**Response:**
```json
{
  "df_tall": { "TIME": [...], "T1": [...], ..., "T28": [...] },
  "df_dis":  { "TIME": [...], "D1": [...], ..., "D5": [...] },
  "info": { "rows": 460, "cols_tall": 29, "D3_min": -52.3, "D3_max": 0.0, ... }
}
```

---

### `POST /feature-selection`

Clustering gerarchico (ward + euclidean) sui sensori T trasposti.  
Il rappresentante di ogni cluster è il **primo elemento** per indice.  
Genera anche il dendrogramma come PNG base64.

**Request body:**
```json
{
  "df_tall": { "TIME": [...], "T1": [...], ..., "T28": [...] },
  "df_dis":  { "TIME": [...], "D1": [...], ..., "D5": [...] },
  "num_clusters": 6
}
```

**Response:**
```json
{
  "df_clustered":      { "TIME": [...], "T17": [...], "T4": [...], ... },
  "df_dis":            { ... },
  "df_tall":           { ... },
  "selected_features": ["T17", "T4", "T2", "T6", "T1", "T26"],
  "cluster_map": {
    "1": { "members": ["T17","T18","T19","T20"], "representative": "T17" },
    "2": { "members": ["T4","T13","T21"], "representative": "T4" }
  },
  "dendrogram_png": "<base64>",
  "info": { "num_clusters": 6, "cols_clustered": 7, ... }
}
```

---

### `POST /train`

Allena due modelli in parallelo tramite KFold CV (default: 10 fold, shuffle=True):

- **MLRA** (Multiple Linear Regression) — su `df_clustered` (feature HCA)
- **LASSO** — su `df_tall` (tutti i 28 sensori), con GridSearch su `alpha ∈ logspace(-4, 0, 30)`

Il target è la colonna `displ` di `df_dis` (default index 3 = D3).

**Request body:**
```json
{
  "df_clustered": { "TIME": [...], "T17": [...], ... },
  "df_tall":      { "TIME": [...], "T1": [...], ..., "T28": [...] },
  "df_dis":       { "TIME": [...], "D1": [...], ..., "D5": [...] },
  "displ": 3,
  "k_folds": 10,
  "random_state": 42
}
```

**Response:**
```json
{
  "mlra": {
    "avg_coef": [...], "avg_intercept": -0.12,
    "rmse": 1.43, "pearson_r": 0.98, "pearson_p": 0.0,
    "features": ["T17", "T4", "T2", "T6", "T1", "T26"],
    "y_pred": [...]
  },
  "lasso": {
    "avg_coef": [...], "avg_intercept": -0.08,
    "best_alpha": 0.0023, "rmse": 1.21,
    "pearson_r": 0.99, "nonzero_coef": 12,
    "features": ["T1", ..., "T28"], "y_pred": [...]
  },
  "target": { "col_name": "D3", "col_index": 3, "y_true": [...], "time": [...] }
}
```

---

### `POST /evaluate`

Calcola metriche su entrambi i modelli e genera un plot comparativo PNG base64.  
Suggerisce automaticamente il modello con RMSE minore.

**Request body:** output completo di `/train`

**Response:**
```json
{
  "metrics": {
    "MLRA":  { "MAE": 1.1, "MSE": 2.0, "RMSE": 1.43, "R2": 0.96 },
    "LASSO": { "MAE": 0.9, "MSE": 1.5, "RMSE": 1.21, "R2": 0.97 }
  },
  "reduction_vs_baseline": {
    "MLRA": 72.4, "LASSO": 76.8,
    "baseline_std": 5.23,
    "note": "% riduzione RMSE rispetto a std(y_true). >70% = buono, >50% = accettabile"
  },
  "suggested_model": "LASSO",
  "comparison_plot": "<base64 PNG>"
}
```

---

### `POST /compensate`

Applica il modello suggerito e approvato per calcolare gli offset di compensazione termica in µm.  
`compensation_offset = -displacement_predicted`

**Request body:**
```json
{
  "suggested_model": "LASSO",
  "lasso": { "avg_coef": [...], "avg_intercept": -0.08, "features": ["T1",...] },
  "mlra":  { "avg_coef": [...], "avg_intercept": -0.12, "features": ["T17",...] },
  "df_clustered": { "TIME": [...], "T17": [...], ... },
  "df_tall":      { "TIME": [...], "T1": [...], ..., "T28": [...] }
}
```

**Response:**
```json
{
  "model_used": "LASSO",
  "features_used": ["T1", ..., "T28"],
  "n_samples": 460,
  "displacement_predicted_um": [...],
  "compensation_offset_um": [...],
  "summary": {
    "mean_displacement_um": -18.4,
    "max_displacement_um": 52.3,
    "mean_compensation_um": 18.4,
    "std_compensation_um": 14.2
  }
}
```

---

## HITL 1 — Verifica stato macchina

### Scopo
Permettere all'operatore di segnalare problemi noti **prima** del training, evitando che dati corrotti influenzino il modello.

### Form (Wait node — On Form Submission)

| Campo | Tipo | Valori |
|---|---|---|
| `Stato della macchina` | Dropdown | `nessun_problema` / `sensore_guasto` / `macchina_anomala` |
| `Sensore da escludere` | Text | Es. `T17` (solo se stato = `sensore_guasto`) |
| `Note aggiuntive` | Textarea | Opzionale |

### Branch A — Nessun problema
→ `Body5` → `Training`

### Branch B — Sensore guasto

**`sensor removal`** — rimuove il sensore da `df_tall` e riduce `num_clusters`:
```javascript
const fs = $("Feature Selection").item.json;
const daEscludere = ($input.first().json["Sensore da escludere"] || "").trim().toUpperCase();

const dfTall = { ...fs.df_tall };
if (daEscludere) delete dfTall[daEscludere];

const numClusters = Math.max(2, Object.keys(fs.cluster_map).length - 1);

return [{ json: {
  df_tall:         dfTall,
  df_dis:          fs.df_dis,
  num_clusters:    numClusters,
  sensore_escluso: daEscludere
}}];
```

**`code fs2`** — prepara il body per Feature Selection 2:
```javascript
const input = $input.first().json;
return [{ json: {
  df_tall:      input.df_tall,
  df_dis:       input.df_dis,
  num_clusters: input.num_clusters
}}];
```

**`Feature Selection 2`** — stesso endpoint `/feature-selection`, su `df_tall` senza il sensore guasto.  
Produce un nuovo `cluster_map` ottimizzato senza il sensore rimosso.

> ⚠️ **Importante**: rimuovere il sensore rappresentante di un cluster elimina tutta la copertura termica di quel gruppo. Il modello risultante non potrà usare quel sensore neanche in inferenza. Usare solo in caso di guasto definitivo.

### Branch C — Macchina anomala
→ `STOP` (logging alert e notifica operatore)

---

## HITL 2 & Supporto Cognitivo LLM

### Scopo
Abbattere la barriera cognitiva tra i risultati matematici del backend (RMSE, R²) e l'operatore di macchina, delegando la responsabilità della decisione finale a un attore umano qualificato.

### Flusso Operativo
1. **Filtro Payload**: Un nodo Code rimuove la stringa Base64 del grafico generata dall'`Evaluation` per non saturare la context window dell'LLM:
   ```javascript
   const evalData = $input.all()[0].json;
   delete evalData.comparison_plot;
   return { json: evalData };
   ```
2. **Generazione Report (Gemini 2.5 Flash)**: Il modello riceve le metriche e formula un report diagnostico in italiano (indicando modello migliore, % riduzione errore e RMSE residuo).
3. **Form Operatore (Wait node)**: Il workflow si sospende.

### Form (Wait node — On Form Submission)

| Campo | Tipo | Valori / Descrizione |
|---|---|---|
| `Report Diagnostico` | HTML/Text | Sola lettura (Testo generato dall'LLM) |
| `Decisione Operativa` | Dropdown | `approva_compensazione` / `rifiuta_ricalcola` |

### Branch A — Approva
Il workflow recupera il JSON originale dall'Evaluation e lo invia al servizio `Compensation`.
→ `Compensation`

### Branch B — Rifiuta
→ `STOP` (workflow interrotto, macchina protetta da compensazioni errate)

---

## Pattern chiave — Wait node e dati pesanti

In n8n, il **Wait node** (On Form Submission) non propaga il payload dell'esecuzione originale quando il workflow riprende. Con payload grandi (dataframe JSON con migliaia di righe) la perdita del dato è sistematica.

**Regola**: ogni nodo dopo un Wait che necessita di dataframe deve leggerli esplicitamente richiamando il nodo precedente:

```javascript
// ✅ CORRETTO (Recupero diretto dal nodo target)

// Esempio per il nodo Training (dopo HITL 1):
const dati_training = $("Feature Selection").item.json;

// Esempio per il nodo Compensation (dopo HITL 2):
const eval_data = $("Evaluation").item.json;
const original_data = $("Feature Selection").item.json;


// ❌ ERRATO ($input dopo il Wait ha solo i dati del Form)
const dati = $input.first().json;
```

---

## Struttura repository

```text
/
├── docker-compose.yml
├── n8n_workflow_export.json
├── services/
│   ├── ingestion/        (main.py, requirements.txt, Dockerfile)
│   ├── preprocessing/    (main.py, requirements.txt, Dockerfile)
│   ├── feature-selection/(main.py, requirements.txt, Dockerfile)
│   ├── training/         (main.py, requirements.txt, Dockerfile)
│   ├── evaluation/       (main.py, requirements.txt, Dockerfile)
│   └── compensation/     (main.py, requirements.txt, Dockerfile)
└── data/
    ├── TE1.csv – TE4.csv
    ├── TI.csv
    └── Displacements.csv
```

