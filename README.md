# Thermal Compensation Pipeline

A microservices-based ML pipeline for estimating and compensating thermal deformation on a 5-axis CNC machining center (CMS VM-30K), built with real industrial sensor data.

> **Note:** This project contains both Python API services and JavaScript code nodes. The JavaScript snippets handle conditional branching and user input processing within n8n's orchestration layer (see `n8n_workflow_export.json`).

## Context

Precision machining centers experience thermal drift during operation — as the machine heats up, structural components expand, causing positional errors in the order of tens of micrometers. This pipeline processes real-time temperature data from 28 sensors and predicts the resulting displacement, enabling the machine to apply corrective offsets automatically.

The pipeline was developed using real sensor data collected from a production machine (batch sample from a thermal test cycle). The system is designed as a set of independent Python API services orchestrated by [n8n](https://n8n.io), with human-in-the-loop (HITL) checkpoints that allow operators to validate results before applying compensation to the machine.

## Architecture

```text
[Trigger]
    │
    ▼
[Ingestion :8001]       ← loads sensor CSV + displacement data
    │
    ▼
[Preprocessing :8002]   ← interpolation, relative values, merge T1-T28
    │
    ▼
[Feature Selection :8003] ← hierarchical clustering (HCA), representative selection
    │
    ▼
[HITL 1 — Form]         ← operator verifies machine/sensor status
    │
    ├── no_problem ──────────────────────────────────────────────┐
    │                                                             │
    ├── sensor_fault → [sensor removal] → [Feature Selection 2]  │
    │                                                             │
    └── machine_anomaly → [STOP + log]                 → [Training :8004]
                                                                  │
                                                                  ▼
                                                          [Evaluation :8005]
                                                                  │
                                                                  ▼
                                                      [LLM Gemini 2.5 Flash] ← Report generation
                                                                  │
                                                                  ▼
                                                          [HITL 2 — Form] ← Operator approves/rejects offsets
                                                                  │
                                                                  ├── Reject → [STOP]
                                                                  │
                                                                  └── Approve → [Compensation :8006]
```

## Services

| Service | Port | Endpoint | Description |
|---|---|---|---|
| `ingestion` | 8001 | `POST /ingest` | Loads TE1-TE4, TI, Displacements.csv and converts timestamps to Unix epoch |
| `preprocessing` | 8002 | `POST /preprocess` | Interpolates DataFrames onto a common time axis, computes relative values, merges T1-T28 |
| `feature-selection` | 8003 | `POST /feature-selection` | Hierarchical clustering (ward + euclidean) on transposed T matrix, returns cluster_map and df_clustered |
| `training` | 8004 | `POST /train` | Trains MLRA (on df_clustered) and LASSO with GridSearch (on df_tall), KFold CV |
| `evaluation` | 8005 | `POST /evaluate` | Computes MAE, MSE, RMSE, R², % reduction vs baseline, generates comparison plot |
| `compensation` | 8006 | `POST /compensate` | Applies the validated model to compute compensation offsets in µm |

All services also expose `GET /health`.
Internal Docker network: `pipeline` (bridge).

## Getting Started

```bash
docker-compose up --build
```

n8n is available at `http://localhost:5678`.

## API Reference

### `POST /ingest`

Reads CSV files from the `DATA_DIR` directory (default `/app/data`).

**Expected files:**

| File | Sensors | Timestamp format |
|---|---|---|
| `TE1.csv` – `TE4.csv` | T1–T12 (external) | `%d/%m/%Y %H.%M.%S` |
| `TI.csv` | T13–T28 (internal) | `%d/%m/%Y %H:%M:%S.%f` |
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

### `POST /preprocess`

Interpolates `df_text` and `df_dis` onto the time axis of `df_tint` (master).
Removes NaN values, normalizes to relative values (including TIME), merges T1-T28 → `df_tall`.

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

### `POST /feature-selection`

Hierarchical clustering (ward + euclidean) on transposed T sensors.
The representative of each cluster is the **first element** by index.
Also generates a dendrogram as base64 PNG.

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

### `POST /train`

Trains two models in parallel via KFold CV (default: 10 folds, shuffle=True):
- **MLRA** (Multiple Linear Regression) — on `df_clustered` (HCA features)
- **LASSO** — on `df_tall` (all 28 sensors), with GridSearch on `alpha ∈ logspace(-4, 0, 30)`

Target is the `displ` column from `df_dis` (default index 3 = D3).

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

### `POST /evaluate`

Computes metrics for both models and generates a comparison plot (base64 PNG).
Automatically suggests the model with lower RMSE.

**Request body:** full output from `/train`

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
    "note": "% RMSE reduction vs std(y_true). >70% = good, >50% = acceptable"
  },
  "suggested_model": "LASSO",
  "comparison_plot": "<base64 PNG>"
}
```

### `POST /compensate`

Applies the approved model to compute thermal compensation offsets in µm.
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

## HITL 1 — Machine Status Verification

### Purpose

Allows the operator to flag known issues **before** training, preventing corrupted data from influencing the model.

### Form (Wait node — On Form Submission)

| Field | Type | Values |
|---|---|---|
| `Machine status` | Dropdown | `no_problem` / `sensor_fault` / `machine_anomaly` |
| `Sensor to exclude` | Text | e.g. `T17` (only if status = `sensor_fault`) |
| `Additional notes` | Textarea | Optional |

### Branch A — No problem

→ `Training`

### Branch B — Sensor fault

**`sensor removal`** — removes the sensor from `df_tall` and reduces `num_clusters`:

```javascript
const fs = $("Feature Selection").item.json;
const toExclude = ($input.first().json["Sensor to exclude"] || "").trim().toUpperCase();

const dfTall = { ...fs.df_tall };
if (toExclude) delete dfTall[toExclude];

const numClusters = Math.max(2, Object.keys(fs.cluster_map).length - 1);

return [{ json: {
  df_tall:          dfTall,
  df_dis:           fs.df_dis,
  num_clusters:     numClusters,
  excluded_sensor:  toExclude
}}];
```

**`Feature Selection 2`** — same `/feature-selection` endpoint, on `df_tall` without the faulty sensor.
Produces a new optimized `cluster_map` without the removed sensor.

> ⚠️ **Important**: removing a cluster representative eliminates all thermal coverage for that group. The resulting model cannot use that sensor during inference either. Use only for permanent faults.

### Branch C — Machine anomaly

→ `STOP` (alert logging and operator notification)

## HITL 2 — LLM-Assisted Decision Support

### Purpose

Bridges the cognitive gap between backend mathematical results (RMSE, R²) and the machine operator, delegating the final compensation decision to a qualified human actor.

### Flow

1. **Payload filter**: A Code node removes the base64 plot string to avoid saturating the LLM context window.
2. **Report generation (Gemini 2.5 Flash)**: The model receives evaluation metrics and produces a diagnostic report (best model, % error reduction, residual RMSE).
3. **Operator form (Wait node)**: The workflow pauses for human review.

### Form (Wait node — On Form Submission)

| Field | Type | Values / Description |
|---|---|---|
| `Diagnostic Report` | HTML/Text | Read-only (LLM-generated text) |
| `Decision` | Dropdown | `approve_compensation` / `reject` |

### Branch A — Approve

The workflow retrieves the original JSON from Evaluation and sends it to the `Compensation` service.

### Branch B — Reject

→ `STOP` (workflow halted, machine protected from incorrect offsets)

## Key Pattern — Wait Node and Large Payloads

In n8n, the **Wait node** (On Form Submission) does not propagate the original execution payload when the workflow resumes. With large payloads (JSON dataframes with thousands of rows), data loss is systematic.

**Rule**: every node after a Wait that needs dataframes must explicitly reference the upstream node:

```javascript
// ✅ CORRECT (direct reference to target node)
const training_data = $("Feature Selection").item.json;
const eval_data = $("Evaluation").item.json;

// ❌ WRONG ($input after Wait only contains form data)
const data = $input.first().json;
```

## Repository Structure

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

## Tech Stack

- **Python** (FastAPI) — all microservices
- **Docker / docker-compose** — containerization and networking
- **n8n** — workflow orchestration with conditional branching and HITL
- **scikit-learn** — MLRA, LASSO, GridSearchCV, KFold
- **scipy** — hierarchical clustering
- **Gemini 2.5 Flash** — LLM-powered diagnostic report generation
