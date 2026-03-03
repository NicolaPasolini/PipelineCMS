# Architettura Sistema di Compensazione Termica CNC

## Overview
Sistema modulare a microservizi per la compensazione degli errori termici su macchine CNC tramite modelli predittivi basati su dati sensoriali (28 temperature + 5 displacement).

---

## Microservizi Implementati

### MS1a — Ingestion Service
**Porta:** `8001`  
**Responsabilità:**
- Caricamento CSV raw (TE1-4, TI, Displacements)
- Parsing timestamp → Unix epoch
- Normalizzazione temporale [0, 1] opzionale
- Allineamento strutturale dati

**Endpoint:**
- `POST /ingest` → `{df_text, df_tint, df_dis}`
- `GET /health`

**Stack:** FastAPI + Pandas + NumPy

---

### MS1b — Preprocessing Service
**Porta:** `8002`  
**Responsabilità:**
- Interpolazione lineare (SciPy) su timestamp comune
- Allineamento intervallo temporale [t_min, t_max] comune
- Calcolo ΔT (delta dalla prima lettura) per scaling
- Rimozione NaN e gestione ±inf

**Endpoint:**
- `POST /preprocess` → `{df_tall, df_target, target_col}`
- `GET /health`

**Stack:** FastAPI + Pandas + SciPy

---

### MS2 — Feature Service
**Porta:** `8003`  
**Responsabilità:**
- **HCA (Hierarchical Clustering)** su matrice correlazione 28 sensori → riduzione a ~6 cluster
- Selezione rappresentante per cluster (max correlazione media)
- **VIF Check** (Variance Inflation Factor) → rimozione iterativa sensori multicollineari (soglia 10.0)
- Filtraggio sensori per tipo test (`test_type: Z|Y|X|ETVE`)
- Generazione dendrogramma (PNG base64)

**Endpoint:**
- `POST /select-features` → `{df_clustered, selected_features, vif_scores, dendrogram_png}`
- `GET /health`

**Stack:** FastAPI + SciPy + Statsmodels + Matplotlib

**Note:**
- ETVE usa solo T13-T28 (sensori interni)
- Altri assi usano tutti i 28 sensori

---

### MS3 — Model Service
**Porta:** `8004`  
**Responsabilità:**
- Training 4 modelli regressivi: **OLS, LASSO, SGD, XGBoost**
- 10-fold Cross-Validation per tuning iperparametri
- **ETVE Correction** opzionale (detrending temperatura ambiente)
- Calcolo metriche: MAE, MSE, RMSE, R²
- Grid Search per α (LASSO/SGD) e parametri XGBoost

**Endpoint:**
- `POST /train` → `{model, metrics, cv_rmse, coefficients, predictions}`
- `GET /health`

**Stack:** FastAPI + Scikit-learn + XGBoost

**Parametri chiave:**
- `model`: "OLS" | "LASSO" | "SGD" | "XGBoost"
- `etve_correction`: bool (applica correzione ambiente)
- `ambient_sensors`: lista sensori per calcolo T_amb

---

### MS4 — Compensation Service
**Porta:** `8005`  
**Responsabilità:**
- Assemblaggio matrici errore `{ErrZ, ErrY, ErrX, ErrAmb}`
- Generazione formule PLC-ready (Structured Text/Ladder)
- Scaling unità (µm → mm opzionale)
- Matrice globale sensori × assi

**Endpoint:**
- `POST /build-compensation` → `{matrices, global_matrix, plc_block}`
- `GET /health`

**Stack:** FastAPI + NumPy

**Output:**
```
ErrZ = (0.123456 * T17) + (0.045678 * T4) + ... + intercept
ErrY = ...
```

---

## Orchestrazione N8N (Human-in-the-Loop)

### Nodi Essenziali

#### 1. **HTTP Request Nodes** (5×)
- `Ingest Data` → `POST http://ingestion:8001/ingest`
- `Preprocess` → `POST http://preprocessing:8002/preprocess`
- `Select Features` → `POST http://feature-selection:8003/select-features`
- `Train Model` → `POST http://prediction:8004/train`
- `Build Compensation` → `POST http://compensation:8005/build-compensation`

**Config comune:**
- Authentication: None (rete interna Docker)
- Response Format: JSON
- Timeout: 120s (modelli pesanti)

---

#### 2. **Set Node** (4×)
Preparazione body JSON tra ogni step:
- Dopo Ingest → mappa `df_text`, `df_tint`, `df_dis` in Preprocess
- Dopo Preprocess → passa `df_tall`, `df_target` a Feature Selection
- Dopo Feature → inietta `selected_features` in Train
- Dopo Train → aggrega output multi-asse per Compensation

---

#### 3. **Code Node** (Python/JavaScript)
**Validazione + trasformazioni:**
```javascript
// Esempio: estrai solo sensori con |coeff| > 0.01
const coefficients = $json.extra.coefficients;
const filtered = Object.entries(coefficients)
  .filter(([k,v]) => Math.abs(v) > 0.01)
  .reduce((acc, [k,v]) => ({...acc, [k]:v}), {});
return {filtered_coefficients: filtered};
```

**Uso:** Pulizia coefficienti LASSO/SGD, calcolo metriche custom

---

#### 4. **Switch Node** (decisioni)
**Branch logic per tipo test:**
```
IF $json.test_type == "ETVE"
  → usa solo T13-T28
ELSE
  → usa T1-T28
```

**Branch per selezione modello:**
```
IF $json.metrics.R2 > 0.95
  → procedi a Compensation
ELSE
  → notifica operatore + retry con modello diverso
```

---

#### 5. **Edit Fields (Set) Node**
Rinomina/mappa campi tra servizi:
```
df_clustered → input_data
selected_features → feature_list
target_col → label_column
```

---

#### 6. **Merge Node**
Combina output da training paralleli (es. Z, Y, X):
```
Wait for all 3 branches → merge in array
→ passa a Compensation Service
```

---

#### 7. **Slack/Email Node**
Notifiche HITL:
- Invio dendrogramma (base64 → attachment)
- Alert se R² < soglia
- Report finale matrici compensazione

---

### Workflow N8N Completo (esempio asse Z)

```
[Manual Trigger]
    ↓
[Set: test_type=Z, normalize_time=true]
    ↓
[HTTP: Ingest] ──→ [Set: body preprocess]
    ↓
[HTTP: Preprocess] ──→ [Set: displacement_col=D3]
    ↓
[HTTP: Feature Selection] ──→ [Code: check VIF ok]
    ↓
[Switch: num_features >= 4?]
    ├─ YES → [HTTP: Train LASSO]
    │           ↓
    │       [Code: valida R² > 0.90]
    │           ↓
    │       [HTTP: Build Compensation]
    │           ↓
    │       [Slack: "Modello Z pronto"]
    │
    └─ NO  → [Slack: "Troppi pochi sensori, rivedi clustering"]
```

---

### Pipeline Multi-Asse (3 branch paralleli)

```
[Manual Trigger]
    ↓
[Split: 3 rami Z/Y/X]
    ├─ Branch Z (D3) ──┐
    ├─ Branch Y (D2) ──┼─→ [Merge: wait for all]
    └─ Branch X (D1) ──┘        ↓
                           [HTTP: Compensation con model_Z, model_Y, model_X]
                                ↓
                           [Email: PLC block formulas]
```

---

## Espansione Futura: LLM Agent

### Obiettivo
Agente conversazionale per:
- **Analisi automatica** risultati modelli (interpretazione metriche)
- **Suggerimento modelli** alternativi se performance scarse
- **Generazione report** in linguaggio naturale
- **Debug assistito** (es. "Perché T17 ha coefficiente negativo?")

---

### Architettura Proposta

#### MS5 — LLM Service
**Porta:** `8006`  
**Stack:** FastAPI + LangChain + OpenAI API / Ollama locale

**Endpoint:**
- `POST /analyze` → analisi JSON risultati training
- `POST /chat` → conversazione context-aware
- `POST /suggest` → raccomandazioni modello/features

**Funzionalità:**

1. **Analisi Automatica Output**
```python
# Input: JSON da MS3
{
  "model": "LASSO",
  "metrics": {"R2": 0.87, "RMSE": 12.3},
  "coefficients": {"T17": 0.45, "T4": -0.12}
}

# Output LLM
"Il modello LASSO mostra R²=0.87 (buono ma migliorabile). 
RMSE=12.3µm indica errore residuo accettabile per fresatura. 
T17 (sensore esterno) ha forte impatto positivo, mentre T4 
mostra contributo negativo → verificare posizionamento."
```

---

2. **Suggerimenti Intelligenti**
```python
IF R² < 0.85:
  → "Prova XGBoost per catturare non-linearità"
IF VIF removed > 5 features:
  → "Molti sensori correlati rimossi. Considera PCA invece di HCA"
IF ETVE correction not applied:
  → "Temperatura ambiente varia di ±3°C. Abilita etve_correction"
```

---

3. **Nodo N8N integrato**
```
[HTTP: Train Model]
    ↓
[HTTP: LLM Analyze] ──→ input: training output JSON
    ↓
[Code: parse suggestions]
    ↓
[Switch: action needed?]
    ├─ retry_different_model → loop back to Train
    ├─ manual_review → Slack notification
    └─ proceed → Compensation Service
```

---

4. **RAG (Retrieval Augmented Generation)**
Knowledge base:
- Datasheet sensori (range, risoluzione)
- Manuali macchina CNC (assi, tolleranze)
- Paper di riferimento (compensazione termica)
- Log esecuzioni precedenti

**Esempio query:**
```
User: "Perché T23 viene sempre rimosso dal VIF check?"
LLM → cerca in vector DB → trova:
"T23 (mandrino posteriore) altamente correlato con T24-T26 
(stesso gruppo termico). Normale rimozione per multicollinearità."
```

---

### Tool Calling per LLM

Agent può invocare direttamente microservizi:

```python
# LLM decide: "R² basso, proviamo XGBoost"
tools = [
    {
        "name": "retrain_model",
        "params": {"model": "XGBoost", "n_folds": 10}
    }
]
→ Chiama POST /train con nuovi parametri
→ Analizza nuovo output
→ Confronta metriche
```

---

### Nodo N8N "LLM Decision Loop"

```
[Manual Trigger: "Ottimizza modello asse Z"]
    ↓
[HTTP: Train con config base]
    ↓
[HTTP: LLM Analyze] ──→ streaming response
    ↓
[Code: parse action JSON]
    │
    ├─ action="retry" → loop back con nuovo config
    ├─ action="tune_features" → torna a Feature Selection
    ├─ action="accept" → procedi Compensation
    └─ action="manual" → Slack + stop
```

**Max 3 iterazioni** per evitare loop infiniti.

---

## Stack Tecnologico Completo

| Componente | Tecnologia |
|---|---|
| **Microservizi** | FastAPI 0.115+ |
| **Containerizzazione** | Docker Compose 3.9 |
| **Orchestrazione HITL** | N8N 1.x |
| **Data Processing** | Pandas, NumPy, SciPy |
| **Feature Engineering** | Statsmodels (VIF), SciPy (HCA) |
| **ML Models** | Scikit-learn, XGBoost |
| **Visualizzazione** | Matplotlib (dendrogrammi) |
| **LLM (futuro)** | LangChain + OpenAI / Ollama |
| **Vector DB (RAG)** | ChromaDB / Pinecone |
| **Version Control** | Git + GitHub |

---

## Deployment

### Struttura Repository
```
thesys/
├── docker-compose.yml
├── .gitignore
├── data/                    # escluso da Git (pesante)
│   ├── TE1-4.csv
│   ├── TI.csv
│   └── Displacements.csv
├── services/
│   ├── ingestion/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── requirements.txt
│   ├── preprocessing/       (stesso pattern)
│   ├── feature_selection/
│   ├── prediction/
│   └── compensation/
└── n8n_workflows/           # export JSON workflow
    ├── full_pipeline_Z.json
    └── multi_axis.json
```

### Avvio Sistema
```bash
docker compose up --build    # avvia tutti i servizi
# N8N UI: http://localhost:5678
# Swagger docs: http://localhost:800X/docs (X = 1-5)
```

---

## Metriche di Successo

| Metrica | Target | Note |
|---|---|---|
| **R² test** | > 0.90 | Modello spiega >90% varianza |
| **RMSE** | < 15 µm | Errore residuo sotto tolleranza |
| **Tempo processing** | < 2 min | Pipeline completa end-to-end |
| **Sensori selezionati** | 4-8 | Bilanciamento info/complessità |
| **VIF max** | < 10.0 | No multicollinearità severa |

---

## Roadmap

### ✅ Fase 1 (completata)
- Microservizi base (MS1-MS4)
- Docker Compose
- Git versioning

### 🔄 Fase 2 (in corso)
- N8N workflow HITL
- Testing end-to-end
- Validazione su dati reali

### 📋 Fase 3 (pianificata)
- MS5 LLM Service
- RAG knowledge base
- Auto-tuning iperparametri
- Dashboard monitoring (Grafana)

---

## Note Implementative

### ETVE Correction
Formula applicata in MS3:
```
Err_ETVE = α * (T_amb - T_amb_ref)
Err_totale = Err_modello - Err_ETVE
```
Dove `T_amb = mean(sensori_ambiente)` (es. T1-T12 esterni)

### VIF Iterativo
```python
while max(VIF) > threshold:
    remove feature with highest VIF
    recalculate VIF on remaining features
```

### Grid Search Ranges
- **LASSO α**: [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
- **SGD α**: [1e-5, 1e-4, 1e-3, 1e-2] × penalty [l1, l2, elasticnet]
- **XGBoost**: n_estimators [100, 300], max_depth [3, 5], lr [0.05, 0.1]

---

## Contatti & Manutenzione
- **Repo GitHub**: [https://github.com/NicolaPasolini/thesys](https://github.com/NicolaPasolini/thesys)
- **Autore**: Nicola Pasolini
- **Supervisione**: [Nome supervisore tesi]
- **Anno**: 2026

---

*Documento generato: 3 Marzo 2026*