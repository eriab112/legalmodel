# NAP Legal AI — Komplett systemöversikt för handover

**Datum:** 2026-02-13
**Status:** Repo-förbättringar (tester, dokumentation, konfiguration) genomförda. Redo för Phase A-fortsättning.

---

## 1. Vad är det här?

**NAP Legal AI Advisor** — ett Streamlit-baserat beslutstödsverktyg för svenska miljödomstolsbeslut inom ramen för Nationella planen för moderna miljövillkor (NAP/vattenkraft).

Systemet erbjuder:
- **Chat-läge:** Frågor/svar om domstolsbeslut (keyword-routing, inte LLM)
- **Sök-läge:** Semantisk sökning via `paraphrase-multilingual-MiniLM-L12-v2`
- **Riskindikation:** HIGH/MEDIUM/LOW via fine-tunad KB-BERT (`KB/bert-base-swedish-cased`, 110M parametrar)

**Tech stack:** Python 3.11, Streamlit, PyTorch, Hugging Face Transformers, sentence-transformers, scikit-learn.

---

## 2. Repostruktur och filstorleker

```
legalmodel/                          # repo root
├── nap-legal-ai-advisor/            # Streamlit-app (1 811 rader Python)
│   ├── app.py                       # 156 rader — huvudapplikation
│   ├── requirements.txt             # beroenden
│   ├── backend/
│   │   ├── risk_predictor.py        # 204 rader — LegalBERT riskklassificering
│   │   ├── search_engine.py         # 276 rader — semantisk sökning (MiniLM)
│   │   └── rag_system.py            # 243 rader — RAG med keyword-routing
│   ├── integration/
│   │   ├── chat_handler.py          # 72 rader — chattlogik
│   │   ├── search_handler.py        # 72 rader — sökgränssnitt
│   │   └── shared_context.py        # 89 rader — delad session state
│   ├── ui/
│   │   ├── chat_interface.py        # 89 rader
│   │   ├── search_interface.py      # 255 rader
│   │   └── styles.py               # 158 rader — CSS
│   └── utils/
│       ├── data_loader.py           # 172 rader — DataLoader + DecisionRecord
│       └── ssl_fix.py               # 25 rader — proxy SSL workaround
│
├── scripts/                         # Pipeline-script (3 821 rader Python)
│   ├── 02_clean_court_texts.py      # 550 rader — OCR-rensning, sektionering
│   ├── 03_create_labeled_dataset.py # 613 rader — bygger train/val/test-splits
│   ├── 04_link_to_viss.py           # 450 rader — VISS-koppling (kräver nap_model-main)
│   ├── 05_finetune_legalbert.py     # 614 rader — 5-fold CV fine-tuning
│   ├── 06_evaluate_basic.py         # 385 rader — utvärdering med confusion matrix
│   ├── sundin_feature_extraction.py # 229 rader — Sundin-features
│   ├── sundin_validation.py         # 185 rader — RF importance + KMeans
│   ├── weak_labels_applications.py  # 157 rader — weak labels
│   ├── build_dapt_corpus.py         # 114 rader — DAPT-korpus
│   ├── add_4_new_domar.py           # 103 rader — hitta/kopiera 4 PDF
│   ├── extract_all_pdfs.py          # 338 rader — batch PDF-extraktion
│   ├── extract_pdf_text.py          # 39 rader — enskild PDF
│   └── inventory.py                 # 44 rader
│
├── tests/                           # Testsvit (1 042 rader Python, 94 tester)
│   ├── conftest.py                  # 112 rader — Streamlit stub, fixtures
│   ├── test_risk_predictor.py       # 164 rader — softmax, chunking, labels
│   ├── test_search_engine.py        # 206 rader — sökning, filter, dedup
│   ├── test_data_loader.py          # 195 rader — DataLoader queries
│   ├── test_rag_system.py           # 167 rader — intent-routing, formatering
│   └── test_integration.py          # 198 rader — SharedContext, Chat, Search
│
├── run_dapt.py                      # 128 rader — DAPT pre-training (Phase A)
├── run_finetune.py                  # 242 rader — fine-tuning efter DAPT
├── run_evaluate.py                  # 221 rader — test-utvärdering
│
├── Data/
│   ├── Domar/data/processed/court_decisions/  # ~54 rå TXT-filer
│   ├── Ansökningar/                           # PDF (ansökningar + några domar)
│   ├── Lagstiftningdiverse/                   # lagstiftning, riktlinjer
│   └── processed/                             # pipeline-output (se nedan)
│
├── models/
│   ├── nap_legalbert_cv/            # 5-fold CV-modeller (fold_0–4, ~476 MB/fold)
│   ├── nap_dapt/                    # DAPT checkpoints (checkpoint-12, -18, final)
│   └── nap_final/                   # Fine-tuned efter DAPT (checkpoint-95, -190)
│
├── pyproject.toml                   # projektconfig + pytest + black + flake8
├── .env.example                     # miljövariabler template
├── .gitignore                       # uppdaterad
├── README.md                        # uppdaterad med tester/utveckling
├── USER_GUIDE.md                    # ny — användarguide
├── STRATEGY_AND_PHASE_A.md          # Phase A implementation rules
├── CONTEXT_HANDOVER_FOR_STRATEGY.md # full handover-kontext
├── PIPELINE_INTEGRATION_RESULTS.md  # resultat efter 4 nya domar
├── SYSTEM_REVIEW.md                 # arkitektur + förbättringsförslag
├── COMPLETE_DATA_INVENTORY.md       # datakällor-inventering
└── EXPLORATION_REPORT.md            # historisk data-exploration
```

---

## 3. Current data state

### cleaned_court_texts.json — 50 beslut

Alla 50 har `text_full`, `key_text`, `sections`, `metadata`, `extracted_costs`, `extracted_measures`.

### labeled_dataset.json — 44 beslut

| Split | Antal |
|-------|-------|
| Train | 30 |
| Val | 7 |
| Test | 7 |
| **Totalt** | **44** |

**Etikettfördelning:**

| Etikett | Antal | % |
|---------|-------|---|
| HIGH_RISK | 8 | 18% |
| MEDIUM_RISK | 26 | 59% |
| LOW_RISK | 10 | 23% |

**Exkluderade (ej vattenkraft):** m8024-05, m7708-22, m899-23, m3273-22, m2479-22, m2024-01

### label_overrides.json — 11 entries
6 EXCLUDE + 1 HIGH_RISK override + 4 nya beslutsetiketter.

### Övriga processade filer

| Fil | Storlek | Innehåll |
|-----|---------|----------|
| cleaned_court_texts.json | 13.4 MB | 50 beslut |
| labeled_dataset.json | 1.4 MB | 44 märkta, splits |
| dapt_corpus.json | 11.7 MB | 96 dokument, 1.42M ord (38 lag + 12 ansökningar + 46 domar) |
| ansokan_texts.json | 1.7 MB | ansökningstexter |
| lagtiftning_texts.json | 5.7 MB | lagstiftningstexter |
| weakly_labeled_applications.json | 641.7 KB | 12 ansökningar med weak labels |
| decision_features_sundin2026.json | 33.5 KB | 44 beslut, Sundin-features |
| sundin_feature_importance.json | 1.3 KB | RF importance (5-fold) |
| clustering_validation_report.json | 2.4 KB | KMeans diagnostik, ARI=0.0353 |
| linkage_table.json | 23.2 KB | VISS-koppling |

---

## 4. Modellprestanda (nuvarande, 5-fold CV)

**Bas:** KB/bert-base-swedish-cased, 110M parametrar
**Träning:** 40 beslut (före integration), sliding window 512/256, 5 epoker, effektiv batch 8
**Viktning:** HIGH_RISK 1.67, MEDIUM_RISK 0.58, LOW_RISK 1.48

### Aggregerade resultat

| Metrik | Medel | Std |
|--------|-------|-----|
| Doc accuracy | 0.65 | 0.094 |
| Doc F1 macro | 0.42 | 0.215 |
| Doc precision macro | 0.39 | 0.254 |
| Doc recall macro | 0.48 | 0.186 |

### Per klass

| Klass | Precision | Recall | F1 |
|-------|-----------|--------|----|
| HIGH_RISK | 0.40 ± 0.49 | 0.30 ± 0.40 | 0.33 ± 0.42 |
| MEDIUM_RISK | 0.64 ± 0.09 | 0.95 ± 0.10 | 0.76 ± 0.05 |
| LOW_RISK | 0.13 ± 0.27 | 0.20 ± 0.40 | 0.16 ± 0.32 |

**Känt problem:** Stark MEDIUM_RISK-bias (95% recall; HIGH/LOW har ~30%). Fold-4 (i appen) har bäst spread men är ändå begränsad.

### Sundin RF Feature Importance (topp 5, n=44)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | upstream_has_eel_ramp | 0.158 |
| 2 | upstream_slope_pct | 0.123 |
| 3 | upstream_type_int | 0.109 |
| 4 | downstream_angle_degrees | 0.105 |
| 5 | monitoring_required | 0.104 |

---

## 5. Testsvit

94 enhetstester — alla passerar (2.85 s). Mockar Streamlit och modeller — inga GPU:er eller datafiler behövs.

```
tests/test_data_loader.py      — 14 tester  (DecisionRecord, DataLoader queries)
tests/test_integration.py      — 22 tester  (SharedContext, ChatHandler, SearchHandler)
tests/test_rag_system.py       — 20 tester  (intent-routing, formatering, keywords)
tests/test_risk_predictor.py   — 12 tester  (softmax, chunking, labels, PredictionResult)
tests/test_search_engine.py    — 16 tester  (chunking, sökning, filter, dedup, liknande)
```

Kör med: `python -m pytest tests/ -v`

**conftest.py notis:** Streamlit-stubben använder `types.ModuleType` (inte MagicMock) och skyddas mot dubbelladdning (pytest laddar conftest separat från `from tests.conftest import`).

---

## 6. Konfiguration

### pyproject.toml
- Projektmetadata, dependencies, optional dev/pipeline extras
- pytest: `testpaths = ["tests"]`, `pythonpath = ["nap-legal-ai-advisor"]`
- black: line-length 100, target py310
- flake8: max-line-length 100

### .env.example
```
BATCH_SIZE=4          # sänk till 2 vid CUDA OOM
DEVICE=cpu            # eller cuda
MODEL_PATH=models/nap_legalbert_cv/fold_4/best_model
DATA_DIR=Data/processed
```

### .gitignore
Ignorerar: `models/`, raw data dirs, `.pt`-filer, DAPT datasets, `.env`, `.venv/`, `__pycache__/`, `logs/`, `wandb/`, `.pytest_cache/`, `.coverage`, `.ipynb_checkpoints/`, IDE-filer.

---

## 7. Git-status

**Branch:** master (1 commit: `515b5b4 Initial commit`)

**Modifierade (ej committade):**
- `.gitignore` — utökad
- `README.md` — tillagda sektioner om tester och utveckling

**Nya (ej trackade):**
- `.env.example`
- `USER_GUIDE.md`
- `pyproject.toml`
- `tests/` (hela testsviten)
- `run_dapt.py`, `run_finetune.py`, `run_evaluate.py`
- `PRE_TRAINING_REVIEW.json`

---

## 8. Strategi och nästa steg

### Hybrid-princip
*"Sundin säger vad vi ska titta på, datan säger hur vi ska vikta det."*

Sundin et al. 2026 = feature-taxonomi. Trösklar och vikter lärs från 44 märkta beslut (RF). Klustring som diagnostik, inte auto-relabeling.

### Vad som är genomfört (Phase A delvis)

- ✅ A1: 4 nya domar integrerade (50 cleaned, 44 labeled)
- ✅ A2: Sundin-baserade etiketter för 4 nya (i label_overrides)
- ✅ A3: Feature-extraktion för alla 44 (decision_features_sundin2026.json)
- ✅ A4: Weak labels för 12 ansökningar
- ✅ A5: DAPT-korpus (96 dok, 1.42M ord)
- ⏳ A6: DAPT pre-training — `run_dapt.py` finns men behöver körning/validering
- ⏳ A7: Omträning med starka + weak labels — `run_finetune.py` finns

### Kvarvarande nästa steg

1. **DAPT (A6):** Kör `run_dapt.py` med `dapt_corpus.json`. Använd `datasets.load_dataset("text")` + `DataCollatorForLanguageModeling` (inte `LineByLineTextDataset`). Spara till `models/nap_dapt/final`.

2. **Omträning (A7):** Fine-tune med 44 starka + 12 weak. Val/test **enbart** på 44 starka. Sliding window (512/256). WeightedTrainer med vikter per sample. 5-fold CV.

3. **Phase B — Multi-task:** Auxiliary heads (downstream/upstream/flow/monitoring) med mål från Sundin-features. Aux-loss med låg λ (t.ex. 0.2).

4. **Phase B — Ensemble:** RF(Sundin) + BERT-ensemble; 5-fold-utvärdering.

5. **App-förbättringar:**
   - Ensemble alla 5 folds (inte bara fold_4)
   - Binär klassificering (HIGH vs MEDIUM+LOW) som alternativ
   - Threshold-kalibrering
   - LLM-integration i chatten
   - Auth, rate limiting, strukturerad logging

### Kritiska implementationsregler

- Använd **`text_full`** (inte `cleaned_text`) i all kod
- **Redigera aldrig** `labeled_dataset.json` manuellt — använd `label_overrides.json` + kör `03`
- Behåll **sliding window** (512/256) i both träning och inference
- Weak labels **bara i train**, stratifiera och validera **enbart** på 44 starka
- DAPT: använd `datasets` + DataCollatorForLanguageModeling, inte `LineByLineTextDataset`
- A7 Trainer: Dataset returnerar `input_ids`, `attention_mask`, `labels`, `weights`; WeightedTrainer viktar loss

---

## 9. Nyckelstrukturer i JSON-data

### cleaned_court_texts.json
```json
{
  "decisions": [
    {
      "id": "m1234-22",
      "filename": "Nacka TR M 1234-22 Dom 2024-01-15.txt",
      "text_full": "...",
      "key_text": "...",
      "sections": { "domslut": "...", "yrkanden": "...", ... },
      "metadata": { "court": "Nacka TR", "date": "2024-01-15", "case_number": "M 1234-22" },
      "extracted_measures": ["Fiskväg", "Minimitappning"],
      "extracted_costs": [{"amount": 500000, "currency": "SEK"}]
    }
  ],
  "pipeline_version": "...",
  "source_dir": "..."
}
```

### labeled_dataset.json
```json
{
  "label_distribution": { "HIGH_RISK": 8, "MEDIUM_RISK": 26, "LOW_RISK": 10 },
  "splits": {
    "train": [ { "id": "...", "label": "HIGH_RISK", "key_text": "...", "metadata": {...}, "scoring_details": {...} }, ... ],
    "val": [ ... ],
    "test": [ ... ]
  },
  "excluded_decisions": [ ... ]
}
```

### label_overrides.json
```json
{
  "m8024-05": "EXCLUDE",
  "m3753-22": "HIGH_RISK",
  "m483-22": "LOW_RISK",
  "m2796-24": "MEDIUM_RISK"
}
```

### weakly_labeled_applications.json
```json
{
  "metadata": { ... },
  "applications": [
    { "text": "...", "weak_label": "MEDIUM_RISK", "confidence": 0.7, "weight": 0.7, ... }
  ]
}
```

### decision_features_sundin2026.json
```json
{
  "decisions": [
    {
      "id": "m1234-22",
      "label": "HIGH_RISK",
      "features": {
        "downstream_gap_mm": 15,
        "downstream_angle_degrees": 80,
        "upstream_has_fishway": true,
        "upstream_type_int": 2,
        "cost_msek": 0.5,
        "monitoring_required": true,
        ...
      }
    }
  ]
}
```

---

## 10. Beroenden

### Produktion (nap-legal-ai-advisor/requirements.txt)
```
streamlit>=1.28.0
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.17.0
scikit-learn>=1.3.0
```

### Utveckling (pyproject.toml [dev])
```
pytest>=8.0
pytest-cov>=4.0
black>=24.0
flake8>=7.0
```

### Pipeline (pyproject.toml [pipeline])
```
pymupdf>=1.23.0
datasets>=2.14.0
```

---

## 11. Kända begränsningar

- **Modell:** 65% accuracy (5-fold medel), stark MEDIUM_RISK-bias. Fold_4 enda aktiva.
- **Data:** 44 märkta beslut — litet för 110M-parametrar. MEDIUM_RISK överrepresenterad (59%).
- **App:** Keyword-baserad intent i chatten — ej LLM. Ingen auth/rate limiting.
- **SSL:** `utils/ssl_fix.py` stänger av certifikatverifiering — inte för produktion.
- **Sundin-features:** 0 importance för `downstream_has_screen` och `downstream_bypass_ls` — granska extraktionskvaliteten.
- **Klustring:** ARI 0.035 (mycket låg) — labels fångar kontext bortom Sundin-features.

---

## 12. Viktiga dokument

| Dokument | Syfte |
|----------|-------|
| **STRATEGY_AND_PHASE_A.md** | Implementationsregler, kritiska fix, ordning A1–A7, risker |
| **CONTEXT_HANDOVER_FOR_STRATEGY.md** | Full kontexthandover: vad som gjorts, inte gjorts, sökvägar |
| **PIPELINE_INTEGRATION_RESULTS.md** | Resultat efter 4 nya domar: 50/44, Sundin-features, RF importance |
| **SYSTEM_REVIEW.md** | Nuvarande arkitektur, kända problem, Phase B/C-riktningar |
| **COMPLETE_DATA_INVENTORY.md** | Datakällor, expansionsmöjligheter |
| **USER_GUIDE.md** | Användning av Streamlit-appen |

---

## 13. Akademisk referens

Sundin et al. (2026). *Insights from a nation-wide environmental relicensing of hydropower facilities in Sweden: a review of court verdicts from a biological perspective.* Knowl. Manag. Aquat. Ecosyst. 2026, 427, 6.
https://doi.org/10.1051/kmae/2025034
