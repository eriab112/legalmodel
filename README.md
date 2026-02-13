# NAP Legal Model

**Stöd för rådgivning kring Nationella planen för moderna miljövillkor (NAP)** – vattenkraft, miljödomstolar och svensk miljörätt. Detta repo innehåller datapipeline, tränade modeller och **NAP Legal AI Advisor** (Streamlit-app) med semantisk sökning, riskindikation (HIGH/MEDIUM/LOW) och Gemini-baserad RAG över domstolsbeslut, lagstiftning och ansökningar.

---

## Vad finns i repot?

| Del | Beskrivning |
|-----|-------------|
| **nap-legal-ai-advisor/** | Streamlit-app: **Chat** (Gemini RAG) och **Sök** (semantisk sökning över domar, lagstiftning och ansökningar). Riskprediktion med LegalBERT (HIGH/MEDIUM/LOW). KnowledgeBase med tre dokumenttyper. |
| **scripts/** | Pipeline: rensning av domtexter → etiketter → träning → utvärdering. Plus Sundin-feature-extraktion, weak labels, DAPT-korpus, PDF-extraktion m.m. 13 script, ~3 800 rader. |
| **tests/** | 94 enhetstester (pytest) för backend, integration och utils. Mockar Streamlit och modeller — kräver varken GPU eller datafiler. |
| **Data/** | Rådata (Domar, Ansökningar, Lagstiftningdiverse) och **Data/processed/** med JSON som appen och scripten använder. |
| **models/** | Fine-tunad KB-BERT (5-fold CV), DAPT-checkpoints och fine-tuned-efter-DAPT. Appen använder **fold_4**. |
| **run_dapt.py / run_finetune.py / run_evaluate.py** | Phase A-script för DAPT pre-training, fine-tuning med weighted loss + class weights, och utvärdering. |
| **evaluation_reports/** | Utvärderingsrapporter: överfittning-analys, slutresultat, iterationslogg. |

Appen läser från `Data/processed/` (cleaned_court_texts, labeled_dataset, linkage_table, lagtiftning_texts, ansokan_texts) och från `models/`.

---

## Snabbstart

### Förutsättningar

- Python 3.10+ (rekommenderas)
- För träning av modell: GPU med CUDA rekommenderas (~5 h för 05)

### Installation

```bash
# Klona och gå till repo root
cd legalmodel

# Virtuell miljö (rekommenderas)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS

# Beroenden för appen (och för de flesta script)
pip install -r nap-legal-ai-advisor/requirements.txt

# Utvecklingsverktyg (valfritt)
pip install pytest pytest-cov black flake8
```

För PDF-extraktion (t.ex. Sundin-script eller `extract_pdf_text.py`):
`pip install pymupdf`

### Starta appen

**Kör alltid från repo root** (appen använder relativa sökvägar mot root):

```bash
# Kopiera och konfigurera miljövariabler
copy .env.example .env
# Redigera .env: sätt GEMINI_API_KEY för RAG-chat (valfritt), MODEL_PATH vid behov

streamlit run nap-legal-ai-advisor/app.py
```

Öppna http://localhost:8501. Första gången laddas MiniLM och byggs sökindex (~45 s).

### Köra datapipelinen (rå text → modell)

```bash
python scripts/02_clean_court_texts.py
python scripts/03_create_labeled_dataset.py
# Valfritt: python scripts/04_link_to_viss.py   # kräver nap_model-main
python scripts/05_finetune_legalbert.py
python scripts/06_evaluate_basic.py
```

---

## Repostruktur (översikt)

```
legalmodel/
├── README.md                    # Denna fil
├── nap-legal-ai-advisor/       # Streamlit-app (Chat + Sök, riskprediktion)
│   ├── backend/                # risk_predictor, search_engine, rag_system, llm_engine
│   ├── integration/            # chat_handler, search_handler, shared_context
│   ├── ui/                     # chat_interface, search_interface, styles
│   └── utils/                  # data_loader (DataLoader + KnowledgeBase), ssl_fix
├── scripts/                     # Pipeline och hjälpscript (02–06 + Sundin, DAPT, m.m.)
├── tests/                       # 94 enhetstester (pytest)
├── Data/
│   ├── Domar/data/processed/court_decisions/   # Rå TXT (domar)
│   ├── Ansökningar/                            # PDF (ansökningar + några domar)
│   ├── Lagstiftningdiverse/                    # Lagstiftning, riktlinjer
│   └── processed/                              # cleaned_court_texts.json, labeled_dataset.json, m.m.
├── models/
│   ├── nap_legalbert_cv/       # 5-fold LegalBERT (fold_4 används i appen)
│   ├── nap_dapt/               # DAPT pre-training checkpoints + final
│   └── nap_final/              # Fine-tuned efter DAPT (3 epoker, class weights)
├── evaluation_reports/          # Utvärderingsresultat och diagnostik
├── run_dapt.py                  # DAPT pre-training (Phase A, steg 6)
├── run_finetune.py              # Fine-tuning efter DAPT (Phase A, steg 7)
├── run_evaluate.py              # Utvärdering på testset
├── pyproject.toml               # Projektconfig, pytest, black, flake8
├── .env.example                 # Miljövariabler: MODEL_PATH, GEMINI_API_KEY, GEMINI_MODEL
└── [dokumentation .md]          # STRATEGY_AND_PHASE_A, CONTEXT_HANDOVER, m.m.
```

---

## Nuvarande datatillstånd

| Datamängd | Antal | Detaljer |
|-----------|-------|----------|
| **cleaned_court_texts.json** | **50 beslut** | Alla med `text_full`, `key_text`, `sections`, `metadata` |
| **labeled_dataset.json** | **44 märkta** | Train: 30, Val: 7, Test: 7 |
| **Etikettfördelning** | | HIGH_RISK: 8 (18%), MEDIUM_RISK: 26 (59%), LOW_RISK: 10 (23%) |
| **Exkluderade** | 6 | Ej vattenkraft (m8024-05, m7708-22, m899-23, m3273-22, m2479-22, m2024-01) |

### Datahantering

- **Rå domar:** TXT-filer i `Data/Domar/data/processed/court_decisions/`. Nya domar läggs här; därefter körs **02** så att `cleaned_court_texts.json` uppdateras.
- **Etiketter:** Nya/ändrade etiketter läggs i **`Data/processed/label_overrides.json`** (`{ "decision_id": "HIGH_RISK" }`). **Redigera inte** `labeled_dataset.json` manuellt – kör **03** så byggs train/val/test om.
- **Nyckelstrukturer:**
  - `cleaned_court_texts.json` → `decisions[]` med `id`, `text_full`, `key_text`, `sections`, `metadata`.
  - `labeled_dataset.json` → `splits.train`, `splits.val`, `splits.test` (inga toppnivå-`decisions`).

---

## Modellprestanda

### Baseline: 5-fold CV (fold_4 används i appen)

**Bas:** KB/bert-base-swedish-cased (110M parametrar), 5-fold CV, sliding window 512/256.

| Metrik | Medel | Std |
|--------|-------|-----|
| Doc accuracy | 0.65 | 0.094 |
| Doc F1 macro | 0.42 | 0.215 |

| Klass | Precision | Recall | F1 |
|-------|-----------|--------|----|
| HIGH_RISK | 0.40 | 0.30 | 0.33 |
| MEDIUM_RISK | 0.64 | 0.95 | 0.76 |
| LOW_RISK | 0.13 | 0.20 | 0.16 |

Stark MEDIUM_RISK-bias. Modellen ska användas som **indikator**, inte beslutsunderlag.

### Phase A: DAPT + Fine-tuning

**DAPT:** MLM pre-training på 96 juridiska dokument (1.42M ord). Eval loss: 1.604 → 1.353.

**Fine-tuning:** 3 epoker, inverse-frequency class weights, gradient checkpointing (4GB VRAM). 30 dokument + 12 weak labels.

| Metrik | DAPT-modell | Baseline |
|--------|------------|----------|
| Test accuracy (7 dok) | 57.1% | 65.0% (5-fold avg) |
| F1 macro | 0.46 | 0.42 |
| LOW_RISK F1 | **0.80** | 0.16 |
| HIGH_RISK F1 | 0.00 | 0.33 |

DAPT-modellen slog inte baseline på accuracy men förbättrade LOW_RISK-detektion kraftigt (F1: 0.16 → 0.80). Testsettet (7 dokument) är för litet för tillförlitliga slutsatser — varje dokument = 14.3% accuracy. Se `evaluation_reports/final_report.json` för fullständig analys.

---

## Script

| Script | Syfte |
|--------|--------|
| **02_clean_court_texts.py** | Rensa OCR, segmentera sektioner, bygg key_text → `cleaned_court_texts.json` |
| **03_create_labeled_dataset.py** | Bygg train/val/test från cleaned + `label_overrides.json` → `labeled_dataset.json` |
| **04_link_to_viss.py** | Länka vattenförekomster till VISS (kräver nap_model-main) |
| **05_finetune_legalbert.py** | 5-fold fine-tuning KB-BERT → `models/nap_legalbert_cv/` |
| **06_evaluate_basic.py** | Sammanfatta träningsresultat, confusion, rapporter |
| **sundin_feature_extraction.py** | Extrahera Sundin-features (passage, flöde, övervakning, kostnad) → `decision_features_sundin2026.json` |
| **sundin_validation.py** | RF feature importance (5-fold) + KMeans-klustring som diagnostik |
| **weak_labels_applications.py** | Weak labels för ansökningar → `weakly_labeled_applications.json` |
| **build_dapt_corpus.py** | Bygg DAPT-korpus (lagstiftning + ansökningar + domar) → `dapt_corpus.json` |
| **add_4_new_domar.py** | Hitta 4 dom-PDF i Ansökningar, kopiera och extrahera till TXT |
| **extract_pdf_text.py** | Extrahera text från en PDF till .txt (PyMuPDF) |
| **extract_all_pdfs.py** | Extrahera text från PDF i Ansökningar + Lagstiftningdiverse → JSON |
| **run_dapt.py** *(root)* | DAPT pre-training med `dapt_corpus.json` (Phase A, steg 6) |
| **run_finetune.py** *(root)* | Fine-tuning efter DAPT med starka + weak labels (Phase A, steg 7) |
| **run_evaluate.py** *(root)* | Utvärdering på testset efter omträning |

Alla script körs från **repo root**, t.ex. `python scripts/02_clean_court_texts.py`.

---

## Strategi och akademisk grund

- **Etiketter:** HIGH_RISK / MEDIUM_RISK / LOW_RISK, satta utifrån domslut, kostnader och åtgärder (inte enbart "tillstånd/avslag").
- **Sundin et al. 2026** (kmae250140) används som **feature-taxonomi**: vad som ska extraheras (nedströms/uppströms passage, flöde, övervakning). **Vikter och trösklar** lärs från de märkta besluten (hybrid: "Sundin säger vad vi tittar på, datan säger hur vi viktar det").
- **Phase A:** +4 domar, Sundin-features, weak labels för ansökningar (endast i träning), DAPT-korpus, omträning. Detaljer och implementationregler i dokumentationen nedan.

---

## Dokumentation

| Dokument | Innehåll |
|----------|----------|
| **SYSTEM_HANDOVER_COMPLETE.md** | Komplett systemöversikt med alla detaljer — data, modell, tester, config, JSON-strukturer, nästa steg. **Bäst för handover till ny utvecklare/Claude.** |
| **STRATEGY_AND_PHASE_A.md** | Samlad strategi (hybrid: Sundin + datadriven viktning) och Phase A: kritiska fix, ordning, risker. |
| **CONTEXT_HANDOVER_FOR_STRATEGY.md** | Full kontexthandover: vad som gjorts, vad som inte ändrats, sökvägar, regler. |
| **PIPELINE_INTEGRATION_RESULTS.md** | Resultat efter integration av 4 nya domar: 50/44, Sundin-features, RF importance. |
| **SYSTEM_REVIEW.md** | Arkitektur, kända begränsningar, förbättringsförslag (ensemble, binär klass, kalibrering, UI, säkerhet). |
| **COMPLETE_DATA_INVENTORY.md** | Datakällor (Domar, Ansökningar, Lagstiftningdiverse), fynd, DAPT-möjligheter. |
| **USER_GUIDE.md** | Användarguide för Streamlit-appen (chat, sök, felsökning). |

---

## Kända begränsningar

- **Modell:** ~65 % genomsnittlig accuracy (5-fold); stark MEDIUM_RISK-bias. Endast **fold_4** används i appen (ingen ensemble).
- **Data:** 44 märkta beslut är litet för 110M-parametrar; generalisering osäker. MEDIUM_RISK överrepresenterad (59 %).
- **App:** Gemini LLM-engine integrerad (RAG med källhänvisning), men inte ännu kopplad till chatt-UI. Sök indexerar nu beslut + lagstiftning + ansökningar. Ingen auth, rate limit eller strukturerad logging.
- **Säkerhet:** SSL-workaround i `utils/ssl_fix.py` och i 05 – inte lämpligt för produktion utan korrekt proxy/CA.

Mer detaljer: **SYSTEM_REVIEW.md**.

---

## Nästa steg

Phase A (DAPT + fine-tuning) är genomförd. Kvarvarande:

1. **Koppla Gemini till chatt-UI:** `llm_engine.py` är redo — integrera `GeminiEngine` i `chat_handler.py` för RAG-baserad fråga/svar.
2. **Fler märkta dokument:** Mer data är den viktigaste förbättringen för modellprestanda (44 → 100+ rekommenderas).
3. **5-fold CV för DAPT-modell:** Rättvis jämförelse med baseline som också använder 5-fold CV.
4. **Phase B:** Multi-task (aux-heads från Sundin), RF(Sundin) + BERT-ensemble.

Full ordning och implementationregler: **STRATEGY_AND_PHASE_A.md** och **CONTEXT_HANDOVER_FOR_STRATEGY.md**.

---

## Tester

Testsviten täcker backend-modulerna (`risk_predictor`, `search_engine`, `data_loader`, `rag_system`) och integrationslogik (`shared_context`, `chat_handler`, `search_handler`). 94 enhetstester med mockade modeller och Streamlit-stub — inga GPU:er eller datafiler behövs.

```bash
# Kör hela testsviten
python -m pytest tests/ -v

# Med täckningsrapport (kräver pytest-cov)
python -m pytest tests/ --cov=backend --cov=integration --cov=utils
```

Testerna finns i `tests/`:

| Fil | Testar |
|-----|--------|
| `test_risk_predictor.py` | Softmax, chunking, label-mappning, PredictionResult |
| `test_search_engine.py` | Textchunking, sökning med filter, deduplicering, liknande beslut |
| `test_data_loader.py` | DecisionRecord, DataLoader-queries (etikett, domstol, datumintervall) |
| `test_rag_system.py` | Intent-routing, formateringsmetoder, keyword-matchning |
| `test_integration.py` | SharedContext sessionhantering, ChatHandler, SearchHandler |

---

## Utveckling

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r nap-legal-ai-advisor/requirements.txt
pip install pytest pytest-cov  # utvecklingsverktyg
```

### Projektstruktur

Konfiguration finns i `pyproject.toml` (pytest, black, flake8). Miljövariabler sätts i `.env` (se `.env.example`):

- `MODEL_PATH` – Sökväg till riskmodellen (default: `models/nap_legalbert_cv/fold_4/best_model`)
- `GEMINI_API_KEY` – API-nyckel för Gemini LLM (valfritt, krävs för RAG-chat)
- `GEMINI_MODEL` – Gemini-modell (default: `gemini-2.5-flash`)

---

## Licens och referenser

- Domar och lagstiftning: enligt gällande offentlighetsprincip och källor.
- **Modellbas:** KB/bert-base-swedish-cased (Hugging Face), Google Gemini (RAG).
- Akademisk referens för features och etiketteringsram: **Sundin et al. (2026)**. *Insights from a nation-wide environmental relicensing of hydropower facilities in Sweden: a review of court verdicts from a biological perspective.* Knowl. Manag. Aquat. Ecosyst. 2026, 427, 6. https://doi.org/10.1051/kmae/2025034
