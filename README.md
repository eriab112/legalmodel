# NAP Legal AI Advisor

AI-drivet beslutsstöd för Sveriges Nationella plan för moderna miljövillkor (NAP) för vattenkraft. Systemet kombinerar tre komponenter: en multi-agent RAG-chatbot med domänspecifika agenter, en binär riskklassificerare baserad på domänanpassad KB-BERT, samt en analytisk dashboard med tre flikar.

---

## Systemöversikt

### Dashboard (Streamlit)

Appen har tre flikar:

- **Översikt** — Nyckeltal, riskfördelning, domstolsstatistik och Plotly-diagram
- **Utforska** — Sök och filtrera domstolsbeslut, lagstiftning och ansökningar; riskprediktion på enskilda beslut
- **AI-assistent** — Multi-agent RAG-chatbot med Gemini LLM och källhänvisning

### Multi-agent-arkitektur

Chatboten använder tre specialiserade agenter plus en syntesrouter:

| Agent | Domän | Dokumenttyper |
|-------|-------|---------------|
| Domstolsagent | Domstolsbeslut och praxis | Beslut, ansökningar |
| Svensk rättsagent | Miljöbalken, NAP, HaV-vägledningar | Lagstiftning, ansökningar |
| EU-agent | Vattendirektivet, CIS guidance | EU-lagstiftning |
| Syntesrouter | Tvärgående frågor | Alla dokumenttyper |

Routern klassificerar frågor med nyckelordsbaserad scoring och dirigerar till rätt agent. Vid signaler från 2+ domäner aktiveras syntesagenten som sammanställer svar från alla tre.

### Binär riskklassificerare

- **Modell:** KB/bert-base-swedish-cased (110M parametrar), domänanpassad med DAPT
- **Accuracy:** 80% | **HIGH_RISK recall:** 100% | **F1 macro:** 0.80
- **Konservativ bias:** modellen flaggar osäkra fall som HIGH_RISK (inga missade högriskfall)

### Kunskapsbas

113 dokument totalt:

- 50 domstolsbeslut (44 etiketterade: 21 HIGH_RISK, 23 LOW_RISK)
- 37 lagstiftningsdokument (svensk lagstiftning + EU-direktiv)
- 26 ansökningar

---

## Modellpipeline

### DAPT (Domain-Adaptive Pre-Training)

MLM-förträning på 96 juridiska dokument (domar, lagstiftning, ansökningar). Eval loss: 1.604 → 1.353.

### Binär klassificering

- **Dataset:** 44 märkta beslut (21 HIGH_RISK, 23 LOW_RISK), omklassificerade från ursprunglig 3-klass (HIGH/MEDIUM/LOW)
- **Split:** Train 30, Val 7, Test 7
- **Träning:** Inverse-frequency class weights, gradient checkpointing (4 GB VRAM)

### Utvärdering

| Metrik | Resultat |
|--------|----------|
| Accuracy | 80% |
| F1 macro | 0.80 |
| HIGH_RISK recall | 100% |
| LOW_RISK recall | 60% |

Modellen har konservativ bias — osäkra fall klassificeras som HIGH_RISK, vilket ger noll missade högriskbeslut.

---

## Snabbstart

### Förutsättningar

- Python 3.10+
- pip

### Installation

```bash
cd legalmodel

# Virtuell miljö (rekommenderas)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS

# Beroenden
pip install -r nap-legal-ai-advisor/requirements.txt
```

### Konfigurera

```bash
copy .env.example .env
# Redigera .env: sätt GEMINI_API_KEY (krävs för AI-assistenten)
```

### Starta

```bash
streamlit run nap-legal-ai-advisor/app.py
```

Körs alltid från **repo root** (appen använder relativa sökvägar).

### Företagsmiljö med SSL-proxy

Förladda modeller för miljöer med proxy/SSL-restriktioner:

```bash
python scripts/download_models.py
```

---

## Repostruktur

```
legalmodel/
├── nap-legal-ai-advisor/       # Streamlit-dashboard
│   ├── backend/                # agents, rag_system, search_engine, risk_predictor, llm_engine
│   ├── integration/            # chat_handler, search_handler, shared_context
│   ├── ui/                     # overview, explorer, chat, search interfaces
│   └── utils/                  # data_loader, ssl_fix, timing
├── scripts/                    # Datapipeline-script
├── tests/                      # Pytest-testsvit
├── Data/processed/             # Bearbetade JSON-dataset
├── models/                     # Tränade modeller (ej i git)
├── evaluation_reports/         # Modellutvärderingsresultat
├── run_dapt.py                 # DAPT-förträning
├── run_finetune.py             # Fine-tuning (3-klass, äldre)
├── run_finetune_binary.py      # Binär fine-tuning
├── run_evaluate.py             # Utvärdering (3-klass, äldre)
├── run_evaluate_binary.py      # Binär utvärdering
├── run_binary_pipeline.py      # Komplett binär pipeline
├── run_dashboard.bat           # Snabbstart Windows
├── pyproject.toml              # Projektconfig, pytest, verktyg
└── .env.example                # Miljövariabler (kopieras till .env)
```

---

## Data

| Datamängd | Antal | Detaljer |
|-----------|-------|----------|
| Domstolsbeslut | 50 totalt | 44 etiketterade: 21 HIGH_RISK, 23 LOW_RISK |
| Lagstiftning | 37 | Svensk lagstiftning + EU-direktiv |
| Ansökningar | 26 | Verksamhetsutövares ansökningar |

### Domstolsfördelning

| Domstol | Antal |
|---------|-------|
| Växjö | 22 |
| Vänersborg | 12 |
| Östersund | 7 |
| Nacka | 5 |
| Umeå | 4 |

38 beslut är MÖD-överklaganden med `originating_court`-metadata.

---

## Tester

Testsviten innehåller **132 enhetstester** (exklusive 2 slow-markerade) med mockade modeller och Streamlit-stub — varken GPU eller datafiler krävs.

```bash
# Kör hela testsviten
python -m pytest tests/ -v -m "not slow"

# Med täckningsrapport
python -m pytest tests/ --cov=backend --cov=integration --cov=utils
```

| Fil | Testar |
|-----|--------|
| `test_agents.py` | Nyckelordsrouter, domänklassificering, multi-agent-routing |
| `test_data_loader.py` | DecisionRecord, DataLoader-queries, etiketter, domstolar, datumintervall |
| `test_knowledge_base.py` | DocumentRecord, KnowledgeBase, corpus stats |
| `test_llm_engine.py` | Context-formatering, Gemini engine initialization |
| `test_rag_system.py` | Intent-routing, formatering, keyword-matchning, RAG-konstanter |
| `test_risk_predictor.py` | Softmax, chunking, label-mappning, PredictionResult (binär) |
| `test_search_engine.py` | Textchunking, sökning med filter, deduplicering, liknande beslut |
| `test_integration.py` | SharedContext, ChatHandler, SearchHandler |
| `test_startup_integration.py` | Sökvägsresolution, dataladdning, KnowledgeBase-laddning |

---

## Kända begränsningar

- **Data:** 44 märkta beslut är litet för deep learning — generalisering osäker
- **SSL:** Workaround i `utils/ssl_fix.py` för företagsmiljöer med SSL-proxy — inte lämpligt för produktion utan korrekt CA-certifikat
- **Säkerhet:** Ingen autentisering eller rate limiting
- **Router:** Nyckelordsbaserad routing (inte ML-baserad) — kan missa komplexa tvärgående frågor

---

## Licens och referenser

- Domar och lagstiftning: enligt gällande offentlighetsprincip och källor
- **Modellbas:** KB/bert-base-swedish-cased (Hugging Face), Google Gemini (RAG)
- Akademisk referens: **Sundin et al. (2026)**. *Insights from a nation-wide environmental relicensing of hydropower facilities in Sweden: a review of court verdicts from a biological perspective.* Knowl. Manag. Aquat. Ecosyst. 2026, 427, 6. https://doi.org/10.1051/kmae/2025034
