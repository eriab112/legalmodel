# NAP Legal Model – Onboarding- och analysrapport

**Datum:** 2026-02-12  
**Syfte:** Mental modell av systemet, fokus på scripts, nap-legal-ai-advisor, models, data och dokumentation. Risker, skuld och next steps för ny utvecklare.

---

## 1) Repo-översikt (5–10 punkter)

- **Root README** – `README.md` i repo-roten beskriver syfte, snabbstart, pipeline, script och dokumentation. Övrig huvuddokumentation: `nap_model-main/README.md` (NAP Decision Support), `SYSTEM_REVIEW.md`, `COMPLETE_DATA_INVENTORY.md`, `CONTEXT_HANDOVER_FOR_STRATEGY.md`, `LABELING_STRATEGY_VS_PAPER.md`.
- **Två huvudsakliga “produkter”:**  
  - **nap-legal-ai-advisor/** – Streamlit-app (Chat + Search) med LegalBERT riskprediktion och semantisk sökning över 46 domar; ~1 811 rader Python.  
  - **nap_model-main/** – Separat NAP Decision Support-system (KPIs, VISS, kostnadsmodell R²=0.958, dashboard, 78 scripts, databas v2.17).
- **Data:** Rådata i `Data/` (Domar, Ansökningar, Lagstiftningdiverse). Bearbetad data i `Data/processed/` (cleaned_court_texts.json, labeled_dataset.json, linkage_table.json). Appen läser **endast** från `Data/processed/`.
- **Modeller:** `models/nap_legalbert_cv/` – 5-fold fine-tuned KB-BERT (HIGH/MEDIUM/LOW risk). Endast **fold_4** används i produktion (`risk_predictor.py`). ~2,4 GB modellfiler.
- **Pipeline:** Rå TXT (court_decisions) → `02_clean_court_texts.py` → `03_create_labeled_dataset.py` → `04_link_to_viss.py` (valfritt) → `05_finetune_legalbert.py` → `06_evaluate_basic.py`. Appen använder output från 02, 03, 04 och modeller från 05.
- **Ingen Docker/CI** – Inga Dockerfile/docker-compose eller GitHub Actions i repot. Körning är manuell (Python + Streamlit).
- **Beroenden:** `nap-legal-ai-advisor/requirements.txt` (Streamlit, PyTorch, transformers, sentence-transformers, numpy, pandas, plotly, scikit-learn). Scripts i root använder samma ekosystem + Hugging Face; `05_finetune_legalbert.py` kräver GPU för rimlig träningstid.
- **Externa tjänster:** Ingen API-nyckel för appen (template-RAG, ingen LLM). Hugging Face för KB-BERT och MiniLM (nedladdning vid första körning). SSL-workaround för corporate proxy (`utils/ssl_fix.py`, inaktiverar verifiering – osäker för produktion).
- **Sökvägar:** Appen antar att den körs med **repo root som cwd** (data_loader använder `Path(__file__).parent.parent.parent` → `Data/processed/`). Scripts använder relativa sökvägar från repo root; `extract_all_pdfs.py` har **hårdkodad Windows-sökväg** (`C:\Users\SE2I7A\Desktop\legalmodel\Data`).

---

## 2) Fokus: SYSTEM_REVIEW.md (sammanfattning + åtgärdspunkter)

### Sammanfattning
- **Syfte:** Granskning av Task 4-leverans (NAP Legal AI Advisor) och projektstatus.
- **Arkitektur:** Streamlit-app med backend (search_engine, risk_predictor, rag_system), integration (chat_handler, search_handler, shared_context), ui (chat_interface, search_interface, styles), utils (data_loader, ssl_fix). Data flödar: DataLoader → merged DecisionRecords; SemanticSearchEngine bygger index från key_text (MiniLM, 500 tecken chunk, 100 overlap); LegalBERTPredictor använder sliding window 512/256 som vid träning; RAGSystem gör keyword-intent + template-svar.
- **Komponenter:** 46 beslut, 40 märkta, 3 187 chunks, 12 ms sökning, ~5 s prediktion per beslut (CPU).
- **Kända problem:** Modell 65 % genomsnittlig accuracy, stark MEDIUM_RISK-bias; 40 exempel är litet; keyword-intent spröd; ingen konversationsminne; ingen re-ranking; UI saknar loading skeleton, export, korrekt svenska tecken (t.ex. “atgarder”); ingen auth/rate limit/logging; 2,4 GB modeller, första start laddar MiniLM (~100 MB).
- **Rekommenderade åtgärder (ur dokumentet):** Ensemble alla 5 folds; binär HIGH vs NOT_HIGH; tröskelkalibrering; LLM för svar; svensk intent-klassificerare; cross-encoder re-ranking; progressiv laddning; export; fixa UTF-8; REST API + auth vid produktion.

### Åtgärdspunkter (kort)
| Prioritet | Åtgärd | Var |
|----------|--------|-----|
| P0 | Dokumentera och (där det går) ta bort SSL-workaround eller begränsa till dev | `utils/ssl_fix.py`, `05_finetune_legalbert.py` |
| P1 | Ensemble 5 folds istället för enbart fold_4 | `backend/risk_predictor.py`, `models/nap_legalbert_cv/` |
| P1 | Fixa svenska tecken i UI (UTF-8) | `ui/`, `integration/chat_handler.py` |
| P2 | README i repo root med körkommandon och beroenden | **Klart** – `README.md` |
| P2 | Progressiv laddning / loading skeleton vid start | `app.py`, `ui/` |

---

## 3) Fokus: COMPLETE_DATA_INVENTORY.md (sammanfattning + datapunkter att bevaka)

### Sammanfattning
- **Datakällor:** Domar (46 unika, 40 märkta, 6 exkluderade), Ansökningar (39 PDF, 26 text-extraherbara, 13 skannade), Lagstiftningdiverse (38 filer, 571K ord). Processed: cleaned_court_texts.json, labeled_dataset.json, linkage_table.json, label_overrides.json, label_review.txt.
- **Format:** Rå TXT i `Data/Domar/data/processed/court_decisions/`; PDF i Ansökningar och Lagstiftningdiverse; JSON i `Data/processed/`.
- **Var data ligger:** Rå: `Data/Domar/`, `Data/Ansökningar/`, `Data/Lagstiftningdiverse/`. Source of truth för appen: `Data/processed/cleaned_court_texts.json`, `labeled_dataset.json`, `linkage_table.json`.
- **PII/känslig data:** Inventariet nämner inte explicit PII; domar är offentliga. Dataägare/uppdateringsfrekvens inte formaliserad i repot.
- **Livscykel:** Rådata → scripts 02–04 → processed JSON → 05 träning → modeller i `models/`. Appen läser bara processed + modeller.
- **Viktiga fynd:** 4 domar felaktigt i Ansökningar (kan utökas till 44 märkta); 6 dubbletter + 2 icke-domar i court_decisions (städning); 571K ord Lagstiftning för DAPT; 13 skannade PDF kräver OCR.

### Datapunkter att bevaka
- **Duplicat:** Ta bort 6 duplicatfiler i court_decisions och flytta 2 icke-domfiler (enligt COMPLETE_DATA_INVENTORY.md).
- **Nya domar:** 4 domar i Ansökningar (M 483-22, M 2694-22, M 2695-22, M 2796-24) – extrahera, märk, lägg i labeled_dataset och överväg omträning.
- **DAPT:** Lagstiftningdiverse + ansökningar som förträningskorpus (~650K ord) – dokumentera och eventuellt automatisera.
- **OCR:** 13 skannade PDF i Ansökningar – behov av OCR-pipeline om de ska användas.
- **Genererade/deriverade:** cleaned_court_texts.json, labeled_dataset.json, linkage_table.json, label_overrides.json är deriverade; behåll versionering/backup vid stora ändringar.

---

## 4) Karta över mapparna scripts/, nap-legal-ai-advisor/, models/, data/

| Del | Ansvar | Viktiga filer | Hur körs |
|-----|--------|----------------|----------|
| **scripts/** | Data-pipeline, träning, utvärdering, inventering | Se tabell nedan | Alla från repo root: `python scripts/<script>.py` |
| **nap-legal-ai-advisor/** | Streamlit-app: chat + sök, RAG, riskprediktion | `app.py`, `backend/rag_system.py`, `backend/risk_predictor.py`, `backend/search_engine.py`, `utils/data_loader.py`, `integration/chat_handler.py`, `integration/search_handler.py` | `cd nap-legal-ai-advisor` sedan `streamlit run app.py` (cwd kan vara repo root eller nap-legal-ai-advisor beroende på sökvägar – data_loader använder parent.parent.parent så **kör från repo root**: `streamlit run nap-legal-ai-advisor/app.py`) |
| **models/** | Fine-tuned LegalBERT (5 folds); endast fold_4 används | `nap_legalbert_cv/fold_4/best_model/` (config, model.safetensors, tokenizer*), `training_metrics.json`, `performance_summary.md` | Ingen direkt körning; laddas av `backend/risk_predictor.py` |
| **Data/** | Rå och processad data | Rå: `Domar/data/processed/court_decisions/*.txt`, `Ansökningar/*.pdf`, `Lagstiftningdiverse/*`. Processed: `processed/cleaned_court_texts.json`, `labeled_dataset.json`, `linkage_table.json` | Läses av scripts 02–04 och av appen (endast processed). |

### Scripts (gruppering och kommandon)

| Script | Syfte | Input | Output | Kommando |
|--------|--------|------|--------|----------|
| **02_clean_court_texts.py** | Rensa OCR, segmentera sektioner, extrahera key_text, metadata | `Data/Domar/data/processed/court_decisions/*.txt` | `Data/processed/cleaned_court_texts.json` | `python scripts/02_clean_court_texts.py` |
| **03_create_labeled_dataset.py** | Skapa etiketter HIGH/MEDIUM/LOW, train/val/test 70/15/15 | `Data/processed/cleaned_court_texts.json`, `label_overrides.json` (valfritt) | `Data/processed/labeled_dataset.json`, `label_review.txt` | `python scripts/03_create_labeled_dataset.py` (ev. redigera label_overrides.json och kör igen) |
| **04_link_to_viss.py** | Länka vattenförekomster till VISS-ID | nap_model-main/data/rich_court_database.json, mcda_rankings_full.json, agent/nap_quantitative_data_v2.17.json, Data/processed/cleaned_court_texts.json | `Data/processed/linkage_table.json` | `python scripts/04_link_to_viss.py` (kräver nap_model-main) |
| **05_finetune_legalbert.py** | 5-fold CV fine-tuning KB-BERT | `Data/processed/labeled_dataset.json` | `models/nap_legalbert_cv/fold_*/best_model/`, `training_metrics.json`, `confusion_matrices.png` | `python scripts/05_finetune_legalbert.py` (GPU rekommenderas, ~5 h) |
| **06_evaluate_basic.py** | Sammanfatta metrics, confusion, rapport | `models/nap_legalbert_cv/training_metrics.json` | `evaluation_report.json`, `performance_summary.md`, `fold_comparison.png` | `python scripts/06_evaluate_basic.py` |
| **extract_all_pdfs.py** | Extrahera text från PDF i Ansökningar + Lagstiftningdiverse | `Data/` (hårdkodad path) | JSON i `Data/processed/` (ansokan_*, lagtiftning_*) | `python scripts/extract_all_pdfs.py` (ändra ROOT för portabilitet) |
| **inventory.py** | Filinventering per extension och mapp | `Data/` | Konsol + `COMPLETE_FILE_INVENTORY.json` | `python scripts/inventory.py` |

---

## 5) Komponentkarta (nap-legal-ai-advisor) – “X anropar Y som använder Z”

- **app.py** anropar **SharedContext.initialize()**, **init_backend()**, **render_sidebar()**, **render_chat_mode(chat_handler)**, **render_search_mode(search_handler)**.
- **init_backend()** anropar **load_data()** (utils.data_loader) → **get_search_engine()**, **get_predictor()** → **search.build_index(data.get_all_decisions())** → skapar **RAGSystem(data, search, predictor)**, **ChatHandler(rag_system)**, **SearchHandler(data, search, predictor)**.
- **load_data()** (DataLoader) läser **Data/processed/labeled_dataset.json**, **cleaned_court_texts.json**, **linkage_table.json** och bygger **DecisionRecord**-listor.
- **get_search_engine()** returnerar **SemanticSearchEngine** som använder **sentence_transformers** (paraphrase-multilingual-MiniLM-L12-v2), chunkar key_text (500/100), cachar embeddings i **.cache/embeddings.pkl**.
- **get_predictor()** returnerar **LegalBERTPredictor** som laddar **models/nap_legalbert_cv/fold_4/best_model** (transformers AutoModelForSequenceClassification, AutoTokenizer), sliding window 512/256, aggregerar logits.
- **ChatHandler.process_message()** anropar **rag.generate_response()**; **RAGSystem.generate_response()** gör keyword-intent och anropar t.ex. **_format_risk_response()**, **_format_search_response(query)** (som använder **search.search()**).
- **SearchHandler** använder **data**, **search**, **predictor** för sökning, filter och riskprediktion i sökflödet.
- **Externa modeller/tjänster:** Hugging Face (KB-BERT, MiniLM); ingen annan API. Konfiguration: inga env-variabler krävs för grundläggande körning; SSL-fix i kod.

---

## 6) Dataflöden och execution paths

### Flöde 1: Rådata → bearbetning → modell/knowledge base → användning i nap-legal-ai-advisor

1. **Rå TXT:** `Data/Domar/data/processed/court_decisions/*.txt` (46 unika efter skip av duplicat/icke-domar).
2. **Rensning:** `scripts/02_clean_court_texts.py` → `Data/processed/cleaned_court_texts.json` (sektioner, key_text, metadata).
3. **Etiketter + splits:** `scripts/03_create_labeled_dataset.py` (läser cleaned, ev. label_overrides) → `Data/processed/labeled_dataset.json`, `label_review.txt`.
4. **VISS-länk (valfritt):** `scripts/04_link_to_viss.py` (läser nap_model-main + cleaned) → `Data/processed/linkage_table.json`.
5. **Träning:** `scripts/05_finetune_legalbert.py` (läser labeled_dataset.json) → `models/nap_legalbert_cv/fold_*/best_model/`, `training_metrics.json`.
6. **Användning i appen:** Vid start läser **utils/data_loader.py** cleaned_court_texts.json, labeled_dataset.json, linkage_table.json → **DecisionRecord**. **backend/search_engine.py** bygger index från key_text (chunk + MiniLM). **backend/risk_predictor.py** laddar fold_4. Allt används av RAG och sök.

### Flöde 2: Användarfråga → retrieval/inference → svar + (saknad) logging

1. **Användarfråga:** Chat- eller sökfliken i Streamlit.
2. **Chat:** **integration/chat_handler.py** → **RAGSystem.generate_response(query)** → keyword-intent i **backend/rag_system.py** → antingen template-svar (t.ex. _format_risk_response) eller **_format_search_response(query)** som anropar **search.search(query, ...)**.
3. **Sök:** **integration/search_handler.py** använder **search.search()** (cosine similarity mot cached embeddings), ev. **predictor.predict(decision)** för risk.
4. **Svar:** Markdown tillbaka till UI. Ingen strukturerad logging, ingen auth, ingen rate limit.

---

## 7) Getting started (onboarding)

### Installation
```bash
cd c:\Users\SE2I7A\Desktop\legalmodel
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r nap-legal-ai-advisor/requirements.txt
```
För att köra **scripts/** (02–06) behövs samma miljö plus t.ex. `matplotlib`, `seaborn`, `scikit-learn` (finns i requirements.txt). För **05_finetune_legalbert.py**: GPU + CUDA rekommenderas.

### Miljövariabler
- Ingen .env krävs för nap-legal-ai-advisor. Vid corporate proxy används inbyggd SSL-workaround (ssl_fix); för produktion bör SSL inte stängas av utan proxy konfigureras korrekt.
- nap_model-main har `.env.example`; används av 04_link_to_viss om något API behövs (ej nämnt i 04-scriptet direkt).

### Minimalt exempel – starta appen
```bash
cd c:\Users\SE2I7A\Desktop\legalmodel
streamlit run nap-legal-ai-advisor/app.py
```
Öppna http://localhost:8501. Första gången: laddning av MiniLM + bygg av index (~45 s).

### Köra pipeline (från rådata till modell)
```bash
python scripts/02_clean_court_texts.py
python scripts/03_create_labeled_dataset.py
# Valfritt: python scripts/04_link_to_viss.py  (kräver nap_model-main data)
python scripts/05_finetune_legalbert.py
python scripts/06_evaluate_basic.py
```

### Tester / lint
- **nap-legal-ai-advisor:** Inga tester eller lint-konfiguration i repot.
- **nap_model-main:** Har `tests/` (98 tester, 97 % pass enligt README); kör t.ex. `pytest nap_model-main/tests` om pytest är installerat.
- **Saknat:** Root-level pytest/tox, pre-commit, CI, tydlig lint (ruff/flake8) för nap-legal-ai-advisor.

---

## 8) Risker och förbättringar (prioriterat)

### Största tekniska risker
- **En enda fold (fold_4)** – hög varians mellan folds (50–75 % accuracy); överanpassning till en fold.
- **40 exempel för 110M-parametrar** – liten träningsmängd; generalisering osäker.
- **Hårdkodade/hårda sökvägar** – data_loader antar repo root; extract_all_pdfs.py har absolut Windows-sökväg; script 04 antar nap_model-main på relativ sökväg.
- **Ingen CI/CD** – ingen automatisk test eller bygg; regression lätt att missa.

### Säkerhetsrisker (nycklar/PII)
- **SSL verifiering avstängd** (utils/ssl_fix.py, 05_finetune_legalbert.py) – man-in-the-middle i produktion.
- Ingen auth eller rate limiting på Streamlit-appen.
- PII inte explicit dokumenterad; domar är offentliga men hantering av eventuell personinfo i texter är inte beskriven.

### Data governance
- Ingen dokumenterad dataägare eller uppdateringsfrekvens.
- Duplicat och icke-domfiler i court_decisions (enligt COMPLETE_DATA_INVENTORY) – risk för dubbelräkning eller fel källor.

### Drift/observability
- Ingen strukturerad logging, spårning av frågor eller fel.
- Ingen health check eller metrics för modell-/index-laddning.

### Föreslagna förbättringar (P0–P2)

| Prio | Förbättring | Filer / plats |
|------|-------------|----------------|
| P0 | Ta bort eller begränsa SSL-workaround till dev; dokumentera proxy-krav | `nap-legal-ai-advisor/utils/ssl_fix.py`, `scripts/05_finetune_legalbert.py` |
| P0 | Gör extract_all_pdfs.py portabel (repo root eller env) | `scripts/extract_all_pdfs.py` (ersätt hårdkodad ROOT) |
| P1 | Ensemble alla 5 folds i risk_predictor | `backend/risk_predictor.py` |
| P1 | Root README med syfte, körkommandon, beroenden, pipeline | **Klart** – `README.md` i repo root |
| P1 | Fixa svenska tecken (UTF-8) i UI-strängar | `ui/*.py`, `integration/chat_handler.py` |
| P1 | Städa court_decisions: ta bort 6 duplicat, flytta 2 icke-domfiler | `Data/Domar/data/processed/court_decisions/` + dokumentera i README |
| P2 | Lägg till enhetstester för data_loader, search, predictor | t.ex. `nap-legal-ai-advisor/tests/` |
| P2 | Progressiv laddning / skeleton vid app-start | `app.py`, `ui/` |
| P2 | Enkel strukturerad logging (query, mode, fel) | `integration/*.py`, `app.py` |
| P2 | .env.example för nap-legal-ai-advisor (HF cache, ev. proxy) | `nap-legal-ai-advisor/.env.example` |

---

## 9) End

Rapporten bygger på genomgång av repo-struktur, `SYSTEM_REVIEW.md`, `COMPLETE_DATA_INVENTORY.md`, `nap-legal-ai-advisor/` (app.py, backend, integration, ui, utils), `scripts/` (02–06, extract_all_pdfs, inventory), `models/nap_legalbert_cv/` och `Data/`. För full reproducerbarhet: kör alla kommandon från **repo root** och säkerställ att `Data/processed/` innehåller cleaned_court_texts.json, labeled_dataset.json och linkage_table.json innan du startar appen eller träningsscript.
