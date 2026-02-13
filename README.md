# NAP Legal Model

**Stöd för rådgivning kring Nationella planen för moderna miljövillkor (NAP)** – vattenkraft, miljödomstolar och svensk miljörätt. Detta repo innehåller datapipeline, tränade modeller och **NAP Legal AI Advisor** (Streamlit-app) för semantisk sökning och riskindikation (HIGH/MEDIUM/LOW) i miljödomstolsbeslut.

---

## Vad finns i repot?

| Del | Beskrivning |
|-----|-------------|
| **nap-legal-ai-advisor/** | Streamlit-app: **Chat** (frågor/svar) och **Sök** (semantisk sökning över domar). Riskprediktion med LegalBERT (HIGH/MEDIUM/LOW). ~1 800 rader Python. |
| **scripts/** | Pipeline: rensning av domtexter → etiketter → träning → utvärdering. Plus Sundin-feature-extraktion, weak labels, DAPT-korpus, PDF-extraktion m.m. |
| **Data/** | Rådata (Domar, Ansökningar, Lagstiftningdiverse) och **Data/processed/** med JSON som appen och scripten använder. |
| **models/** | Fine-tunad KB-BERT (5-fold CV). Appen använder **fold_4**. |
| **nap_model-main/** | Separat NAP Decision Support-system (KPIs, VISS, kostnadsmodell, dashboard). Delad data används i `04_link_to_viss.py`; appen är oberoende. |

Appen läser **endast** från `Data/processed/` (cleaned_court_texts, labeled_dataset, linkage_table) och från `models/nap_legalbert_cv/`.

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
```

För PDF-extraktion (t.ex. Sundin-script eller `extract_pdf_text.py`):  
`pip install pymupdf`

### Starta appen

**Kör alltid från repo root** (appen använder relativa sökvägar mot root):

```bash
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
├── scripts/                     # Pipeline och hjälpscript
├── Data/
│   ├── Domar/data/processed/court_decisions/   # Rå TXT (domar)
│   ├── Ansökningar/                            # PDF (ansökningar + några domar)
│   ├── Lagstiftningdiverse/                    # Lagstiftning, riktlinjer
│   └── processed/                              # cleaned_court_texts.json, labeled_dataset.json, m.m.
├── models/nap_legalbert_cv/    # 5-fold LegalBERT (fold_4 används i appen)
├── nap_model-main/             # Separat NAP Decision Support (VISS, kostnadsmodell)
└── [dokumentation .md]         # SYSTEM_REVIEW, COMPLETE_DATA_INVENTORY, CONTEXT_HANDOVER, m.m.
```

---

## Data och pipeline

- **Rå domar:** TXT-filer i `Data/Domar/data/processed/court_decisions/`. Nya domar läggs här; därefter körs **02** så att `cleaned_court_texts.json` uppdateras.
- **Etiketter:** Nya/ändrade etiketter läggs i **`Data/processed/label_overrides.json`** (`{ "decision_id": "HIGH_RISK" }`). **Redigera inte** `labeled_dataset.json` manuellt – kör **03** så byggs train/val/test om.
- **Nyckelstrukturer:**
  - `cleaned_court_texts.json` → `decisions[]` med `id`, `text_full`, `key_text`, `sections`, `metadata`.
  - `labeled_dataset.json` → `splits.train`, `splits.val`, `splits.test` (inga toppnivå-`decisions`).

Nuvarande tillstånd: **46 beslut** i cleaned, **40 märkta** i labeled. **4 extra domar** finns som TXT (tillagda via `add_4_new_domar.py`) men 02/03 har inte körts för att nå 50/44.

---

## Script (urval)

| Script | Syfte |
|--------|--------|
| **02_clean_court_texts.py** | Rensa OCR, segmentera sektioner, bygg key_text → `cleaned_court_texts.json` |
| **03_create_labeled_dataset.py** | Bygg train/val/test från cleaned + `label_overrides.json` → `labeled_dataset.json` |
| **04_link_to_viss.py** | Länka vattenförekomster till VISS (kräver nap_model-main) |
| **05_finetune_legalbert.py** | 5-fold fine-tuning KB-BERT → `models/nap_legalbert_cv/` |
| **06_evaluate_basic.py** | Sammanfatta träningsresultat, confusion, rapporter |
| **sundin_feature_extraction.py** | Extrahera Sundin-features (passage, flöde, övervakning, kostnad) → `decision_features_sundin2026.json` |
| **sundin_validation.py** | RF feature importance (5-fold) + KMeans-klustring som diagnostik |
| **add_4_new_domar.py** | Hitta 4 dom-PDF i Ansökningar, kopiera och extrahera till TXT |
| **weak_labels_applications.py** | Weak labels för ansökningar → `weakly_labeled_applications.json` |
| **build_dapt_corpus.py** | Bygg DAPT-korpus (lagstiftning + ansökningar + domar) → `dapt_corpus.json` |
| **extract_pdf_text.py** | Extrahera text från en PDF till .txt (PyMuPDF). Exempel: `python scripts/extract_pdf_text.py kmae250140.pdf` |
| **extract_all_pdfs.py** | Extrahera text från PDF i Ansökningar + Lagstiftningdiverse → JSON i `Data/processed/` |

Alla script körs från **repo root**, t.ex. `python scripts/02_clean_court_texts.py`.

---

## Strategi och akademisk grund

- **Etiketter:** HIGH_RISK / MEDIUM_RISK / LOW_RISK, satta utifrån domslut, kostnader och åtgärder (inte enbart “tillstånd/avslag”).
- **Sundin et al. 2026** (kmae250140) används som **feature-taxonomi**: vad som ska extraheras (nedströms/uppströms passage, flöde, övervakning). **Vikter och trösklar** lärs från de märkta besluten (hybrid: “Sundin säger vad vi tittar på, datan säger hur vi viktar det”).
- **Phase A:** +4 domar, Sundin-features, weak labels för ansökningar (endast i träning), DAPT-korpus, omträning. Detaljer och implementationregler finns i dokumentationen nedan.

---

## Dokumentation (viktiga filer)

| Dokument | Innehåll |
|----------|----------|
| **CONTEXT_HANDOVER_FOR_STRATEGY.md** | Full kontexthandover: vad som gjorts, vad som inte ändrats, nästa steg, sökvägar, regler. **Bör läsas** tillsammans med övriga docs. |
| **SYSTEM_REVIEW.md** | Arkitektur för NAP Legal AI Advisor, kända begränsningar, förbättringsförslag (ensemble, binär klass, kalibrering, UI, säkerhet). |
| **COMPLETE_DATA_INVENTORY.md** | Datakällor (Domar, Ansökningar, Lagstiftningdiverse), fynd (4 domar i Ansökningar, duplicat, DAPT-möjligheter). |
| **STRATEGY_AND_PHASE_A.md** | Samlad strategi (hybrid: Sundin + datadriven viktning) och Phase A: kritiska fix, ordning, risker. Ersätter PHASE_A_PLAN_REVIEW och HYBRID_APPROACH_REVIEW. |
| **BESLUTSTODSBEDOMNING.md** | Bedömning: förutsättningar för bra beslutstöd, begränsningar idag, vad “bra” bör innefatta. |

---

## Kända begränsningar

- **Modell:** ~65 % genomsnittlig accuracy (5-fold); stark MEDIUM_RISK-bias. Endast **fold_4** används i appen (ingen ensemble).
- **Data:** 40 märkta beslut är litet för 110M-parametrar; generalisering osäker.
- **App:** Keyword-baserad intent i chatten; ingen LLM; ingen auth, rate limit eller strukturerad logging.
- **Säkerhet:** SSL-workaround i `utils/ssl_fix.py` och i 05 – inte lämpligt för produktion utan korrekt proxy/CA.

Mer detaljer: **SYSTEM_REVIEW.md**.

---

## Nästa steg (kort)

1. Ta med de 4 nya domarna: kör **02** → lägg etiketter i **label_overrides.json** → kör **03** (→ 44 märkta).
2. Kör **sundin_feature_extraction.py** och **sundin_validation.py** på 44 beslut.
3. DAPT (A6) med **dapt_corpus.json**; omträning (A7) med starka + weak, sliding window, val/test endast på 44.

Full ordning och implementationregler: **CONTEXT_HANDOVER_FOR_STRATEGY.md** och **STRATEGY_AND_PHASE_A.md**.

---

## Licens och referenser

- Domar och lagstiftning: enligt gällande offentlighetsprincip och källor.
- Modellbas: **KB/bert-base-swedish-cased** (Hugging Face).
- Akademisk referens för features och etiketteringsram: **Sundin et al. (2026)**. *Insights from a nation-wide environmental relicensing of hydropower facilities in Sweden: a review of court verdicts from a biological perspective.* Knowl. Manag. Aquat. Ecosyst. 2026, 427, 6. https://doi.org/10.1051/kmae/2025034
