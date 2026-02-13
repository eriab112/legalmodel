# Full kontexthandover – NAP Legal Model & strategi

**Skapad:** 2026-02-12  
**Syfte:** Ge full kontext till den Claude du diskuterat strategin med: vad som genomförts, vilka insikter som kommit fram, vad som **inte** ändrats, och hur man går vidare.  
**Läs detta** tillsammans med `SYSTEM_REVIEW.md`, `COMPLETE_DATA_INVENTORY.md` och `STRATEGY_AND_PHASE_A.md` (samlad strategi + Phase A) för full bild.

---

## 1. Repo och system i korthet

- **Produkt:** `nap-legal-ai-advisor/` – Streamlit-app (Chat + Sök) med LegalBERT riskprediktion (HIGH/MEDIUM/LOW) och semantisk sökning över svenska miljödomstolsbeslut (NAP/vattenkraft). ~1 800 rader Python.
- **Separat:** `nap_model-main/` – NAP Decision Support (KPIs, VISS, kostnadsmodell, dashboard). Delad data används i `04_link_to_viss.py`; appen är oberoende.
- **Data:** Rå TXT i `Data/Domar/data/processed/court_decisions/`; bearbetad i `Data/processed/`: **cleaned_court_texts.json**, **labeled_dataset.json**, **linkage_table.json**. Appen läser **endast** dessa tre + modeller.
- **Modell:** `models/nap_legalbert_cv/` – 5-fold fine-tuned KB-BERT. **Endast fold_4** används i `backend/risk_predictor.py`. Träning: sliding window 512 tokens, stride 256, på **key_text**; inference samma protokoll.
- **Pipeline (oförändrad logik):**  
  TXT → `02_clean_court_texts.py` → `03_create_labeled_dataset.py` (+ ev. `04_link_to_viss.py`) → `05_finetune_legalbert.py` → `06_evaluate_basic.py`.  
  Appen: `streamlit run nap-legal-ai-advisor/app.py` **från repo root**.
- **Viktiga datastrukturer:**
  - **cleaned_court_texts.json:** `{ "decisions": [ {...} ], "pipeline_version", "source_dir", ... }`. Varje beslut har **`text_full`** (inte `cleaned_text`), `key_text`, `sections`, `metadata`, `id`, `filename`.
  - **labeled_dataset.json:** **Ingen** toppnivå-array `decisions`. Innehåll: **`splits.train`**, **`splits.val`**, **`splits.test`** (listor med `id`, `filename`, `label`, `key_text`, `metadata`, `scoring_details`), plus `label_distribution`, `excluded_decisions`, m.m. Etiketter för nya beslut läggs **inte** in här manuellt – de läggs i **label_overrides.json** och 03 körs om.

---

## 2. Strategi som beslutats

- **Phase A (Enhanced NAP Data Prep):** +4 domar från Ansökningar, Sundin-baserad feature-extraktion, weak labels för ansökningar (tydligt separerade från domar), DAPT-korpus, omträning med utökad data.
- **Hybrid:** *"Sundin säger vad vi ska titta på, datan säger hur vi ska vikta det."*  
  Sundin et al. 2026 används som **feature-taxonomi** (vilka kategorier och fält som ska extraheras). Trösklar och vikter **lärs från de 44 märkta besluten** (t.ex. RF feature importance, inte hårdkodade regler). Klustring (KMeans på Sundin-features) används som **diagnostik** av etiketter, inte som automatisk “rätt” etikett.
- **Multi-task (framåt):** Auxiliary heads (downstream/upstream/flow/monitoring) ska få **mål från samma Sundin-feature-extraktion** (samma text → samma features som targets). Risk-head använder 44 starka etiketter; val/test **endast** på dessa, weak labels **endast** i träning.

---

## 3. Kritiska implementationregler (från granskningar)

- **Cleaned:** Använd **`text_full`** överallt; bygg **inte** cleaned manuellt. Lägg nya TXT-filer i court_decisions och **kör 02** för att få 50 beslut.
- **Labeled:** **Redigera aldrig** labeled_dataset.json direkt. Lägg nya etiketter i **label_overrides.json** (`{ "decision_id": "HIGH_RISK" }`) och **kör 03** så byggs splits för 44.
- **Träning vs inference:** Behåll **sliding window** (512/256) på key_text både i träning och i risk_predictor. Om A7/omträning bygger på truncation måste antingen träning **eller** inference ändras så att de matchar (rekommendation: behålla sliding window).
- **Weak labels:** Använd weak-label-ansökningar **bara i träning**. Stratifiera och validera **endast** på de 44 starka (court) etiketterna.
- **DAPT:** Använd **inte** `LineByLineTextDataset`; använd `datasets.load_dataset("text", data_files=...)` + `DataCollatorForLanguageModeling`.
- **A7 Trainer:** Träningsdata ska vara ett **Dataset**-objekt som returnerar `input_ids`, `attention_mask`, `labels`, **weights**; WeightedTrainer ska plocka weights ur batch och vikta loss.

---

## 4. Vad som genomförts (ändringar i systemet)

### 4.1 Nya script (alla under `scripts/`)

| Script | Syfte | Läser | Skriver |
|--------|--------|-------|--------|
| **sundin_feature_extraction.py** | Sundin-features för alla märkta beslut (värden, inte bara booleska) | cleaned_court_texts.json, labeled_dataset.json | **decision_features_sundin2026.json** |
| **sundin_validation.py** | Klustring (KMeans) + RF feature importance (5-fold) | decision_features_sundin2026.json | **sundin_feature_importance.json**, **clustering_validation_report.json** |
| **add_4_new_domar.py** | Hitta och kopiera 4 dom-PDF från Ansökningar, extrahera text till TXT | Data/Ansökningar (PDF), PyMuPDF | Data/Domar/data/*.pdf, court_decisions/*.txt |
| **weak_labels_applications.py** | Weak labels för **endast** Ansökningar (exkl. Dom) | ansokan_texts.json | **weakly_labeled_applications.json** |
| **build_dapt_corpus.py** | Bygg DAPT-korpus från lagstiftning + ansökningar + domar | lagtiftning_texts.json, ansokan_texts.json, cleaned_court_texts.json | **dapt_corpus.json** |
| **extract_pdf_text.py** | Extrahera text från PDF till .txt (PyMuPDF) | En PDF (sökväg som argument) | Samma sökväg med .txt (t.ex. kmae250140.txt) |

### 4.2 Körande som faktiskt utförts

- **sundin_feature_extraction.py:** 40 beslut → `Data/processed/decision_features_sundin2026.json`.
- **sundin_validation.py:** Klustring + RF körda. ARI låg (~-0,03); RF-importance sparad (t.ex. upstream_has_eel_ramp, monitoring_required bland de viktigaste). Rapporter skrivna till `sundin_feature_importance.json` och `clustering_validation_report.json`.
- **add_4_new_domar.py:** 4 PDF kopierade till `Data/Domar/data/`, 4 TXT skapade i `court_decisions/` (Växjö M 483-22, M 2796-24; Östersunds M 2694-22, M 2695-22). **02 har inte körts** – cleaned_court_texts innehåller fortfarande 46 beslut.
- **weak_labels_applications.py:** 12 ansökningar fick weak labels → `weakly_labeled_applications.json` (struktur: `{ "metadata": {...}, "applications": [...] }`). Planen nämnde 26; antalet kommer från vad som i ansokan_texts.json är Ansökan med tillräckligt med text.
- **build_dapt_corpus.py:** 96 dokument, **1 419 196 ord** (38 lagstiftning, 12 ansökningar, 46 domar) → `dapt_corpus.json` (`metadata` + `documents`).

### 4.3 Nya eller uppdaterade filer (ingen pipeline-fil skrivs över av dessa script)

- **Data/processed/**  
  - decision_features_sundin2026.json  
  - sundin_feature_importance.json  
  - clustering_validation_report.json  
  - weakly_labeled_applications.json  
  - dapt_corpus.json  
- **Data/Domar/data/**  
  - 4 nya PDF (kopior från Ansökningar).  
- **Data/Domar/data/processed/court_decisions/**  
  - 4 nya TXT (Växjö TR M 483-22, M 2796-24; Östersunds TR M 2694-22, M 2695-22).

---

## 5. Vad som INTE ändrats (nuvarande tillstånd)

- **cleaned_court_texts.json** – oförändrad; fortfarande **46** beslut (4 nya TXT finns men 02 har inte körts).
- **labeled_dataset.json** – oförändrad; fortfarande **40** märkta beslut, samma splits.
- **label_overrides.json** – oförändrad; inga etiketter för de 4 nya domarna tillagda.
- **nap-legal-ai-advisor/** – ingen kodändring; appen använder fortfarande 40/46 och fold_4.
- **models/nap_legalbert_cv/** – oförändrad; ingen omträning.
- **02, 03, 04, 05, 06** – oförändrade; redo att köras enligt nedan.

Så: **systemet är fortfarande “40 märkta, 46 beslut i cleaned”; de 4 nya domarna finns som TXT (och PDF) men är inte med i cleaned/labeled förrän 02 och 03 körs.**

---

## 6. Nästa steg (i ordning)

1. **Ta med de 4 nya domarna i pipelinen**  
   - Kör: `python scripts/02_clean_court_texts.py` → cleaned_court_texts.json får **50** beslut.  
   - Kontrollera vilka **id** 02 ger de 4 (t.ex. m483-22, m2694-22, m2695-22, m2796-24 eller liknande från metadata/filnamn).  
   - Lägg deras **slutgiltiga etiketter** i `Data/processed/label_overrides.json`, t.ex. `{ "m483-22": "MEDIUM_RISK", ... }`.  
   - Kör: `python scripts/03_create_labeled_dataset.py` → labeled_dataset med **44** beslut och nya splits.

2. **Uppdatera Sundin-features och validering**  
   - Kör: `python scripts/sundin_feature_extraction.py` → 44 rader i decision_features_sundin2026.json.  
   - Kör: `python scripts/sundin_validation.py` → uppdaterad importance och klustringsrapport.

3. **DAPT (A6)**  
   - Använd `dapt_corpus.json` (eller export till en .txt per rad). Använd **datasets** + DataCollatorForLanguageModeling (inte LineByLineTextDataset). Spara t.ex. `models/nap_dapt_bert/final`.

4. **Omträning med starka + weak (A7)**  
   - Läs starka från labeled_dataset **splits** (alla 44), weak från `weakly_labeled_applications.json` → **applications**.  
   - Använd **sliding window** för court-dokument; weak kan vara en chunk/trunkering per ansökan.  
   - Val/test **endast** på 44; weak **endast** i train.  
   - Implementera Dataset som returnerar weights och WeightedTrainer enligt STRATEGY_AND_PHASE_A (avsnitt 3.7).

5. **Multi-task (Phase B / hybrid)**  
   - Auxiliary-mål från Sundin-feature-extraktion (samma text → samma features som labels).  
   - Hantera saknade värden (t.ex. “unknown”-klass per head).  
   - Behåll sliding window för risk; aux-loss på dokument- eller chunknivå med små λ.

---

## 7. Viktiga sökvägar och strukturer

- **Repo root:** bas för alla script (kör från här).  
- **App:** `streamlit run nap-legal-ai-advisor/app.py` (från root).  
- **Cleaned:** `Data/processed/cleaned_court_texts.json` → `decisions[]`, varje med `id`, `text_full`, `key_text`, `sections`, `metadata`, …  
- **Labeled:** `Data/processed/labeled_dataset.json` → `splits.train`, `splits.val`, `splits.test`; **ingen** `decisions` på toppnivå.  
- **Overrides:** `Data/processed/label_overrides.json` → `{ "decision_id": "LABEL" }`.  
- **Weak:** `Data/processed/weakly_labeled_applications.json` → `applications[]` med `weak_label`, `confidence`, `weight`, `text`, …  
- **DAPT:** `Data/processed/dapt_corpus.json` → `metadata`, `documents[]` med `text`, `source`, `filename`/`id`/`category`.  
- **Sundin-features:** `Data/processed/decision_features_sundin2026.json` → `decisions[]` med `id`, `label`, `features` (flat dict med t.ex. downstream_gap_mm, cost_msek, …).

---

## 8. Insikter och risker (kort)

### 8.1 Etiketteringsstrategi vs pappret (Sundin et al. 2026)

- **Pappret (kmae250140):** 33 NAP-domstolsfall till slutet av 2024 – 22 permit withdrawal/dam removal, 11 fortsatt drift under villkor. Dataextraktion (Sektion 2.1 / Tabell 2): upstream/downstream passage, guidance (vinkel, gap width), bypass, eel ramps, fishway-typ/slope/flöde, hydropeaking, e-flow, monitoring.
- **Vår strategi:** Tre klasser (HIGH/MEDIUM/LOW) utifrån domslut + kostnad/åtgärder/tidslinje. Sundin används som **feature-taxonomi**; trösklar och vikter lärs från 40/44 beslut.
- **Korskontroll:** Våra Sundin-features (`sundin_feature_extraction.py`) täcker papprets variabler (downstream/upstream/flow/monitoring). Vi har dessutom **cost_msek** och **timeline_years**. Slutsats: strategin är **vettig och förenlig** med pappret; ingen justering behövs.
- **Referens:** Sundin et al. (2026). *Insights from a nation-wide environmental relicensing of hydropower facilities in Sweden...* Knowl. Manag. Aquat. Ecosyst. 2026, 427, 6. https://doi.org/10.1051/kmae/2025034
- **Läsa PDF igen:** `python scripts/extract_pdf_text.py [sökväg/fil.pdf]` → samma sökväg med .txt (kräver PyMuPDF). Full variabeltabell och detaljer: **_archive/LABELING_STRATEGY_VS_PAPER.md**.

- **n=44:** RF/clustring är instabila; använd 5-fold RF och tolka importance som “vad som predicerar risk i denna mängd”, inte universell sanning. Klustring som **granskningsstöd**, inte auto-relabeling.
- **SSL:** `utils/ssl_fix.py` och liknande i 05 stänger av certifikatverifiering; oacceptabelt i produktion.  
- **Ensemble:** SYSTEM_REVIEW rekommenderar att använda alla 5 folds (inte bara fold_4) för mindre varians.  
- **Mappnamn:** På disk heter mappen **Ansökningar** (med ö); script ska använda samma namn (Path från BASE_DIR).

---

## 9. Dokument att använda tillsammans med denna fil

- **SYSTEM_REVIEW.md** – Nuvarande arkitektur, kända problem, förbättringsförslag.  
- **COMPLETE_DATA_INVENTORY.md** – Datakällor, 4 domar i Ansökningar, duplicat/städning, DAPT-möjligheter.  
- **STRATEGY_AND_PHASE_A.md** – Samlad strategi (hybrid: Sundin + datadriven viktning) och Phase A (kritiska fix, ordning, risker). Ersätter tidigare PHASE_A_PLAN_REVIEW och HYBRID_APPROACH_REVIEW.

(Sammanfattning etiketteringsstrategi vs pappret i avsnitt 8.1; full version med variabeltabell i **_archive/LABELING_STRATEGY_VS_PAPER.md**.)

Med denna handover har din andra Claude full kontext för att fortsätta: vad som redan är gjort, vad som är orört, vilka regler som gäller för implementation, och i vilken ordning nästa steg bör tas.
