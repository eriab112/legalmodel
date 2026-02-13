# Strategi och Phase A – samlad granskning

**Datum:** 2026-02-12  
**Innehåll:** Sammanslagning av *Phase A Plan Review* och *Hybrid Approach Review* – strategi (Sundin + datadriven viktning) och implementation (kritiska fix, ordning, risker) i ett dokument för enklare handover.

**Sundin vs pappret:** Feature-taxonomin är korskontrollerad mot kmae250140 (Sundin et al. 2026). Sammanfattning i **CONTEXT_HANDOVER_FOR_STRATEGY.md** (avsnitt 8.1); full variabeltabell i **_archive/LABELING_STRATEGY_VS_PAPER.md**.

---

## 1. Sammanfattning och verdict

| Aspekt | Bedömning |
|--------|------------|
| **Phase A – genomförbarhet** | **Ja**, om datastrukturer, 02/03, sliding window, weak endast i train, DAPT-API och A7 Dataset/weights följs enligt "Kritiska fix" nedan. |
| **Hybrid – strategi** | **Lämplig:** Sundin = vad vi tittar på, datan = hur vi viktar. Feature-importance och trösklar lärs från 44 beslut; klustring som diagnostik, inte auto-relabeling. |
| **Risker** | Hanterbara med: (1) etiketter inte överanpassade till regex, (2) weak labels endast i träning, (3) DAPT + fine-tune samma input-format som nuvarande inference (sliding window). |

**Rekommendation:** Implementera i ordningen nedan. A1–A2 först → 44 märkta beslut och korrekta splits; därefter DAPT och weak-label-träning.

---

## 2. Hybrid: princip och justeringar

### 2.1 Princip

*"Sundin säger vad vi ska titta på, datan säger hur vi ska vikta det."*  
Sundin et al. 2026 används som **feature-taxonomi** (vilka kategorier att extrahera). Trösklar och vikter **lärs från de 44 märkta besluten** (t.ex. RF feature importance). Klustring (KMeans på Sundin-features) används som **diagnostik** av etiketter, inte som automatisk "rätt" etikett.

### 2.2 Vad som fungerar bra

| Idé | Varför det hjälper |
|-----|---------------------|
| Sundin som taxonomi, inte regelbok | Undviker överanpassning till papprets trösklar; era data kan visa t.ex. att gap 15 vs 18 mm inte separerar risk. |
| Extrahera värden, inte bara booleska | `gap_width_mm`, `cost_msek`, `timeline_years` ger modellen och RF något att lära; raw values bättre än hårdkodade flaggor. |
| Feature importance från data | Att lära vilka Sundin-kategorier som predicerar HIGH/MEDIUM/LOW är rätt användning av 44 etiketter. |
| Klustring som sanity-check | KMeans på Sundin-features kan avslöja outliers och möjliga mislabels; bra som **granskningsstöd**, inte som ground truth. |
| Multi-task med aux från Sundin | Auxiliary heads (downstream, upstream, flow, monitoring) som regulariserare; mål från **samma** Sundin-extraktion (samma text → samma features som targets). |
| Phase B: RF(Sundin) + BERT-ensemble | Tolkbarhet (RF) + textnuans (BERT); utvärdering på samma 5-fold. |

### 2.3 Justeringar för n = 44

**RF feature importance:**  
Med 44 prover och många features överanpassar en enda RF; importance blir instabil.  
- **Lösning:** Aggregera över **5-fold CV** (train RF per fold, medel ± std av importance). Regularisera (t.ex. `max_depth=4`, `min_samples_leaf=3`). Tolka som "vilka Sundin-kategorier korrelerar med risk i dessa 44 beslut", inte universell sanning.

**Klustring (KMeans, 3 kluster):**  
- Använd som **diagnostik:** låg ARI → etiketter följer inte naturliga kluster; granska avvikande beslut.  
- **Caveats:** Kluster 0,1,2 mappas inte automatiskt till HIGH/MEDIUM/LOW – dokumentera mappning (t.ex. via medelkostnad eller majoritetsetikett i klustret). Vid ARI < ~0.3: lista beslut där kluster och etikett skiljer sig, gör **manuell genomgång**; **relabela inte** automatiskt från kluster.

### 2.4 Multi-task: var kommer auxiliary labels ifrån?

Risk-head har 44 etiketter. Auxiliary heads (downstream, upstream, flow, monitoring) har **inga separata mänskliga etiketter** – de ska få **mål från samma text** via **Sundin-feature-extraktion**:

- **Downstream:** t.ex. 4 klasser: no screen / inclined / angled / undefined.  
- **Upstream:** t.ex. no fishway / nature-like / vertical-slot / eel-ramp.  
- **Flow:** binär från `flow_hydropeaking_ban`.  
- **Monitoring:** binär från `monitoring_required` (ev. `monitoring_functional`).

**Implementation:** Samma text → extrahera Sundin-features → använd som auxiliary targets. Saknade värden: tydlig policy (t.ex. "no screen", "unknown"). Dokumentnivå aux-labels; antingen samma aux-label på alla chunks per dokument eller bara aux-loss på en representativ chunk. Loss: `total_loss = risk_loss + λ₁·downstream + λ₂·upstream + …` med små λ (t.ex. 0.2) så risk dominerar.

**Konsekvens för Phase A:** Sliding window för risk oförändrat; aux-loss på dokument/chunk-nivå. Ingen konflikt med Phase A-fixen nedan.

---

## 3. Phase A: kritiska fix (måste följas)

### 3.1 Datastrukturer

- **cleaned_court_texts.json**  
  - Faktisk struktur: `{ "decisions": [ {...} ], "pipeline_version", "source_dir", ... }`. Varje beslut har **`text_full`**, inte `cleaned_text`.  
  - **Fix:** Använd **`text_full`** överallt. Lägg inte till beslut manuellt; lägg nya TXT-filer i `Data/Domar/data/processed/court_decisions/` och **kör `02_clean_court_texts.py`** → 50 beslut i cleaned.

- **labeled_dataset.json**  
  - Faktisk struktur: **Ingen** toppnivå-array `decisions`. Innehåll: **`splits.train`**, **`splits.val`**, **`splits.test`** (listor med `id`, `filename`, `label`, `key_text`, `metadata`, `scoring_details`), plus `label_distribution`, `excluded_decisions` m.m.  
  - **Fix:** **Redigera aldrig** labeled_dataset.json manuellt. Lägg etiketter för nya beslut i **`label_overrides.json`** som `{ "decision_id": "HIGH_RISK" }` (id som 02 ger, t.ex. `m483-22`). **Kör `03_create_labeled_dataset.py`** → 44 beslut och nya splits.

### 3.2 Script 02

- Planen kan anta en importbar "clean"-API; verkliga scriptet är **`02_clean_court_texts.py`** och exponerar inte `clean_decision_text` som planen kanske beskriver.  
- **Fix:** Lägg 4 TXT i court_decisions och kör **`python scripts/02_clean_court_texts.py`**. Ingen refaktor krävs om 02 körs som helhet.

### 3.3 Träning vs inference (sliding window)

- **Nuvarande:** Träning (05) och inference (risk_predictor) använder **key_text** och **sliding window** (512 tokens, stride 256).  
- **Plan A7:** Om planen beskriver **full text + truncation** till 512, blir det train/inference-mismatch.  
- **Fix:** Behåll **sliding-window-träning** för domar i A7 (återanvänd/adapt chunking från 05). Weak samples (ansökningar) kan vara en chunk/trunkering per dokument. Då stämmer inference utan ändring i risk_predictor.

### 3.4 Weak labels i val/test

- **Fix:** Stratifiera **endast på de 44 starka (court) etiketterna**. Val/test **endast** på starka. Weak-label-ansökningar **endast i träningssetet** (per fold).

### 3.5 A4: total_cost odefinierat

- I weak-label-script: `if total_cost < 5` kan köras när `cost_matches` är tom → **NameError**.  
- **Fix:** Sätt `total_cost = 0` före cost-blocket och sätt den inuti `if cost_matches:`.

### 3.6 DAPT (A6): föråldrad API

- **LineByLineTextDataset** är legacy.  
- **Fix:** Använd **`datasets.load_dataset("text", data_files=...)`** + **DataCollatorForLanguageModeling** (samma tokenizer, `mlm_probability=0.15`). block_size=512, MLM som i planen.

### 3.7 A7: Trainer och dataset

- Trainer förväntar **Dataset**-objekt med `__getitem__` som returnerar dict med tensors, inte en dict med listor.  
- **Fix:** Dataset som returnerar `input_ids`, `attention_mask`, `labels`, **`weights`**. WeightedTrainer: plocka `weights` ur batch och beräkna `(loss * weights).mean()`.

### 3.8 Sökvägar och namn

- Mapp på disk: **`Data/Ansökningar`** (med ö). Använd samma namn (Path från repo root) för kompatibilitet (t.ex. Windows).  
- ID för de 4 domarna: det som 02 ger (metadata/filnamn), t.ex. `m483-22` – använd **samma id** i **label_overrides.json**.

### 3.9 Sundin-baserad etikettering (A2)

- Sundin-features och variabler är **korskontrollerade** mot kmae250140; sammanfattning CONTEXT_HANDOVER 8.1, full version _archive/LABELING_STRATEGY_VS_PAPER.md.  
- Dokumentera trösklar (t.ex. i label_review eller metod) om ni låser nya regler. Manuell granskning för de 4 nya besluten; skriv slutgiltiga etiketter i label_overrides och kör 03.

### 3.10 DAPT och fine-tune (A6–A7) – övrigt

- **Ladda DAPT för klassificering:** Spara MaskedLM; sedan `AutoModelForSequenceClassification.from_pretrained(dapt_path, num_labels=3)` – encoder laddas, klassificeringshead är ny. OK.  
- **Korpusformat:** Använd **text_full** från cleaned; verifiera nycklar mot befintliga lagstiftning/ansökan-loaders (t.ex. från COMPLETE_DATA_INVENTORY / dapt_corpus.json-struktur).

---

## 4. Implementeringsordning

1. **A1:** Kopiera 4 PDF → extrahera till TXT i court_decisions → kör **02** → bekräfta 50 beslut i cleaned och korrekt `text_full`/`key_text`.  
2. **A2:** Sundin-baserad feature-extraktion på de 4 nya; **manuell granskning**; skriv slutgiltiga etiketter i **label_overrides.json**; kör **03** → 44 beslut och nya splits.  
3. **A3:** Feature-extraktion för alla 44 med **text_full**; spara `decision_features_sundin2026.json`; justera ev. nyckelnamn till 02-output.  
4. **A4:** Weak labels endast för ansökningar; fixa `total_cost`; spara `weakly_labeled_applications.json`; tydlig "application / weak"-flagga.  
5. **A5:** Bygg DAPT-korpus (lagstiftning + ansökningar + domar **text_full**); verifiera nycklar mot befintliga filer.  
6. **A6:** DAPT med `datasets` + DataCollatorForLanguageModeling; spara t.ex. `models/nap_dapt_bert/final`.  
7. **A7:** Data loader med **sliding window för starka domar** och viktade weak samples; **val/test endast på 44**; Dataset med weights + WeightedTrainer; 5-fold CV; spara bästa fold(s) och metrics.

Efter A2/A3: kör **sundin_feature_extraction.py** och **sundin_validation.py** (RF importance, klustring som diagnostik). Phase B: RF(Sundin) + BERT-ensemble, 5-fold-utvärdering.

---

## 5. Risköversikt

| Risk | Åtgärd |
|------|--------|
| Överanpassning till regex (A2) | Sundin som **features + föreslagen etikett**; behåll manuell override och dokumentera trösklar. |
| Weak labels i val/test | Exkludera weak från val/test; stratifiera endast på 44 starka. |
| Train/inference-mismatch | Behåll sliding-window-träning för domar (eller byt både träning och inference till truncation och dokumentera). |
| DAPT-API | Använd `datasets` + DataCollatorForLanguageModeling, inte LineByLineTextDataset. |
| Fel JSON-nycklar | Använd **text_full**, **decisions**, **splits** enligt nuvarande kodbas; verifiera lagstiftning/ansökan-struktur i A5. |

---

## 6. Slutsats

- **Phase A** är genomförbar och vettig (+4 beslut, Sundin, DAPT, weak labels) **om** alla kritiska fix ovan följs (datastrukturer, 02/03, sliding window, weak endast i train, DAPT-API, A7 Dataset/weights).  
- **Hybriden** (“Sundin = vad, datan = hur”) är lämplig och bygger ovanpå samma pipeline; RF och klustring för stabil importance och diagnostik; multi-task med aux från Sundin-extraktion.  
- **En dokumentfil:** Denna fil ersätter *PHASE_A_PLAN_REVIEW.md* och *HYBRID_APPROACH_REVIEW.md* för handover; full kontext för vad som redan gjorts och exakta nästa steg finns i **CONTEXT_HANDOVER_FOR_STRATEGY.md**.
