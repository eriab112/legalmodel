# Phase A Plan Review: Enhanced NAP Data Preparation

**Reviewed:** 2026-02-12  
**Verdict:** **Viable and well-motivated**, with several **must-fix** implementation mismatches and a few design choices to clarify before coding.

---

## Summary

| Aspect | Assessment |
|--------|------------|
| **Strategic fit** | Strong: +4 decisions, Sundin-based criteria, DAPT, weak labels align with SYSTEM_REVIEW and COMPLETE_DATA_INVENTORY. |
| **Academic grounding** | Good: Sundin et al. 2026 as framework is appropriate; plan correctly separates applications (proposals) from decisions (verdicts). |
| **Implementation vs codebase** | **Needs fixes:** JSON structure, script reuse, and train/inference consistency do not match the current repo. |
| **Risks** | Manageable if: (1) labels are not over-fitted to regex, (2) weak labels are not used in val/test, (3) DAPT + fine-tune use the same input format as current inference. |

---

## 1. What Is Intelligent and Viable

- **Adding the 4 misfiled domar** is correct and already recommended in COMPLETE_DATA_INVENTORY; moving/copying and then re-running the existing pipeline is the right idea.
- **Sundin et al. 2026 as labeling framework** is a clear improvement over ad-hoc heuristics, provided the paper’s criteria are mapped to code (and thresholds are justified or tuned).
- **Feature extraction (A3)** gives interpretability and a path to multi-task or rule-augmented models; storing in `decision_features_sundin2026.json` is sensible.
- **Strict separation of applications vs court decisions** and **weak labels with low confidence and explicit “proposal, not verdict”** is methodologically sound and reduces risk of conflating proposal and outcome.
- **DAPT on 1.69M words** (legislation + applications + court texts) is a standard and reasonable way to get a domain-adapted encoder; COMPLETE_DATA_INVENTORY already suggested ~650K words, so expanding is consistent.
- **Weighted loss for strong vs weak** (e.g. weight 1.0 vs 0.2–0.4) is a standard approach in semi-supervised learning.
- **Manual verification step for the 4 new labels** is necessary and well placed.
- **3-day scope** is tight but plausible if most code is adapted from existing scripts rather than written from scratch.

---

## 2. Critical Fixes (Must Address)

### 2.1 Data structures do not match the plan

- **cleaned_court_texts.json**  
  - Actual: `{ "decisions": [ {...} ], "pipeline_version", "source_dir", ... }`.  
  - Each decision has **`text_full`**, not `cleaned_text`.  
  - Plan’s “extend existing_cleaned” and use of `cleaned_text` will break.  
  - **Fix:** Use key `text_full` everywhere. Do not manually extend the list; add the 4 new TXT files to `Data/Domar/data/processed/court_decisions/` and **re-run `02_clean_court_texts.py`** so it regenerates `cleaned_court_texts.json` with 50 decisions (and correct `stats`, `sections`, `key_text`).

- **labeled_dataset.json**  
  - Actual: no top-level **`decisions`** list. It has **`splits.train`**, **`splits.val`**, **`splits.test`** (each a list of items with `id`, `filename`, `label`, `key_text`, `metadata`, `scoring_details`), plus **`label_distribution`**, **`excluded_decisions`**, etc.  
  - Plan’s `labeled_dataset['decisions'].append(...)` is invalid.  
  - **Fix:** Do not edit `labeled_dataset.json` by hand. After the 4 new decisions are in `cleaned_court_texts.json`, add their **final** labels to **`label_overrides.json`** in the same format as today: `{ "decision_id": "HIGH_RISK" }` (IDs must match what `02` produces, e.g. `m483-22`). Then **re-run `03_create_labeled_dataset.py`** so it recomputes stratified splits for 44 decisions. That keeps splits and metadata consistent.

### 2.2 No importable “clean” API in 02

- Plan assumes: `from scripts.clean_court_texts import clean_decision_text, segment_sections`.  
  - The real script is **`02_clean_court_texts.py`** and does not expose `clean_decision_text` (it has `process_decision`, `segment_sections`, etc.).  
  - **Fix:** Either (a) add the 4 TXT files and run **`python scripts/02_clean_court_texts.py`** (recommended), or (b) refactor 02 so a small runner can call `process_decision` for the 4 files and merge the results into the existing `cleaned_court_texts.json` structure (more work, same outcome).

### 2.3 Train vs inference protocol (sliding window vs truncation)

- **Current setup:**  
  - **Training (05):** Uses **key_text**, **sliding window** (512 tokens, stride 256), chunk-level labels, then document-level aggregation.  
  - **Inference (risk_predictor):** Same sliding window over **key_text**; logits averaged over chunks, then argmax.  
- **Plan A7:** Uses **full text**, **truncation** to 512 tokens, one vector per sample.  
  - That would mean: **train** on truncated full text, **inference** on sliding window over key_text → **train/test mismatch** and likely worse performance or hard-to-interpret metrics.  
- **Fix (choose one):**  
  - **Option A (recommended):** Keep **sliding-window training** in A7 (reuse or adapt the chunking logic from `05_finetune_legalbert.py`). Strong samples = documents with chunk-level labels; weak samples = application texts (can stay as single truncated or single-chunk if short). Ensures inference remains valid without changing `risk_predictor.py`.  
  - **Option B:** Switch both training and inference to **truncation-only** (e.g. first 512 tokens of key_text). Simpler but loses long-document signal; then update `risk_predictor.py` to truncate instead of sliding window.

### 2.4 Weak labels in validation/test

- Plan builds one combined set and runs **StratifiedKFold** on `(texts, labels)` including weak labels. So validation/test folds contain application samples with **noisy** weak labels; metrics then reflect noise, not true generalisation on court decisions.  
- **Fix:** Stratify **only on the 44 strong (court) labels**. Use **only strong-label documents in val/test**. Add weak-label samples **only to the training set** (each fold). For example: split the 44 into 5 folds; for each fold, train on (44 − fold_size) strong + all 26 weak, validate on fold_size strong only.

### 2.5 A4: `total_cost` undefined

- In `weak_label_application`, the block `if total_cost < 5: low_indicators += 2` runs even when `cost_matches` is empty, so `total_cost` may be undefined → **NameError**.  
- **Fix:** Define `total_cost = 0` before the cost block, and set it inside `if cost_matches:` (e.g. `total_cost = sum(...)`).

### 2.6 DAPT (A6): Deprecated API

- **LineByLineTextDataset** is legacy in `transformers` and may be removed or broken.  
- **Fix:** Use **`datasets.load_dataset("text", data_files=corpus_txt_path, split="train")`** and a **DataCollatorForLanguageModeling** with the same tokenizer and `mlm_probability=0.15`, or a simple custom Dataset that reads line-by-line. Keep block_size=512 and MLM as in the plan.

### 2.7 A7: Trainer and dataset format

- Plan passes a **dict** of lists as `train_dataset` / `eval_dataset`. Hugging Face **Trainer** expects a **Dataset** (with `__getitem__` returning a dict of tensors).  
- **Fix:** Implement a **torch.utils.data.Dataset** (or `datasets.Dataset`) that returns `input_ids`, `attention_mask`, `labels`, and **`weights`**. In **WeightedTrainer.compute_loss**, pop `weights` from the batch and compute `(loss * weights).mean()` (or equivalent). Ensure the dataset is built from the combined strong+weak lists with the correct weight per sample.

---

## 3. Path and Naming Details

- **Folder name:** Plan uses `Data/Ansokningar` (no ö). On disk the folder may be **`Data/Ansökningar`**. Use the same name as in the repo (or a path from config) to avoid encoding/visibility issues on Windows.
- **File names of 4 domar:** E.g. `Vaxjo TR M 483-22 Dom 2026-01-22.pdf`. After extraction the TXT name will be `Vaxjo TR M 483-22 Dom 2026-01-22.txt`. Script 02 will derive `id` from metadata (e.g. `m483-22`) or from filename stem; use that same `id` in **label_overrides.json** when you add the 4 manual labels.

---

## 4. Sundin-Based Labeling (A2)

- **Threshold inconsistency:** Text says “need **4+** for HIGH”; code has `if score >= 5: return 'HIGH_RISK'`. Decide one rule (e.g. 4+ = HIGH) and implement it consistently.
- **Regex vs paper:** Sundin-feature-listan och variablerna har **korskontrollerats** mot kmae250140 (Sektion 2.1 och Tabell 2); se **LABELING_STRATEGY_VS_PAPER.md**. Inga justeringar behövdes. Fortsätt dokumentera trösklar (t.ex. i label_review eller metod) om ni låser nya regler.
- **Manual review:** Keeping a small JSON (e.g. `new_decisions_preliminary_labels.json`) for the 4 decisions and then copying final labels into **label_overrides.json** before re-running 03 is a good workflow.

---

## 5. DAPT and Fine-Tune (A6–A7)

- **Loading DAPT for classification:** Saving the **MaskedLM** model and then doing **AutoModelForSequenceClassification.from_pretrained(dapt_path, num_labels=3)** is correct: the BERT encoder loads, the classification head is new. No change needed.
- **Corpus format:** Ensure `dapt_corpus.json` and the script that writes the `.txt` for MLM use the same keys as the rest of the repo: e.g. **`text_full`** from cleaned decisions, and the structure expected by your legislation/application loaders (e.g. from COMPLETE_DATA_INVENTORY / existing processed files).
- **lagtiftning_texts.json / ansokan_texts.json:** Plan assumes keys like `documents`, `doc['text']`, `doc['category']`. Verify against the actual JSON written by `extract_all_pdfs.py` (and any legislation extraction script); adjust A5 so it matches real keys and nesting.

---

## 6. Suggested Order of Implementation

1. **A1:** Copy 4 PDFs → extract to TXT in `court_decisions` → run **02** → confirm 50 decisions in `cleaned_court_texts.json` and correct `text_full`/`key_text`.
2. **A2:** Implement Sundin-based indicator extraction and scoring; run on the 4 new decisions; **manual review**; write final labels into **label_overrides.json**; run **03** → confirm 44 decisions and new splits.
3. **A3:** Feature extraction from all 44 using **text_full** (and sections where needed); save `decision_features_sundin2026.json`; fix any key names to match 02 output.
4. **A4:** Weak labels for applications only; fix `total_cost`; save `weakly_labeled_applications.json`; keep a clear “application / weak” flag.
5. **A5:** Build DAPT corpus from legislation + applications + court **text_full**; verify keys against real files; write corpus and word count.
6. **A6:** DAPT with `datasets` (or custom Dataset) + DataCollatorForLanguageModeling; save `models/nap_dapt_bert/final`.
7. **A7:** Data loader that preserves **sliding window for strong court documents** and weighted weak samples; **val/test only on strong labels**; custom Dataset + WeightedTrainer; 5-fold CV; save best fold(s) and metrics.

---

## 7. Risk Summary

| Risk | Mitigation |
|------|------------|
| Overfitting to regex rules in A2 | Treat Sundin indicators as **features + suggested label**; keep manual override and document thresholds. |
| Weak labels in val/test | Exclude weak-label samples from validation and test; stratify only on 44 strong. |
| Train/inference mismatch | Keep sliding-window training for court decisions (or switch both train and inference to truncation and document it). |
| DAPT API breakage | Use `datasets` + DataCollatorForLanguageModeling instead of LineByLineTextDataset. |
| Wrong JSON keys | Use **text_full**, **decisions**, **splits** as in current codebase; verify legislation/application JSON structure in A5. |

---

## 8. Verdict

- **Intelligent:** Yes. The combination of +4 decisions, Sundin-based criteria, rich features, weak supervision, and DAPT is coherent and aligned with the repo and the inventory.
- **Viable:** Yes, **if** you: (1) align all data structures and keys with the current repo, (2) re-use 02/03 for the 4 new decisions and overrides, (3) keep sliding-window training for court documents (or explicitly switch both train and inference), (4) validate only on strong labels, (5) fix the A4 bug and the DAPT/dataset code as above.
- **Recommendation:** Implement in the order above; do A1–A2 first and confirm 44 labeled decisions and correct splits before investing in DAPT and weak-label training.
