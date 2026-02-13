# LegalBERT Data Exploration Report

**Date:** 2026-02-06
**Project:** NAP LegalBERT - Legal Risk Prediction Module
**Author:** Claude Opus 4.6 (automated exploration)

---

## Executive Summary

We have **significantly more data than initially expected**. Instead of ~28 decisions with metadata only, we have **46 unique court decision full texts** (952K words total) already extracted and ready as `.txt` files, plus **38 application PDFs** not yet extracted. The RCRS baseline covers all 15,688 water bodies but has a **heavily skewed distribution** (75% of scores clustered in 60-65 range), which motivates the need for a more discriminating ML-based approach.

---

## 1. Court Decision Inventory

### Text Files (Ready for Training)
| Metric | Value |
|--------|-------|
| **Total text files** | 54 |
| **Unique decisions (deduplicated)** | **46** |
| **Duplicates/versions removed** | 8 (3 versioned, 3 content-identical, 2 renamed) |
| **Non-decision files** | 2 (SvK report, VM guidelines) |
| **Total words** | 952,899 |
| **Average words/decision** | 17,646 |
| **Median words** | 12,930 |
| **Min words** | 631 (procedural beslut) |
| **Max words** | 91,717 (MÖD appeal M-693-22) |
| **Total size** | 6.6 MB |
| **Format** | Plain text (.txt), already extracted from PDF |
| **Language** | Swedish (confirmed) |
| **OCR needed** | NO - text already extracted |

### Application PDFs (Not Yet Extracted)
| Metric | Value |
|--------|-------|
| **Total PDF files** | 38 |
| **Total size** | 79.5 MB |
| **Format** | PDF (need text extraction) |

### Date Range
| Year | Decisions | Notes |
|------|-----------|-------|
| 2021 | 2 | Earliest decisions |
| 2022 | 5 | |
| 2023 | 8 | |
| 2024 | 16 | Most active year |
| 2025 | 14 | Most recent |
| **Total** | **45** | (1 file missing date in filename) |

### Decision Categories
| Category | Count | Description |
|----------|-------|-------------|
| `m-` (MMD decisions) | 22 | Mark- och miljödomstolen first instance |
| `NAP_` (NAP-specific) | 9 | Explicitly NAP-related cases |
| `Vaxjo_` (Växjö TR) | 8 | Växjö tingsrätt decisions |
| `MOD_` (MÖD appeals) | 5 | Mark- och miljööverdomstolen (appeal court) |
| `dom` (generic) | 1 | Generic naming (dom3.txt) |
| `om-` (procedural) | 2 | Procedural decisions |
| `f-` / `svea-` / `Umea` | 3 | Other courts |

### Court Distribution (Ansökningar/Applications)
| Court | Applications |
|-------|-------------|
| Vänersborgs TR | 17 |
| Östersunds TR | 8 |
| Växjö TR | 6 |
| Umeå TR | 4 |
| Nacka TR | 3 |

---

## 2. Data Quality Assessment

| Check | Status | Notes |
|-------|--------|-------|
| Complete text files | 46 unique | All readable, UTF-8 encoded |
| Missing/corrupted | 0 | All files open correctly |
| Language verified | Swedish | Confirmed in sample reads |
| OCR needed | NO | Text already extracted from PDFs |
| Structural consistency | PARTIAL | OCR artifacts present (page headers, footer addresses) |
| Character encoding | OK | Swedish chars (å, ä, ö) preserved |
| Section markers | VARIABLE | Some have clear sections, others free-flowing |

### Quality Issues Identified
1. **OCR artifacts**: Page headers ("Sid X (Y)", "SVEA HOVRÄTT DOM M XXXX-XX") repeated on every page
2. **Addresses/boilerplate**: Court contact details embedded in text (need removal)
3. **Line break artifacts**: Some entity names split across lines (e.g., "Marbäcks\nkraftverk")
4. **Inconsistent naming**: Same case appears with different filename conventions (e.g., `MOD_M10258-23` vs `MOD_M-10258-23`)
5. **dom3.txt**: One file with generic name, no identifiable case number in filename

---

## 3. Court Decision Structure Analysis

### Sample Decision 1: `m-1275-22-dom-2023-04-26.txt` (MÖD appeal)
- **Court**: Svea Hovrätt, Mark- och miljööverdomstolen
- **Case**: Prövning av latent villkor (Sunnerstaholms kraftverk, Voxnan)
- **Parties**: Kammarkollegiet vs Fortum Sverige AB, Bollnäs kommun, Länsstyrelsen
- **Structure**: Header → Parter → Saken → Domslut → Yrkanden → Utveckling av talan → Domskäl
- **Costs mentioned**: 40,800 kr + 113,906 kr (rättegångskostnader)
- **Word count**: ~12,000+

### Sample Decision 2: `NAP_M3426-24_2025-05-23_Karsbols.txt` (NAP case)
- **Court**: Svea Hovrätt, Mark- och miljööverdomstolen
- **Case**: Föreläggande att ansöka om tillstånd för vattenverksamhet (Karsbol Kraftverk)
- **Parties**: Karsbol Kraftverk AB vs Länsstyrelsen i Värmlands län
- **Structure**: Header → Parter → Saken → Domslut → Yrkanden → Utveckling av talan
- **NAP reference**: Explicitly mentions nationella planen för omprövning av vattenkraften
- **Historical references**: Häradsdom from 1854, Västerbygdens vattendomstol 1926/1936

### Common Legal Patterns Identified
| Pattern | Description | Frequency |
|---------|-------------|-----------|
| **NAP references** | Nationella planen for omprövning | Common in NAP_ prefixed files |
| **Environmental measures** | fiskvandring, omlöp, minimitappning, utskov, kontrollprogram, biotopvård, fiskväg | 7 distinct measure types |
| **Cost sections** | Rättegångskostnader, kostnader i SEK/MSEK | Present in most decisions |
| **Water body references** | Named rivers (vattendrag) and kraftverk | 18 water bodies indexed |
| **Legal citations** | Miljöbalken, vattendirektivet, EU references | Common in reasoning sections |
| **Verdict structure** | Domslut section clearly delineated | Consistent across decisions |

---

## 4. RCRS Baseline Statistics

| Metric | Value |
|--------|-------|
| **Total water bodies scored** | 15,688 |
| **Mean RCRS** | 60.31 |
| **Median RCRS** | 64.12 |
| **Std Dev** | 7.36 |
| **Min** | 42.59 |
| **Max** | 65.49 |
| **Range** | 22.90 |

### Distribution (CRITICAL FINDING)
```
Score Range  Count   Visualization
40-45:       1,210   ############
45-50:       1,214   ############
50-55:       1,351   #############
55-60:         158   #
60-65:       7,316   #########################################################################
65-70:       4,439   ############################################
```

**Key Insight**: The RCRS proxy distribution is **heavily right-skewed**. 75% of water bodies (11,755) have scores between 60-65.49, creating very poor discrimination. This is exactly why a ML-based approach could add significant value - the current proxy treats most water bodies as near-identical risk.

---

## 5. Water Body Linkages

| Metric | Value |
|--------|-------|
| **Water bodies in court decisions** | 18 named water bodies |
| **Total decision-WB entries** | 23 |
| **Unique decision files referenced** | 18 (out of 46 decisions) |
| **VISS IDs found** | 5 (very low!) |
| **Unique kraftverk** | 21 |
| **Unique kommuner** | 12 |

### NAP Measures Referenced in Decisions
| Measure | Swedish Term |
|---------|-------------|
| Fish passage | fiskvandring |
| Fish way | fiskväg |
| Bypass channel | omlöp |
| Minimum flow | minimitappning |
| Spillway | utskov |
| Monitoring program | kontrollprogram |
| Habitat restoration | biotopvård |

### VISS ID Coverage Gap
Only **5 VISS IDs** are linked to court decisions, vs 15,688 water bodies in the MCDA. This is a critical linkage gap that limits direct comparison between LegalBERT predictions and RCRS proxy scores for specific water bodies. **Strategy needed**: Map decisions to water bodies via kraftverk names, kommun, and water body names rather than relying on VISS IDs alone.

---

## 6. NER (Named Entity Recognition) Data

From `ner_all_decisions.json`:
| Entity Type | Count | Examples |
|-------------|-------|---------|
| KRAFTVERK | 79 | Sunnerstaholms, Karsbol, Marbäcks, Strömsbro |
| VATTENDRAG | 138 | Fylleån, Testeboån, Voxnan, Emån, Dalälven |
| KOMMUN | 88 | Halmstads, Bollnäs, Söderköpings, Sunne |
| KOSTNAD | 155 | Various amounts in SEK/MSEK |
| ÅTGÄRD | 146 | fiskvandring, omlöp, minimitappning etc. |
| **Total entities** | **606** | Across 28 analyzed decisions |

---

## 7. Available Pre-trained Models

### KB-BERT (swedish-bert-models/ repo)
- **Model**: `KB/bert-base-swedish-cased` (v1.1) from KBLab/National Library of Sweden
- **Training data**: ~15-20GB Swedish text (books, news, government publications, Wikipedia, forums)
- **Vocabulary**: ~50K tokens, cased, whole word masking
- **Variants available**:
  - `KB/bert-base-swedish-cased` - Base model (recommended for fine-tuning)
  - `KB/bert-base-swedish-cased-ner` - NER fine-tuned (useful for entity extraction)
  - `KB/albert-base-swedish-cased-alpha` - Smaller ALBERT variant
  - `KB/electra-small-swedish-cased` - Small Electra models
- **Status**: Repository cloned but **no model weights downloaded** - will download from HuggingFace during training
- **Dependencies**: transformers>=2.4.1, torch>=1.3.1

---

## 8. Additional Data Sources Available

| Source | Location | Size | Relevance |
|--------|----------|------|-----------|
| rich_court_database.json | nap_model-main/data/ | 43 KB | WB-to-decision linkages, NER results |
| ner_all_decisions.json | nap_model-main/data/ | 157 KB | Pre-extracted named entities |
| ner_training_data.json | nap_model-main/data/ | 1.9 MB | NER training annotations |
| extracted_costs.json | nap_model-main/data/ | varies | Cost data from decisions |
| top_20_case_studies.json | nap_model-main/data/ | 39 KB | Detailed case analyses |
| mcda_rankings_full.json | nap_model-main/data/ | 11 MB | RCRS baseline for all 15,688 WBs |
| nap_quantitative_data_v2.17.json | nap_model-main/agent/ | 171 MB | Master database |
| 38 Ansökningar PDFs | Data/Ansökningar/ | 79.5 MB | Application documents (not yet extracted) |
| nap_analysis.db | nap_model-main/database/ | 4 KB | SQLite DB (empty - no tables) |

---

## 9. Training Data Feasibility Assessment

### Dataset Size
| Split | Count (70/15/15) | Adequate? |
|-------|-------------------|-----------|
| Training | ~32 decisions | SMALL but viable |
| Validation | ~7 decisions | Minimum viable |
| Test | ~7 decisions | Minimum viable |

**With Ansökningar extraction**: Could add up to 38 more documents, bringing total to ~84 documents.

### Proposed Labeling Strategy

**Option A: Risk Score Regression (0-100)**
- Label each decision with a continuous risk score based on:
  - Verdict outcome (favorable/unfavorable to kraftverk operator)
  - Cost burden imposed (in SEK/MSEK)
  - Environmental measures required (count and severity)
  - Timeline strictness (immediate vs. flexible compliance)
  - Precedent impact (first-instance vs. appeal, affirmed vs. overturned)

**Option B: Risk Classification (3-class)**
- HIGH_RISK (unfavorable to operator): Expensive measures ordered, strict deadlines
- MEDIUM_RISK (mixed outcome): Partial measures, flexible timeline
- LOW_RISK (favorable to operator): Minimal measures, economic arguments prevailed

**Recommendation**: Start with **Option B (classification)** due to small dataset size. 3 classes are more learnable with 46 samples than continuous regression. Can refine to regression later with data augmentation.

### Data Augmentation Strategies
1. **Section-level splitting**: Each decision has 4-6 sections. Train on sections independently → ~200 training samples
2. **Sliding window**: Long decisions (12K-91K words) can be split into 512-token windows → 10-100x more samples
3. **Ansökningar extraction**: Add 38 application documents (different document type but same domain)
4. **Cross-validation**: Use k-fold (k=5) instead of fixed split to maximize training data usage

---

## 10. Identified Challenges

| # | Challenge | Severity | Mitigation |
|---|-----------|----------|------------|
| 1 | **Small dataset** (46 decisions) | HIGH | Section splitting, sliding window augmentation, k-fold CV |
| 2 | **VISS ID linkage gap** (only 5 IDs) | HIGH | Map via kraftverk names + kommun + water body names |
| 3 | **Long documents** (avg 17K words, BERT max 512 tokens) | MEDIUM | Hierarchical approach or key section extraction |
| 4 | **OCR artifacts** in text | LOW | Regex cleanup of headers, footers, page numbers |
| 5 | **Labeling subjectivity** | MEDIUM | Define clear rubric, consider inter-annotator agreement |
| 6 | **Class imbalance** risk | MEDIUM | Stratified splitting, class weights during training |
| 7 | **RCRS baseline is poorly discriminating** | LOW (actually helps!) | Makes improvement easier to demonstrate |
| 8 | **No pre-downloaded model weights** | LOW | Download KB-BERT from HuggingFace at training time |

---

## 11. Recommendations

### Preprocessing Pipeline (Priority Order)
1. **Clean OCR artifacts**: Remove page headers/footers, normalize whitespace
2. **Section segmentation**: Split decisions into Bakgrund/Yrkanden/Domskäl/Domslut
3. **Entity preservation**: Keep legal citations, costs, water body names intact
4. **Text chunking**: Create 512-token windows with overlap for BERT input
5. **Ansökningar extraction**: Extract text from 38 application PDFs (PyMuPDF/pdfplumber)

### Model Architecture
- **Base model**: `KB/bert-base-swedish-cased` (best Swedish BERT available)
- **Task**: 3-class classification (HIGH/MEDIUM/LOW risk) initially
- **Input strategy**: Use DOMSLUT + DOMSKÄL sections (most informative for risk)
- **Fallback**: If section segmentation fails, use first + last 256 tokens of each document
- **Evaluation**: 5-fold stratified cross-validation due to small dataset

### Timeline Estimate
| Phase | Duration | Output |
|-------|----------|--------|
| Preprocessing & cleaning | 1-2 days | cleaned_court_texts.json |
| Labeling (manual review needed) | 1-2 days | labeled_dataset.json |
| Model fine-tuning & evaluation | 2-3 days | trained model + metrics |
| Prediction & comparison | 1 day | LegalBERT vs RCRS comparison |
| Documentation & visualization | 1 day | Final report + plots |
| **Total** | **6-9 days** | |

---

## 12. File Location Summary

### Primary Data Sources (USE THESE)
```
Court texts:     Data/Domar/data/processed/court_decisions/*.txt  (46 unique)
Application PDFs: Data/Ansökningar/*.pdf                          (38 files)
MCDA rankings:   nap_model-main/data/mcda_rankings_full.json      (15,688 WBs)
Rich court DB:   nap_model-main/data/rich_court_database.json     (WB linkages)
NER data:        nap_model-main/data/ner_all_decisions.json       (606 entities)
Cost data:       nap_model-main/data/extracted_costs.json
Case studies:    nap_model-main/data/top_20_case_studies.json
```

### Duplicate Locations (AVOID - same data)
```
nap_model-main/data/processed/court_decisions/  = SAME as Data/Domar/data/processed/court_decisions/
nap_model-main/data/dom1.pdf, dom2.pdf, dom3.pdf = SAME as Data/Domar/data/
```

### Base Model
```
Repository:      swedish-bert-models/ (KBLab Swedish BERT docs)
HuggingFace ID:  KB/bert-base-swedish-cased (download at training time)
```

---

*Report generated by automated exploration. Manual verification of labeling strategy recommended before proceeding to Task 2.*
