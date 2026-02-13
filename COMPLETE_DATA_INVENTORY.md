# Complete NAP Data Inventory

**Date:** 2026-02-12
**Analyst:** Claude Code
**Scope:** ALL files in `legalmodel/Data/` (149 files, 175.5 MB)

---

## Executive Summary

### Current Training Data
- **Used:** 40 court decisions (labeled HIGH/MEDIUM/LOW)
- **Unused but available:** 109 documents across 3 folders
- **Current model accuracy:** 65% (5-fold CV)

### Key Findings
1. **Ansokningar folder contains NO matching court decisions** -- zero case number overlap with Domar. These are separate, ongoing cases without verdicts yet. Cannot create application-verdict training pairs.
2. **13 of 39 Ansokningar PDFs are scanned images** (no extractable text, need OCR). Only 26 are text-extractable.
3. **Lagstiftningdiverse is a goldmine for domain pre-training**: 571K words of Swedish legal/environmental text, 26 files highly NAP-relevant.
4. **All 6 excluded decisions were correctly excluded** -- none should be re-included.
5. **6 duplicate files + 2 non-court files** exist in the court_decisions folder (needs cleanup).
6. **4 NEW court decisions (Domar)** exist in Ansokningar as PDFs that were never processed as training data.

---

## Detailed Inventory

### 1. Ansokningar (Applications) -- `Data/Ansokningar/`

| Metric | Value |
|--------|-------|
| Total PDFs | 39 |
| Successfully extracted | 39 (but 13 are scanned) |
| Text-extractable | 26 files |
| Scanned (need OCR) | 13 files |
| Total words (text-based) | 217,263 |
| NAP-relevant | 26/39 (67%) |
| Matched to court decisions | **0 pairs** |
| **Usable for direct training?** | **Limited** |

#### Document Types Found

| Type | Count | Total Words | Description |
|------|-------|-------------|-------------|
| Ansokan (Application) | 24 | ~92K | Actual permit applications filed with courts |
| Dom (Judgment) | 4 | 75,212 | **Court decisions misfiled in Ansokningar!** |
| Dagboksblad (Docket) | 5 | 22,122 | Court docket/proceedings summaries |
| Beslut (Interim Decision) | 4 | 2,096 | Procedural decisions during trial |
| Komplettering (Supplement) | 1 | 11 | Scanned supplementary filing |
| Unknown | 1 | 40,742 | UUID-named file, large application |

#### CRITICAL FINDING: 4 Court Decisions Misfiled as Ansokningar

These are actual **domar** (judgments) that should be processed as training data:

| File | Words | Pages | Case |
|------|-------|-------|------|
| Vaxjo TR M 483-22 Dom 2026-01-22.pdf | 24,487 | 92 | M 483-22 |
| Ostersunds TR M 2694-22 Dom 2025-12-18.pdf | 22,545 | 85 | M 2694-22 |
| Ostersunds TR M 2695-22 Dom 2025-08-15.pdf | 16,390 | 59 | M 2695-22 |
| Vaxjo TR M 2796-24 Dom 2025-03-07.pdf | 11,790 | 44 | M 2796-24 |

**These 4 domar represent +75K words of new court decision text that could expand training from 40 to 44 decisions.**

#### Scanned PDFs (13 files -- need OCR)

All from Vanersborg TR (M 3413-22 through M 3460-22) plus Umea TR (M 2430-22, M 2431-22) and one Nacka TR supplement. Total ~200 pages of scanned images.

#### Training Strategy for Ansokningar

The 20 genuine ansokningar (applications) cannot be directly labeled with outcome labels since no matching court decisions exist. However, they can be used for:

1. **Domain pre-training** -- 92K words of Swedish legal/hydropower text
2. **RAG context** -- Rich context documents for retrieval-augmented generation
3. **Future labeling** -- When corresponding court decisions are issued, these become training pairs

---

### 2. Lagstiftningdiverse (Legislation) -- `Data/Lagstiftningdiverse/`

| Metric | Value |
|--------|-------|
| Total files | 38 (36 PDFs + 2 TXTs) |
| Successfully extracted | 38 (100%) |
| Total words | **571,306** |
| NAP-relevant | 28/38 (74%) |
| Domain pre-training candidates | 26 files |
| **Usable for training?** | **YES -- domain pre-training** |

#### By Category

| Category | Files | Words | NAP-Relevant | Training Use |
|----------|-------|-------|--------------|--------------|
| Technical Guidelines | 12 | 83,375 | 12 (100%) | Pre-training + RAG |
| EU Directives | 4 | 148,812 | 0 | Pre-training (general) |
| Swedish Laws | 5 | 76,760 | 5 (100%) | Pre-training |
| Water District Regulations | 4 | 69,435 | 1 | Pre-training |
| Environmental Assessments | 5 | 44,185 | 5 (100%) | Pre-training + RAG |
| Report/Consultation | 4 | 85,207 | 2 | Pre-training |
| NAP Government Decisions | 2 | 11,271 | 2 (100%) | Pre-training + RAG |
| Other | 2 | 52,261 | 1 | Pre-training |

#### Key Documents

| File | Words | Category | Significance |
|------|-------|----------|-------------|
| miljobalken.pdf | 38,174 | Swedish Law | **The Environmental Code** -- foundational law |
| NAP_full_text.txt | 9,153 | NAP Government | National Plan full text |
| VM_Riktlinjer_Vattenkraft.txt | 19,833 | Technical Guideline | Water authority guidelines |
| miljoatgarder-i-vattendrag-exempelsamling... | 46,441 | Technical Guideline | Practical examples of environmental measures |
| eudirektiv.pdf | 28,652 | EU Directive | EU Water Framework Directive |
| CIS_Guidance_Article_4_7_FINAL.pdf | 41,129 | EU Directive | Article 4(7) exemption guidance |
| se-final-paf-22-nov-2021.pdf | 44,167 | Report | Priority Action Framework |
| 5x Miljokonsekvensbeskrivning files | ~44K | Env. Assessment | All 5 water districts |
| bilaga-5-* series (8 files) | ~33K | Technical | Fish passage, costs, dam safety |
| regeringsbeslut-nationell-plan... | 2,118 | NAP Government | Government decision on NAP |

#### Note: 1 Duplicate

`CIS_Guidance_Article_4_7_FINAL (1).pdf` is a duplicate of `CIS_Guidance_Article_4_7_FINAL.pdf` (both 41,129 words).

#### Training Strategy for Lagstiftning

**Domain-Adaptive Pre-Training (DAPT):**
- Use all 571K words (deduped: ~530K) as continued pre-training corpus for KB-BERT
- This teaches the model Swedish legal vocabulary, environmental law concepts, and hydropower-specific terminology BEFORE fine-tuning on the 40 labeled decisions
- Expected gain: 5-10% accuracy improvement based on DAPT literature

---

### 3. Domar (Court Decisions) -- `Data/Domar/`

| Metric | Value |
|--------|-------|
| PDFs in data/ | 13 (source documents) |
| TXT files in court_decisions/ | 54 |
| Unique court decisions | 46 |
| Used for training | 40 |
| Correctly excluded | 6 |
| Duplicate files | 6 (to remove) |
| Non-court files | 2 (to relocate) |

#### Excluded Decisions (6) -- All Correctly Excluded

| ID | File | Reason | Re-include? |
|----|------|--------|-------------|
| f-14649-21 | f-14649-21-dom-2022-11-16.txt | Fastighetsreglering (property regulation) | No |
| m-7708-22 | m-7708-22-dom-2023-09-12.txt | EU emission allowances | No |
| m-899-23 | m-899-23-dom-2024-12-03.txt | Quarry operations | No |
| MOD_M-10258-23 | MOD_M-10258-23_2024-12-19.txt | Rock quarry permit | No |
| MOD_M-13461-22 | MOD_M-13461-22_2024-04-03.txt | Solar panel installation | No |
| Vaxjo_TR_P_3853-22 | Vaxjo_TR_P_3853-22_Dom_2024-01-31.txt | Zoning plan appeal | No |

#### Duplicate Files to Remove (6)

| File | Duplicate Of |
|------|-------------|
| NAP_M3426-24_..._Karsbols_v2.txt | NAP_M3426-24_..._Karsbols.txt |
| NAP_M3426-24_..._Karsbols_v3.txt | NAP_M3426-24_..._Karsbols.txt |
| NAP_M16477-23_..._v2.txt | NAP_M16477-23_2025-03-10.txt |
| MOD_M10258-23_2024-12-19.txt | MOD_M-10258-23_2024-12-19.txt |
| dom1.txt | NAP_M9349-24_2025-10-02.txt |
| dom2.txt | MOD_M10196-24_2025-09-24.txt |

#### Non-Court Files to Relocate (2)

| File | Actual Content |
|------|---------------|
| SvK_NAP_Slutrapport_2023.txt | Svenska kraftnat report on grid consequences of NAP |
| VM_Riktlinjer_Vattenkraft.txt | Water authority guidelines (duplicate of Lagstiftningdiverse copy) |

Both are valuable domain documents but should not be in court_decisions/.

---

### 4. Processed Data -- `Data/processed/`

| File | Size | Content |
|------|------|---------|
| cleaned_court_texts.json | 12.1 MB | 46 cleaned court decision texts |
| labeled_dataset.json | 1.2 MB | 40 labeled decisions (train/val/test splits) |
| label_overrides.json | <1 KB | 6 EXCLUDE + 1 label override |
| linkage_table.json | <1 KB | 16/17 water bodies linked to VISS |
| label_review.txt | <1 KB | Manual review notes |

---

## Complete Data Summary

| Source | Files | Words | Used? | Training Potential |
|--------|-------|-------|-------|--------------------|
| Court decisions (labeled) | 40 txt | ~952K | YES | Current training set |
| Court decisions (excluded) | 6 txt | ~105K | Excluded | Negative examples only |
| **New domar in Ansokningar** | **4 pdf** | **75K** | **NO** | **+4 labelable decisions** |
| Ansokningar (applications) | 20 pdf | 92K | NO | Domain pre-training, RAG |
| Ansokningar (scanned) | 13 pdf | 0 (images) | NO | Needs OCR first |
| Dagboksblad | 5 pdf | 22K | NO | Procedural context |
| Beslut (interim) | 4 pdf | 2K | NO | Too short for training |
| Lagstiftningdiverse | 38 files | 571K | NO | Domain pre-training |
| Non-court reference docs | 2 txt | 30K | NO | Domain pre-training |
| **TOTAL** | **132 unique** | **~1.85M** | **40 used** | **See below** |

---

## Training Data Expansion Opportunities

### Option 1: Add 4 Misfiled Court Decisions (QUICK WIN)

**Approach:** Extract text from the 4 domar PDFs in Ansokningar, label them (HIGH/MEDIUM/LOW), add to training set.

| Metric | Value |
|--------|-------|
| Expected gain | +4 training examples (40 -> 44) |
| Effort | 2-4 hours (extract + manual label) |
| Risk | Low |
| Expected accuracy improvement | +2-3% |

**Implementation:**
1. Extract text from the 4 dom PDFs (already done -- in `ansokan_texts.json`)
2. Clean and format like existing court decisions
3. Manually label each (read domslut section)
4. Add to `labeled_dataset.json`
5. Re-run 5-fold CV

### Option 2: Domain-Adaptive Pre-Training (DAPT)

**Approach:** Continue pre-training KB-BERT on the 530K-word legal corpus from Lagstiftningdiverse + applications before fine-tuning on labeled decisions.

| Metric | Value |
|--------|-------|
| Corpus size | ~530K words (Lagstiftningdiverse) + 92K (applications) + 30K (reference) = ~650K words |
| Expected gain | Better Swedish legal language understanding |
| Effort | 3-5 days (scripting + GPU training time) |
| Risk | Low-Medium (well-established technique) |
| Expected accuracy improvement | +5-10% |

**Implementation:**
1. Combine all Lagstiftningdiverse texts + application texts into a pre-training corpus
2. Continue MLM pre-training of KB-BERT for 3-5 epochs
3. Then fine-tune on the 40-44 labeled decisions as before

### Option 3: OCR for Scanned Applications

**Approach:** Run OCR on 13 scanned PDFs (~200 pages) to extract text.

| Metric | Value |
|--------|-------|
| Expected gain | +13 application documents (~50-80K words estimated) |
| Effort | 1-2 days (OCR setup + processing + QA) |
| Risk | Medium (OCR quality varies) |
| Expected accuracy improvement | Indirect -- adds to pre-training corpus |

### Option 4: Section-Level Data Augmentation (Already Planned)

**Approach:** Split each court decision into sections (Domslut, Domskal, Bakgrund) and train on sections independently.

| Metric | Value |
|--------|-------|
| Expected gain | 3-5x more training examples from existing 40 decisions |
| Effort | 2-3 days |
| Risk | Medium (label propagation from document to section) |
| Expected accuracy improvement | +5-8% |

### Option 5: Source More Court Decisions Externally

**Approach:** Search Swedish court databases for additional NAP-related decisions.

| Metric | Value |
|--------|-------|
| Expected gain | Potentially 20-50+ new decisions |
| Effort | 1-2 weeks (sourcing + processing + labeling) |
| Risk | Low (if decisions are publicly available) |
| Expected accuracy improvement | +10-15% |

---

## Recommendations

### Immediate Actions (This Week)

1. **Process the 4 misfiled domar** -- extract, label, add to training set (+4 examples, 2 hours)
2. **Clean up court_decisions folder** -- remove 6 duplicates, relocate 2 non-court files
3. **Prepare DAPT corpus** -- combine Lagstiftningdiverse texts into a single pre-training corpus file

### Short-term (1-2 Weeks)

4. **Run domain-adaptive pre-training** on KB-BERT with the 650K word legal corpus
5. **Implement section-level splitting** for data augmentation
6. **Re-train model** with expanded data (44 decisions + DAPT + section augmentation)

### Long-term (1 Month+)

7. **OCR the 13 scanned PDFs** to expand pre-training corpus
8. **Source additional court decisions** from Swedish court databases
9. **Investigate ansokningar-dom pairing** as new verdicts are issued for the 20 pending cases

---

## Projected Model Improvement

| Scenario | Training Examples | Pre-training Corpus | Expected Accuracy | Effort |
|----------|-------------------|--------------------|--------------------|--------|
| Current baseline | 40 | None (raw KB-BERT) | 65% | Done |
| + 4 new domar | 44 | None | 67-68% | 2 hours |
| + DAPT on legal corpus | 44 | 650K words | 72-76% | 3-5 days |
| + Section augmentation | ~150-200 sections | 650K words | 75-80% | +2-3 days |
| + External sourcing | 60-90 decisions | 650K words | 78-85% | +1-2 weeks |

---

## Data Quality Assessment

### High Quality (Ready to Use)
- 40 labeled court decisions (already in training)
- 4 misfiled domar PDFs (text extracted, need labeling only)
- 2 TXT files in Lagstiftningdiverse (NAP_full_text.txt, VM_Riktlinjer_Vattenkraft.txt)

### Medium Quality (Needs Processing)
- 36 Lagstiftningdiverse PDFs (extracted, may need cleaning)
- 20 genuine ansokningar (extracted but no labels available)
- 5 dagboksblad documents
- SvK_NAP_Slutrapport_2023.txt, VM_Riktlinjer_Vattenkraft.txt (in Domar)

### Low Quality (Needs OCR)
- 13 scanned Ansokningar PDFs (~200 pages, image-only)

### Not Usable for Training
- 4 beslut documents (too short, 272-614 words, procedural only)
- 6 excluded court decisions (confirmed non-hydropower)

---

## Appendix A: Complete File Listing

### Ansokningar/ (39 files, 82.3 MB)
```
0f516575-0b08-4303-9dfe-33de32915437.pdf          (40,742 words - Unknown type)
Nacka TR M 5642-24 Aktbil 1, Ansokan (e-post).pdf (10,611 words - Application)
Nacka TR M 5642-24 Aktbil 51, Konsoliderad...pdf  (11 words - SCANNED)
Nacka TR M 5642-24 Dagboksblad 2026-02-06.pdf     (4,335 words - Docket)
Umea TR M 2430-22 Aktbil 1.pdf                    (11 words - SCANNED)
Umea TR M 2431-22 Aktbil 1.pdf                    (11 words - SCANNED)
Umea TR M 302-22 Aktbil 1.pdf                     (5,030 words - Application)
Umea TR M 303-22 Aktbil 1.pdf                     (12,018 words - Application)
Vanersborgs TR M 2743-23 Aktbil 1.pdf             (6,824 words - Application)
Vanersborgs TR M 2744-23 Aktbil 1.pdf             (5,882 words - Application)
Vanersborgs TR M 2746-23 Aktbil 1.pdf             (7,545 words - Application)
Vanersborgs TR M 2747-23 Aktbil 1.pdf             (6,118 words - Application)
Vanersborgs TR M 3125-23 Aktbil 1.pdf             (11,105 words - Application)
Vanersborgs TR M 3413-22 Aktbil 1.pdf             (10 words - SCANNED)
Vanersborgs TR M 3414-22 Aktbil 1.pdf             (10 words - SCANNED)
Vanersborgs TR M 3415-22 Aktbil 1.pdf             (10 words - SCANNED)
Vanersborgs TR M 3420-22 Aktbil 1.pdf             (10 words - SCANNED)
Vanersborgs TR M 3423-22 Aktbil 1.pdf             (10 words - SCANNED)
Vanersborgs TR M 3429-22 Aktbil 1.pdf             (10 words - SCANNED)
Vanersborgs TR M 3430-22 Aktbil 1.pdf             (10 words - SCANNED)
Vanersborgs TR M 3433-22 Aktbil 1.pdf             (10 words - SCANNED)
Vanersborgs TR M 3435-22 Aktbil 1.pdf             (10 words - SCANNED)
Vanersborgs TR M 3460-22 Aktbil 1.pdf             (10 words - SCANNED)
Vanersborgs TR M 3472-22 Aktbil 1.pdf             (5,265 words - Application)
Vanersborgs TR M 422-22 Aktbil 1.pdf              (4,380 words - Application)
Vaxjo TR M 2796-24 Dom 2025-03-07.pdf             (11,790 words - DOM/NEW)
Vaxjo TR M 483-22 Dom 2026-01-22.pdf              (24,487 words - DOM/NEW)
Vaxjo TR M 485-22 Beslut 2024-01-10.pdf           (272 words - Interim decision)
Vaxjo TR M 486-22 Beslut 2023-06-21.pdf           (611 words - Interim decision)
Vaxjo TR M 488-22 Beslut 2023-06-21.pdf           (614 words - Interim decision)
Vaxjo TR M 533-22 Beslut 2023-06-21.pdf           (599 words - Interim decision)
Ostersunds TR M 2694-22 Dagboksblad 2026-02-06.pdf(3,716 words - Docket)
Ostersunds TR M 2694-22 Dom 2025-12-18.pdf        (22,545 words - DOM/NEW)
Ostersunds TR M 2695-22 Dagboksblad 2026-02-06.pdf(3,546 words - Docket)
Ostersunds TR M 2695-22 Dom 2025-08-15.pdf        (16,390 words - DOM/NEW)
Ostersunds TR M 285-22 Aktbil 1, Ansokan.pdf      (1,415 words - Application)
Ostersunds TR M 285-22 Dagboksblad 2026-02-06.pdf (6,835 words - Docket)
Ostersunds TR M 287-22 Aktbil 1, Ansokan.pdf      (898 words - Application)
Ostersunds TR M 287-22 Dagboksblad 2026-02-06.pdf (3,690 words - Docket)
```

### Lagstiftningdiverse/ (38 files, 60.0 MB)
```
EU DIRECTIVES (4 files, 148,812 words):
  CIS_Guidance_Article_4_7_FINAL.pdf               41,129 words
  CIS_Guidance_Article_4_7_FINAL (1).pdf            41,129 words (DUPLICATE)
  eudirektiv.pdf                                    28,652 words
  Guidance No 4 - heavily modified water bodies.pdf 37,902 words

TECHNICAL GUIDELINES (12 files, 83,375 words):
  VM-riktlinjer-vattenkraft_atgarder-undantag.pdf   19,589 words
  VM_Riktlinjer_Vattenkraft.txt                     19,833 words
  bilaga-5-1-dammsakerhet.pdf                        1,075 words
  bilaga-5-4-fiskinformation-malarter.pdf            2,249 words
  bilaga-5-5-losningar-uppstromsvandrande-fisk.pdf   9,823 words
  bilaga-5-6-stromning-vattendrag-fiskpassager.pdf   7,240 words
  bilaga-5-7-hydraulisk-design-lockvatten.pdf        3,339 words
  bilaga-5-8-hydrauliska-samband-fiskpassager.pdf    3,529 words
  bilaga-5-11-losningar-nedstromsvandrande-fisk.pdf  3,751 words
  bilaga-5-12-kostnader.pdf                          1,950 words
  allmanna-villkor-reviderade-20240327.pdf            5,816 words
  vagledning-natura2000-beslutad.pdf                 5,181 words

SWEDISH LAWS (5 files, 76,760 words):
  miljobalken.pdf                                   38,174 words
  swe126179.pdf                                      5,744 words
  swe187160.pdf                                      3,985 words
  swe202925.pdf                                      4,820 words
  swe221922.pdf                                     24,037 words

ENVIRONMENTAL ASSESSMENTS (5 files, 44,185 words):
  Miljokonsekvensbeskrivning Atgardsprogram Bottenhavet.pdf    8,673 words
  Miljokonsekvensbeskrivning Atgardsprogram Bottenviken.pdf    8,705 words
  Miljokonsekvensbeskrivning Atgardsprogram Norra Ostersjon.pdf 9,190 words
  Miljokonsekvensbeskrivning Atgardsprogram Sodra Ostersjon.pdf 8,997 words
  Miljokonsekvensbeskrivning Atgardsprogram Vasterhavet.pdf    8,620 words

WATER DISTRICT REGULATIONS (4 files, 69,435 words):
  2021-43 Vattenkvalitetskrav Vasterhavets.pdf      63,777 words
  19FS 2021-10-TGA.pdf                               1,874 words
  202111 Kvalitetskrav Sodra Ostersjons.pdf           1,895 words
  22FS 2021-5 Bottenhavets vattendistrikt.pdf         1,889 words

NAP GOVERNMENT (2 files, 11,271 words):
  NAP_full_text.txt                                  9,153 words
  regeringsbeslut-nationell-plan.pdf                  2,118 words

REPORTS & CONSULTATIONS (4 files, 85,207 words):
  miljoatgarder-i-vattendrag-exempelsamling.pdf     46,441 words
  se-final-paf-22-nov-2021.pdf                      44,167 words
  rapport-2025-19-samradsunderlag.pdf                4,316 words
  remiss-2025-3666-miljobedomning.pdf                  445 words

OTHER (2 files, 52,261 words):
  remiss-2025-3660-marin-strategi.pdf               34,005 words
  GD 07 - Monitoring - Policy Summary.pdf            8,094 words
```

### Domar/data/processed/court_decisions/ (54 files -> 46 unique)
```
INCLUDED IN TRAINING (40 decisions):
  [See labeled_dataset.json for full list]
  Distribution: 8 HIGH, 23 MEDIUM, 9 LOW

EXCLUDED (6 decisions):
  f-14649-21-dom-2022-11-16.txt        (property regulation)
  m-7708-22-dom-2023-09-12.txt         (emission allowances)
  m-899-23-dom-2024-12-03.txt          (quarry operations)
  MOD_M-10258-23_2024-12-19.txt        (rock quarry)
  MOD_M-13461-22_2024-04-03.txt        (solar panels)
  Vaxjo_TR_P_3853-22_Dom_2024-01-31.txt (zoning plan)

DUPLICATES TO REMOVE (6 files):
  NAP_M3426-24_..._v2.txt, _v3.txt
  NAP_M16477-23_..._v2.txt
  MOD_M10258-23_2024-12-19.txt (missing hyphen variant)
  dom1.txt, dom2.txt

NON-COURT FILES TO RELOCATE (2 files):
  SvK_NAP_Slutrapport_2023.txt         (government report)
  VM_Riktlinjer_Vattenkraft.txt        (guidelines, duplicate)
```

---

## Appendix B: Generated Output Files

| File | Description |
|------|-------------|
| `COMPLETE_FILE_INVENTORY.json` | Full file inventory by extension |
| `Data/processed/ansokan_texts.json` | All 39 ansokningar extracted texts |
| `Data/processed/ansokan_structure_analysis.json` | Content analysis per ansokningar |
| `Data/processed/ansokan_dom_matches.json` | Case number matching results |
| `Data/processed/lagtiftning_texts.json` | All 38 legislation texts |
| `Data/processed/lagtiftning_categorization.json` | Categorization per file |
| `Data/processed/lagtiftning_usefulness.json` | Training potential assessment |

---

## Appendix C: Answers to Success Criteria

### 1. How many total documents are available?
- Court decisions: 46 unique (40 labeled + 6 excluded)
- Applications (text-based): 20 genuine ansokningar
- Applications (scanned): 13 (need OCR)
- **New court decisions found:** 4 (misfiled in Ansokningar)
- Legislation/reference: 38 files
- Docket sheets: 5
- Interim decisions: 4
- Reference documents: 2
- **TOTAL: 132 unique documents**

### 2. How many can be used for training?
- Direct labels: 40 (current) + 4 new domar = **44 labelable decisions**
- Domain pre-training corpus: ~650K words (Lagstiftning + applications)
- Section augmentation from 44 decisions: ~150-200 training examples
- **POTENTIAL: 44 labeled decisions + 650K word pre-training corpus**

### 3. What's the expansion strategy?
- **Recommended:** Option 1 (add 4 domar) + Option 2 (DAPT) + Option 4 (section augmentation)
- Combined approach: minimal risk, maximum gain, ~1 week effort

### 4. What's the expected improvement?
- Current: 65% accuracy
- After full expansion: 75-80% accuracy
- Timeline: 5-7 days
- Effort: ~30-40 hours

### 5. What are the risks?
- Risk 1: DAPT may not help if domain gap is small (mitigation: compare with/without)
- Risk 2: Section-level labels may introduce noise (mitigation: use domslut labels only for sections containing legal reasoning)
