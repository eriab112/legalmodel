# Pipeline Integration Results: 4 New Court Decisions + Sundin Validation

**Date:** 2026-02-13
**Task:** Integrate 4 new court decisions into the NAP Legal AI pipeline, extract Sundin features, and re-run validation.

---

## Summary

All 10 pipeline steps completed successfully. The system now has **50 cleaned decisions** and **44 labeled decisions** (up from 46/40). Sundin feature extraction and validation have been re-run on the full 44-decision dataset.

---

## Pipeline State After Integration

### cleaned_court_texts.json — 50 decisions

- Previously 46, now 50 after adding 4 new court decisions
- All entries contain: `text_full`, `sections`, `key_text`, `metadata`, `extracted_costs`, `extracted_measures`, `text_hash`, `stats`
- 6 non-hydropower decisions remain in cleaned but are excluded from labeling

### labeled_dataset.json — 44 decisions

| Split | Count |
|-------|-------|
| Train | 30 |
| Val | 7 |
| Test | 7 |
| **Total** | **44** |

**Label distribution:**

| Label | Count | % |
|-------|-------|---|
| HIGH_RISK | 8 | 18% |
| MEDIUM_RISK | 26 | 59% |
| LOW_RISK | 10 | 23% |

**Excluded (non-hydropower):** m8024-05, m7708-22, m899-23, m3273-22, m2479-22, m2024-01

### label_overrides.json — 11 entries

- 6 EXCLUDE overrides (non-hydropower)
- 1 HIGH_RISK override (m3753-22)
- 4 new decision labels (see below)

---

## The 4 New Decisions

| ID | Filename | Court | Label | Split |
|----|----------|-------|-------|-------|
| m483-22 | Växjö TR M 483-22 Dom 2026-01-22.txt | Växjö | LOW_RISK | val |
| m2796-24 | Växjö TR M 2796-24 Dom 2025-03-07.txt | Växjö | MEDIUM_RISK | train |
| m2694-22 | Östersunds TR M 2694-22 Dom 2025-12-18.txt | Östersund | MEDIUM_RISK | train |
| m2695-22 | Östersunds TR M 2695-22 Dom 2025-08-15.txt | Östersund | MEDIUM_RISK | train |

### Sundin Features for the 4 New Decisions

**m483-22 (LOW_RISK)**
- Downstream: gap 15mm, angle 80°, bypass 10 L/s
- Upstream: nature-like type, slope 6.5%, no fishway bool, no eel ramp
- Flow: min 100 L/s, no hydropeaking ban
- Monitoring: required, not functional
- Cost/timeline: not specified

**m2796-24 (MEDIUM_RISK)**
- Downstream: gap 13mm, angle 80°
- Upstream: fishway present (undefined type), slope 12%, discharge 10 L/s, no eel ramp
- Flow: hydropeaking banned
- Monitoring: required, not functional
- Cost/timeline: not specified

**m2694-22 (MEDIUM_RISK)**
- Downstream: gap 12mm, angle 45°
- Upstream: fishway present (nature-like), eel ramp present
- Flow: hydropeaking banned
- Monitoring: required, not functional
- Cost/timeline: not specified

**m2695-22 (MEDIUM_RISK)**
- Downstream: gap 13mm, angle 30°
- Upstream: fishway present (vertical-slot), slope 10%, eel ramp present, flow_percent_mq 10%
- Flow: no hydropeaking ban
- Monitoring: required, functional evaluation
- Cost/timeline: not specified

---

## Sundin Validation Results (n=44)

### RF Feature Importance (5-fold stratified CV)

| Rank | Feature | Importance | Std |
|------|---------|-----------|-----|
| 1 | upstream_has_eel_ramp | 0.158 | 0.050 |
| 2 | upstream_slope_pct | 0.123 | 0.029 |
| 3 | upstream_type_int | 0.109 | 0.039 |
| 4 | downstream_angle_degrees | 0.105 | 0.020 |
| 5 | monitoring_required | 0.104 | 0.040 |
| 6 | upstream_has_fishway | 0.098 | 0.044 |
| 7 | cost_msek | 0.069 | 0.027 |
| 8 | flow_min_ls | 0.066 | 0.023 |
| 9 | downstream_gap_mm | 0.049 | 0.019 |
| 10 | flow_percent_mq | 0.039 | 0.020 |
| 11 | upstream_discharge_ls | 0.031 | 0.024 |
| 12 | monitoring_functional | 0.031 | 0.033 |
| 13 | flow_hydropeaking_banned | 0.015 | 0.013 |
| 14 | timeline_years | 0.003 | 0.003 |
| 15 | downstream_has_screen | 0.000 | 0.000 |
| 16 | downstream_bypass_ls | 0.000 | 0.000 |

**Interpretation:** Upstream passage features (eel ramp, type, slope) dominate. Downstream angle and monitoring also matter. Screen presence and bypass flow have zero importance in this dataset.

### KMeans Clustering (k=3)

- **Adjusted Rand Index: 0.0353** (very low)
- **21 out of 44 decisions** have cluster-label disagreements
- Cluster-to-risk mapping (by mean cost heuristic): Cluster 1 → HIGH, Cluster 0 → MEDIUM, Cluster 2 → LOW

**Disagreements involving new decisions:**
- m2695-22: labeled MEDIUM_RISK, cluster suggests HIGH_RISK

**Interpretation:** The low ARI indicates human labels incorporate contextual judgment beyond what Sundin features alone capture. Clustering is useful for diagnostics only, not auto-relabeling.

---

## Files Modified/Created

| File | Action | Result |
|------|--------|--------|
| `Data/processed/cleaned_court_texts.json` | Updated by 02 script | 50 decisions |
| `Data/processed/labeled_dataset.json` | Updated by 03 script | 44 decisions (30/7/7 split) |
| `Data/processed/label_overrides.json` | Updated | 11 entries (6 exclude + 5 labels) |
| `Data/processed/decision_features_sundin2026.json` | Updated by sundin_feature_extraction.py | 44 decisions |
| `Data/processed/sundin_feature_importance.json` | Updated by sundin_validation.py | 16 features, n=44 |
| `Data/processed/clustering_validation_report.json` | Updated by sundin_validation.py | ARI=0.0353, 21 disagreements |
| `Data/processed/new_4_decisions_features.json` | Created | Intermediate: 4 new decision features |
| `Data/processed/label_suggestions_4_new.json` | Created | Intermediate: ML label suggestions |
| `scripts/sundin_validation.py` | Fixed | Line 141: hardcoded note → dynamic n_samples |

---

## Code Fix Applied

`scripts/sundin_validation.py` line 141 — changed hardcoded note to dynamic:

```python
# Before:
"note": "Averaged over 5-fold stratified CV; n_samples=40 (or 44). Interpret as correlation with risk in this set."

# After:
"note": f"Averaged over 5-fold stratified CV; n_samples={len(labels)}. Interpret as correlation with risk in this set."
```

---

## Known Issues / Notes for Next Steps

1. **MEDIUM_RISK bias** — 59% of labeled decisions are MEDIUM_RISK. Consider whether some should be reclassified.
2. **Low ARI (0.0353)** — Labels don't follow natural Sundin feature clusters. This is expected (human judgment incorporates more context) but worth noting for model training.
3. **Sparse features** — Many decisions have null cost_msek and timeline_years. These features have low RF importance as a result.
4. **downstream_has_screen and downstream_bypass_ls** have zero importance — may want to review extraction quality or consider dropping from the feature set.
5. **Next pipeline steps** from STRATEGY_AND_PHASE_A.md:
   - DAPT (A6): Run domain-adaptive pretraining using `dapt_corpus.json` (96 docs, 1.42M words)
   - Retraining (A7): Fine-tune LegalBERT with weak labels in train only, sliding window (512/256)
   - Phase B: Multi-task learning with auxiliary Sundin-feature heads
