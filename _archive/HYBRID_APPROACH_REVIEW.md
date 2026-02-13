# Review: Hybrid Approach (Sundin Foundation + Data-Driven Weights)

**Reviewed:** 2026-02-12  
**Context:** Adjustment to Phase A – "Sundin tells us WHAT to look for, our data tells us HOW to weight it."

**Sundin vs pappret:** Feature-taxonomin har korskontrollerats mot kmae250140 (Sundin et al. 2026); se **LABELING_STRATEGY_VS_PAPER.md** för variabel-mappning (Sektion 2.1 / Tabell 2) och slutsats om etiketteringsstrategin.

---

## Verdict: **Sound and preferable to rigid Sundin-only rules**

The principle is right: use Sundin et al. 2026 to define **feature categories and what to extract**, and use your **44 labeled decisions** to learn importance and thresholds. The main constraints are **sample size** (44) and **how auxiliary tasks get their labels** in the multi-task model. Below: what to keep, what to adjust, and how it fits with the Phase A review.

---

## 1. What works well

| Idea | Why it helps |
|------|----------------|
| **Sundin as feature taxonomy, not rule book** | Avoids overfitting to one paper’s thresholds; your data can show e.g. that gap 15 vs 18 mm doesn’t separate risk in your set. |
| **Extract values, not booleans** | `gap_width_mm`, `cost_msek`, `timeline_years` etc. give the model (and RF) something to learn from; raw values are better than hand-coded "strict" flags. |
| **Feature importance from data** | Learning which Sundin categories predict HIGH/MEDIUM/LOW is exactly the right use of the 44 labels; interpretable and defensible. |
| **Clustering to sanity-check labels** | Unsupervised structure (e.g. KMeans on Sundin features) can surface outliers and possible mislabels; good as a **review aid**, not as ground truth. |
| **Sundin-informed multi-task model** | Auxiliary heads (downstream, upstream, flow, monitoring) can act as regularisers and improve the risk head if implemented with **same-text-derived** auxiliary labels (see below). |
| **Phase B: RF (Sundin features) + BERT ensemble** | Combines interpretability (RF on Sundin) with text nuance (BERT); evaluation on same 5-fold splits keeps comparison fair. |

This aligns with the Phase A review: feature extraction (A3) stays; you add **data-driven weighting** and **validation/clustering** on top, and keep DAPT + pipeline fixes (02/03, splits, weak-only-in-train, sliding window) as before.

---

## 2. Adjustments for n = 44

### 2.1 Random Forest feature importance

- **Risk:** With 44 samples and many Sundin-derived features, a single RF fit will **overfit** and importance scores will be unstable.
- **Recommendations:**
  - **Aggregate over 5-fold CV:** For each fold, train RF on train portion of the 44, get feature importances; average (or median) importances across folds. Report mean ± std so you see stability.
  - **Regularise:** Use a small forest (e.g. `max_depth=4`, `min_samples_leaf=3`) or L1/L2 linear model (e.g. logistic regression with your features) to get more stable coefficients.
  - **Interpretation:** Frame as “which Sundin categories correlate with risk in our 44 decisions” rather than “definitive importance for the population”.

### 2.2 Clustering (KMeans, 3 clusters) vs labels

- **Use:** Good as a **diagnostic**: low Adjusted Rand Index (ARI) suggests labels don’t align with natural feature clusters; then inspect disagreeing decisions.
- **Caveats:**
  - **Cluster ↔ risk mapping is undefined:** Clusters are 0, 1, 2; you must define which cluster corresponds to HIGH/MEDIUM/LOW (e.g. by mean cost or by majority label in cluster). Document that mapping.
  - **n=44:** ARI can be noisy; don’t over-interpret small differences. Use it to **flag** decisions for re-read, not to auto-change labels.
- **Recommendation:** If ARI < ~0.3, list decisions where cluster disagrees with label and do a **targeted manual review**; don’t auto-relabel from clusters.

---

## 3. Multi-task model: where do auxiliary labels come from?

The sketch has:

- `risk_head` → HIGH/MEDIUM/LOW (you have 44 labels).
- `downstream_head`, `upstream_head`, `flow_head`, `monitoring_head` → no separate human labels.

So auxiliary tasks need **labels derived from the same text**. The natural source is **your Sundin feature extraction**:

- **Downstream:** e.g. 4 classes: no screen / inclined / angled / undefined (from `extract_sundin_informed_features`).
- **Upstream:** e.g. no fishway / nature-like / vertical-slot / eel-ramp (or similar from extraction).
- **Flow:** binary from `flow_hydropeaking_ban`.
- **Monitoring:** binary from `monitoring_required` (and optionally `monitoring_functional`).

So: **same text → extract Sundin features → use those as auxiliary targets**. BERT then learns to predict those structured aspects, and the risk head can use shared representation + optional concatenation of auxiliary logits. That’s a standard and valid setup.

**Implementation details:**

- **Missing values:** Many decisions will have `None` for e.g. `gap_width_mm`. For auxiliary heads you need a clear policy: e.g. “no screen” → one class; “screen but no numeric value” → another; or mask loss for missing. Define a small set of discrete targets per head so every document has an auxiliary label (even if “unknown”).
- **Chunk vs document:** Current training is **document-level** risk with **chunk-level** BERT (sliding window). For multi-task, simplest is **document-level** auxiliary labels (one per decision, from your feature extraction) and either (a) apply the same auxiliary label to all chunks of that document, or (b) only compute auxiliary loss on one representative chunk per document to avoid up-weighting long docs. (a) is easier and consistent with “one decision, one set of Sundin features”.)
- **Loss:** e.g. `total_loss = risk_loss + λ₁·downstream_loss + λ₂·upstream_loss + λ₃·flow_loss + λ₄·monitoring_loss`. Start with small λ (e.g. 0.2 each) so risk remains dominant; you can tune later.

This stays consistent with the Phase A review: keep **sliding-window** BERT for court decisions; add auxiliary losses at document/chunk level as above.

---

## 4. Alignment with Phase A review

| Phase A review point | Hybrid approach |
|----------------------|-----------------|
| Use **02** to get 50 decisions, **03** + **label_overrides.json** for 44 labels | Unchanged; still do A1–A2 first. |
| **text_full** / **decisions** / **splits** (no top-level `decisions` in labeled_dataset) | Unchanged; feature extraction and RF/clustering read from the same structures. |
| **Weak labels only in train**, val/test only strong (44) | Unchanged; clustering and RF use only the 44; multi-task and ensemble idem. |
| **Sliding window** for BERT (or explicit truncation everywhere) | Unchanged; keep sliding window for risk (+ optional auxiliary loss per doc/chunk). |
| DAPT with **datasets** + DataCollatorForLanguageModeling | Unchanged. |

So the hybrid approach **sits on top of** the corrected Phase A pipeline; it doesn’t conflict with it.

---

## 5. Suggested workflow (concise)

- **Day 8:**  
  - A1–A2: Add 4 decisions (02 → 03 + label_overrides) → 44 labeled.  
  - Extract **Sundin-informed features** (values, not only booleans) for all 44 → e.g. `decision_features_sundin2026.json`.  
  - **Clustering check:** KMeans(3) on scaled features; ARI vs current labels; list disagreements for manual review (do not auto-relabel).  
  - Optional: fix obvious label errors and re-run 03 if needed.

- **Day 9:**  
  - Weak labels for applications (Sundin-style features optional); DAPT corpus; **5-fold RF** (or regularised linear) for feature importance; report mean ± std importances.

- **Day 10:**  
  - DAPT; then **Sundin-informed multi-task** BERT (risk + auxiliary heads with labels from your Sundin extraction), sliding window unchanged; val/test on 44 only.

- **Phase B (Days 11–12):**  
  - RF(Sundin features) + BERT (single-task or multi-task) ensemble; 5-fold evaluation; report and packaging.

---

## 6. Summary

- **Principle:** “Sundin = what to look for, data = how to weight it” is **intelligent and viable**.
- **Changes to make:** (1) Stabilise feature importance (e.g. 5-fold RF or regularised model, report variance). (2) Use clustering as a **review aid** with an explicit cluster–risk mapping and no automatic relabeling. (3) Define **auxiliary labels** for the multi-task model from your Sundin feature extraction and handle missing values; keep risk evaluation on the 44 only and training protocol (sliding window) as in the Phase A review.
- **Integration:** Treat the hybrid as an **extension** of the corrected Phase A (same data, same splits, same weak-in-train rule); then add RF, clustering, and optional multi-task + ensemble on top.
