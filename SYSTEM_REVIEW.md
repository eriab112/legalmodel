# NAP Legal AI Advisor - System Review

**Date**: 2026-02-10
**Scope**: Full review of Task 4 deliverable + project state assessment
**Purpose**: Basis for discussion on improvements and future directions

---

## 1. What Has Been Done (Tasks 1-4)

### Task 1: Data Exploration
- Inventoried 54 raw court decision files, identified 46 unique decisions (952K words)
- Mapped all data sources: court texts, MCDA rankings, NER data, VISS water bodies
- Produced `EXPLORATION_REPORT.md` with feasibility assessment
- Key finding: RCRS proxy poorly discriminating (75% of 15,688 water bodies score 60-65)

### Task 2: Preprocessing & Labeling
- **Scripts**: `02_clean_court_texts.py`, `03_create_labeled_dataset.py`, `04_link_to_viss.py`
- Cleaned OCR artifacts, segmented into legal sections (domslut, domskal, yrkanden, etc.)
- Labeled 40 decisions: 8 HIGH_RISK (20%), 23 MEDIUM_RISK (58%), 9 LOW_RISK (22%)
- Linked 16/17 water bodies to VISS IDs (94% coverage)
- Labeling approach: domslut outcome classification (not keyword counting)

### Task 3: Fine-tuning KB-BERT
- **Script**: `05_finetune_legalbert.py` (614 lines)
- 5-fold stratified cross-validation on `KB/bert-base-swedish-cased`
- Sliding window: 512 tokens, stride 256 (895 total chunks from 40 docs)
- Class-weighted loss to handle imbalance
- Training time: ~5 hours on GPU
- **Results**: 65% avg accuracy, fold_4 best at 75% accuracy / 0.739 F1

### Task 4: NAP Legal AI Advisor (This Deliverable)
- **16 files, 1,811 lines** of production Streamlit code
- Dual-mode interface: Chat (Q&A) + Search (semantic search)
- LegalBERT inference with sliding window matching training code exactly
- Semantic search: 3,187 chunks, MiniLM multilingual, 12ms query time
- Template-based RAG responses (no external LLM API dependency)

---

## 2. Current Architecture

```
nap-legal-ai-advisor/ (1,811 lines)
├── app.py                          # Entry point, mode toggle, sidebar
├── backend/
│   ├── search_engine.py            # MiniLM + numpy cosine similarity
│   ├── risk_predictor.py           # LegalBERT fold_4 inference
│   └── rag_system.py               # Intent detection + template responses
├── integration/
│   ├── shared_context.py           # Session state management
│   ├── chat_handler.py             # Chat routing + quick actions
│   └── search_handler.py           # Search orchestration + filters
├── ui/
│   ├── chat_interface.py           # Chat mode UI
│   ├── search_interface.py         # Search mode + detail view
│   └── styles.py                   # CSS theming
└── utils/
    ├── data_loader.py              # Merges 3 JSON data sources
    └── ssl_fix.py                  # Corporate proxy workaround
```

### Data Flow
1. `DataLoader` merges `labeled_dataset.json` + `cleaned_court_texts.json` + `linkage_table.json` into `DecisionRecord` objects
2. `SemanticSearchEngine` chunks key_text (~500 chars, 100 overlap), encodes with MiniLM, stores as numpy array
3. `LegalBERTPredictor` uses sliding window (512 tokens, stride 256) matching training code exactly
4. `RAGSystem` detects intent via keywords, routes to template-formatted responses
5. UI renders via Streamlit tabs (Chat / Search)

---

## 3. Verified Test Results

| Component | Test | Result |
|-----------|------|--------|
| DataLoader | Load all sources | 46 decisions, 40 labeled, correct distribution |
| SearchEngine | Index build | 3,187 chunks in ~45s (first run), cached after |
| SearchEngine | Query speed | 12ms per query |
| SearchEngine | Relevance | "faunapassage" returns relevant cases (M 1849-22, M 5221-21) |
| SearchEngine | Filters | Label filter correctly limits results |
| RiskPredictor | MEDIUM_RISK case | m605-24: predicted MEDIUM_RISK, 90.4% confidence (correct) |
| RiskPredictor | HIGH_RISK case | m3753-22: predicted HIGH_RISK, 97.3% confidence (correct) |
| RAGSystem | Risk query | Returns 8 HIGH_RISK decisions with metadata |
| RAGSystem | Comparison | Side-by-side table for two cases |
| RAGSystem | Search fallback | Returns ranked semantic search results |
| Streamlit | App launch | Starts without errors at localhost:8501 |

---

## 4. Known Limitations & Issues

### 4.1 Model Performance
- **65% average accuracy** across 5 folds - above baselines but limited
- **Strong MEDIUM_RISK bias**: 95% recall for MEDIUM_RISK but poor for HIGH/LOW (30% and 20% recall respectively)
- **Fold variance is high**: accuracy ranges from 50% (fold_3) to 75% (fold_4)
- Only fold_4 learned to distinguish all 3 classes; other folds mostly predict MEDIUM_RISK
- Using only fold_4 (the best) rather than an ensemble could overfit to that fold's validation split

### 4.2 Dataset Size
- **40 labeled decisions is very small** for fine-tuning a 110M parameter model
- 8 HIGH_RISK and 9 LOW_RISK samples provide minimal signal for minority classes
- No external validation set - all evaluation is within-sample CV
- Labeling was semi-automated with manual overrides (7 total) - potential for inconsistency

### 4.3 Chat System
- **Keyword-based intent detection** is brittle - relies on exact keyword matches
- No conversation memory across turns (each message is independent)
- Template responses are static - no ability to synthesize or reason across multiple documents
- No Swedish NLP for intent classification (no lemmatization, no synonym expansion)
- Quick actions only appear on first message (disappear after first interaction)

### 4.4 Search System
- **No result re-ranking** - pure cosine similarity without learned relevance
- Chunks are character-based (500 chars) which can split mid-sentence
- No query expansion or synonym handling for Swedish legal terminology
- Filter UI in sidebar is disconnected from the search button (requires re-search)

### 4.5 UI/UX
- No loading skeleton or progressive rendering during first startup (~45s)
- Decision detail view text areas are read-only `st.text_area` (not ideal for reading)
- No export functionality (PDF report, CSV of results)
- Mode switching via tabs doesn't preserve scroll position
- Swedish text in UI uses ASCII approximations (e.g., "atgarder" instead of "atgarder")

### 4.6 Deployment
- Requires ~2.4 GB of model files (5 folds, only fold_4 used)
- First startup downloads MiniLM model from HuggingFace (~100MB)
- SSL workaround needed for corporate proxy (not production-ready)
- No authentication, no rate limiting, no logging

---

## 5. What Could Be Improved (Short-term)

### 5.1 Model Improvements
- **Ensemble prediction**: Average predictions across all 5 folds instead of only fold_4
  - Each fold saw different validation data, so ensembling reduces variance
  - Simple implementation: load all 5 models, average logits
  - Expected benefit: more robust predictions, reduced fold_4 overfitting

- **Binary classification**: Collapse to HIGH_RISK vs NOT_HIGH_RISK
  - With 40 samples, 3-class is ambitious; binary would have 8 vs 32 samples
  - More actionable for decision-makers (flag high-risk cases)
  - Expected accuracy improvement: 10-15 percentage points

- **Threshold calibration**: Instead of argmax, calibrate probability thresholds
  - E.g., flag as HIGH_RISK if P(HIGH_RISK) > 0.3 (rather than requiring it to be the argmax)
  - Better for a decision support system where false negatives are costlier

### 5.2 Chat System Improvements
- **LLM-powered responses**: Integrate Claude API or local LLM for natural language generation
  - Currently template-based - limits expressiveness significantly
  - With RAG context, an LLM could synthesize, compare, and reason across documents
  - Cost consideration: ~$0.01-0.05 per query with Claude Haiku

- **Swedish intent classifier**: Train a small intent model on example queries
  - Or use a multilingual NLI model (e.g., `joeddav/xlm-roberta-large-xnli`) for zero-shot
  - Would handle synonyms, misspellings, varied phrasing

- **Conversation memory**: Track conversation context for follow-up questions
  - "Show high risk decisions" -> "Tell me more about the first one" should work

### 5.3 Search Improvements
- **Cross-encoder re-ranking**: Use a cross-encoder to re-rank top-20 results
  - Sentence-transformers bi-encoders are fast but less accurate for ranking
  - Cross-encoders score (query, document) pairs jointly for better relevance
  - Only needed for top-k, so speed impact is minimal

- **Section-aware search**: Weight domslut and domskal sections higher
  - Currently searches key_text uniformly
  - Domslut is most informative for risk; domskal for reasoning

- **Faceted search**: Add measure-based filtering (e.g., "only faunapassage cases")

### 5.4 UI Improvements
- **Progressive loading**: Show chat interface immediately while models load in background
- **Decision comparison view**: Side-by-side visual comparison (not just table)
- **Risk dashboard tab**: Aggregate statistics with plotly charts (distribution, trends over time)
- **Export**: Generate PDF analysis reports for individual decisions
- **Fix Swedish characters**: Use proper UTF-8 throughout UI text

---

## 6. Ways Forward - Designing a Great Decision Support System

### 6.1 Vision: From Demo to Decision Support

The current system is a **demo-grade prototype**. To become a genuine decision support system for NAP prioritization, it needs to evolve along three axes:

#### Axis 1: Better Predictions
| Approach | Effort | Expected Impact |
|----------|--------|----------------|
| Ensemble all 5 folds | Low (days) | Reduce prediction variance |
| Binary HIGH vs NOT_HIGH | Low (days) | Better accuracy, more actionable |
| More training data (100+ decisions) | High (months) | Significant model improvement |
| Legal expert labeling review | Medium (weeks) | Better label quality |
| Domain-adapted pretraining (Swedish legal corpus) | High (months) | Better feature representation |
| Multi-task learning (risk + cost + measures) | Medium (weeks) | Shared representations |

#### Axis 2: Better Intelligence
| Approach | Effort | Expected Impact |
|----------|--------|----------------|
| LLM integration (Claude/GPT) for response generation | Medium (days) | Natural language answers, reasoning |
| Cross-document analysis (trends, patterns) | Medium (weeks) | Strategic insights |
| Automated case summarization | Medium (weeks) | Faster decision review |
| Predictive cost modeling (integrate existing R²=0.885 model) | Low (days) | Cost estimates per case |
| Legal precedent chain detection | High (weeks) | Identify related rulings |
| Risk factor explanation (SHAP/attention) | Medium (weeks) | Interpretable predictions |

#### Axis 3: Better Integration
| Approach | Effort | Expected Impact |
|----------|--------|----------------|
| VISS/MCDA integration in UI (map view) | Medium (weeks) | Geographic context |
| Connect to existing NAP dashboard | Low (days) | Unified platform |
| Real-time court decision ingestion | High (months) | Always up-to-date |
| REST API for programmatic access | Medium (weeks) | Integration with other tools |
| Multi-user auth + audit logging | Medium (weeks) | Production readiness |

### 6.2 Recommended Priority Path

**Phase A - Quick Wins (1-2 weeks)**
1. Ensemble all 5 folds for prediction (reduces variance, no retraining)
2. Add risk dashboard tab with plotly charts (distribution, timeline, court comparison)
3. Fix Swedish character encoding in all UI strings
4. Add decision export (markdown summary to clipboard)
5. Integrate existing cost prediction model from `nap_model-main`

**Phase B - Intelligence Upgrade (2-4 weeks)**
1. Integrate Claude API for response generation (replaces template system)
2. Add conversation memory (follow-up questions)
3. Implement cross-encoder re-ranking for search
4. Add SHAP-based risk factor explanation ("why did the model predict HIGH_RISK?")
5. Section-aware search with configurable weighting

**Phase C - Production Path (1-2 months)**
1. Acquire 50-100 more labeled court decisions
2. Retrain with expanded dataset + domain-adapted pretraining
3. Add VISS map visualization (geographic decision explorer)
4. REST API + authentication
5. Connect to live court decision feeds (if available)

### 6.3 Alternative Architecture: LLM-First Approach

Instead of improving the current BERT-based classifier, consider an alternative architecture:

```
User Query
    ↓
LLM (Claude/GPT-4) with system prompt containing:
    - All 40 labeled decisions as few-shot examples
    - Risk labeling criteria
    - Legal domain knowledge
    ↓
RAG retrieves top-5 relevant chunks
    ↓
LLM generates:
    - Risk assessment with reasoning
    - Comparison to similar cases
    - Recommended measures
    - Confidence level
```

**Advantages**:
- No fine-tuning needed (works immediately)
- Better reasoning and explanation
- Handles novel queries naturally
- Can incorporate new decisions without retraining

**Disadvantages**:
- API cost (~$0.05-0.50 per complex query)
- Latency (2-10 seconds per response)
- Less reproducible (LLM outputs vary)
- Requires internet connection

**Hybrid approach**: Use LegalBERT for fast screening/classification, LLM for detailed analysis and explanation. This combines the speed/consistency of BERT with the reasoning/flexibility of LLMs.

---

## 7. Project Metrics Summary

| Metric | Value |
|--------|-------|
| Total Python code | ~5,500 lines |
| Court decisions analyzed | 46 unique |
| Labeled training data | 40 decisions |
| Model accuracy (best fold) | 75.0% |
| Model accuracy (average) | 65.0% |
| Search index | 3,187 chunks |
| Search latency | 12ms |
| Prediction latency | ~5s per decision (CPU) |
| First startup time | ~45s |
| Cached startup time | <2s |
| VISS linkage coverage | 94% (16/17) |
| Total model storage | 2.4 GB (5 folds) |

---

## 8. Key Risks for Thesis Defense

1. **65% accuracy may be questioned** - Needs strong framing as proof-of-concept with small dataset. The 13pp improvement over majority baseline and 32pp over random should be emphasized.

2. **Class imbalance narrative** - The model essentially learned to predict MEDIUM_RISK well. The thesis should discuss this honestly and present the binary classification alternative.

3. **Generalizability** - Only tested on 40 Swedish hydropower decisions from 2021-2025. Cannot claim the system generalizes to other legal domains or time periods.

4. **No external validation** - All metrics are from cross-validation on the same 40 documents. Ideally would need held-out decisions from a different time period.

5. **Template responses vs. true AI** - The chat system uses keyword matching and templates, not genuine NLU. This should be positioned as a deliberate design choice (no API dependency, reproducibility, speed) with the LLM integration discussed as future work.
