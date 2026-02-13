"""
Evaluation script for NAP Legal AI.
Evaluates the fine-tuned model on the held-out test set.
Uses document-level majority voting (same as baseline).
"""
import json
import os
from collections import Counter, defaultdict

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print("=" * 60)
print("STEP 4: EVALUATION ON TEST SET")
print("=" * 60)

# Load model
model = AutoModelForSequenceClassification.from_pretrained("models/nap_final/best")
tokenizer = AutoTokenizer.from_pretrained("models/nap_final/best")
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")

print(f"Loaded final model")

# Load test data - we need document-level labels for fair comparison
labeled = json.load(open("Data/processed/labeled_dataset.json", encoding="utf-8"))
label_map = {"HIGH_RISK": 0, "MEDIUM_RISK": 1, "LOW_RISK": 2}
id2label = {0: "HIGH_RISK", 1: "MEDIUM_RISK", 2: "LOW_RISK"}

test_docs = labeled["splits"]["test"]
print(f"Test documents: {len(test_docs)}")

# Evaluate at document level using sliding window + majority vote
# (same approach as baseline for fair comparison)
doc_preds = {}
doc_true = {}

for doc in test_docs:
    doc_id = doc["id"]
    doc_true[doc_id] = label_map[doc["label"]]
    text = doc["key_text"]

    # Sliding window
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunk_preds = []

    stride = 256
    max_len = 510  # leave room for CLS/SEP

    for i in range(0, len(tokens), stride):
        chunk = tokens[i : i + max_len]
        if len(chunk) < 50:
            continue

        input_ids = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        # Pad
        pad_len = 512 - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len

        inputs = {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([attention_mask]),
        }

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            chunk_preds.append(pred)

    # Majority vote
    if chunk_preds:
        vote = Counter(chunk_preds).most_common(1)[0][0]
        doc_preds[doc_id] = vote
    else:
        doc_preds[doc_id] = 1  # fallback to MEDIUM_RISK

    print(
        f"  {doc_id}: {len(chunk_preds)} chunks, "
        f"pred={id2label[doc_preds[doc_id]]}, "
        f"true={id2label[doc_true[doc_id]]}, "
        f"{'OK' if doc_preds[doc_id] == doc_true[doc_id] else 'WRONG'}"
    )

# Calculate metrics
y_true = [doc_true[d] for d in doc_preds]
y_pred = [doc_preds[d] for d in doc_preds]

acc = accuracy_score(y_true, y_pred)
precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)
precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)

# Per-class
per_class = precision_recall_fscore_support(
    y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

# Baseline comparison
baseline_per_class = {
    "HIGH_RISK": {"precision": 0.40, "recall": 0.30, "f1": 0.33},
    "MEDIUM_RISK": {"precision": 0.64, "recall": 0.95, "f1": 0.76},
    "LOW_RISK": {"precision": 0.13, "recall": 0.20, "f1": 0.16},
}

print(f"\n{'='*60}")
print(f"FINAL RESULTS (Document-Level)")
print(f"{'='*60}\n")

print(f"Overall Performance:")
print(f"  Accuracy: {acc*100:.1f}%")
print(f"  Precision (weighted): {precision_w:.4f}")
print(f"  Recall (weighted): {recall_w:.4f}")
print(f"  F1 (weighted): {f1_w:.4f}")
print(f"  F1 (macro): {f1_m:.4f}")

baseline = 0.65
improvement = (acc - baseline) * 100
print(f"\nComparison to Baseline:")
print(f"  Baseline (5-fold avg): {baseline*100:.1f}%")
print(f"  Baseline (best fold 4): 75.0%")
print(f"  New model: {acc*100:.1f}%")
print(f"  Improvement vs avg: {improvement:+.1f} percentage points")

labels_names = ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"]
print(f"\nPer-Class Metrics:")
for i, name in enumerate(labels_names):
    bl = baseline_per_class[name]
    print(f"\n  {name}:")
    print(f"    Precision: {per_class[0][i]:.4f} (baseline: {bl['precision']:.2f})")
    print(f"    Recall:    {per_class[1][i]:.4f} (baseline: {bl['recall']:.2f})")
    print(f"    F1:        {per_class[2][i]:.4f} (baseline: {bl['f1']:.2f})")

print(f"\nConfusion Matrix:")
print(f"              Pred_HIGH  Pred_MED  Pred_LOW")
for i, row in enumerate(cm):
    print(f"  True_{labels_names[i][:4]:4s}:  {row[0]:6d}    {row[1]:6d}    {row[2]:6d}")

# Recommendation
print(f"\n{'='*60}")
print(f"RECOMMENDATION")
print(f"{'='*60}\n")

if acc >= 0.73:
    rec = "SUCCESS - Target achieved!"
    print(f"SUCCESS - Target achieved! ({acc*100:.1f}% >= 73%)")
    print(f"Ready for deployment testing.")
elif acc >= 0.70:
    rec = "GOOD - Improvement achieved but below target"
    print(f"GOOD - Above minimum threshold ({acc*100:.1f}% >= 70%)")
    print(f"Consider Phase B (multi-task) for further improvement.")
elif acc > baseline:
    rec = "MODEST - Some improvement"
    print(f"MODEST - Some improvement ({acc*100:.1f}% vs {baseline*100:.1f}%)")
    print(f"May need more data or Phase B.")
else:
    rec = "NO IMPROVEMENT"
    print(f"NO IMPROVEMENT ({acc*100:.1f}% vs {baseline*100:.1f}%)")
    print(f"Investigate: overfitting, data quality, hyperparameters.")

# Save results
os.makedirs("evaluation_reports", exist_ok=True)

results = {
    "recommendation": rec,
    "accuracy": float(acc),
    "precision_weighted": float(precision_w),
    "recall_weighted": float(recall_w),
    "f1_weighted": float(f1_w),
    "f1_macro": float(f1_m),
    "baseline_accuracy_avg": baseline,
    "baseline_accuracy_best_fold": 0.75,
    "improvement_vs_avg": float(improvement),
    "per_class": {
        labels_names[i]: {
            "precision": float(per_class[0][i]),
            "recall": float(per_class[1][i]),
            "f1": float(per_class[2][i]),
            "support": int(per_class[3][i]),
            "baseline_precision": baseline_per_class[labels_names[i]]["precision"],
            "baseline_recall": baseline_per_class[labels_names[i]]["recall"],
            "baseline_f1": baseline_per_class[labels_names[i]]["f1"],
        }
        for i in range(3)
    },
    "confusion_matrix": cm.tolist(),
    "doc_predictions": {
        doc_id: {
            "predicted": id2label[doc_preds[doc_id]],
            "true": id2label[doc_true[doc_id]],
            "correct": doc_preds[doc_id] == doc_true[doc_id],
        }
        for doc_id in doc_preds
    },
}

with open("evaluation_reports/final_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to: evaluation_reports/final_results.json")
print(f"\n{'='*60}")
print(f"EVALUATION COMPLETE")
print(f"{'='*60}")
