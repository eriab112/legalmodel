"""
Evaluation script for NAP Legal AI - Binary Classification.
Evaluates the binary fine-tuned model on val and test splits.
Uses document-level logit averaging with sliding window.
"""
import json
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


print("=" * 60)
print("BINARY EVALUATION: HIGH_RISK vs LOW_RISK")
print("=" * 60)

# Load model
model = AutoModelForSequenceClassification.from_pretrained("models/nap_binary/best")
tokenizer = AutoTokenizer.from_pretrained("models/nap_binary/best")
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")

print(f"Loaded binary model")

# Load binary test data
labeled = json.load(
    open("Data/processed/labeled_dataset_binary.json", encoding="utf-8")
)
label_map = {"HIGH_RISK": 0, "LOW_RISK": 1}
id2label = {0: "HIGH_RISK", 1: "LOW_RISK"}


def evaluate_split(docs, split_name):
    """Evaluate a split using sliding window + logit averaging."""
    doc_preds = {}
    doc_confidences = {}
    doc_true = {}

    for doc in docs:
        doc_id = doc["id"]
        doc_true[doc_id] = label_map[doc["label"]]
        text = doc["key_text"]

        # Sliding window
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunk_logits = []

        stride = 256
        max_len = 510

        for i in range(0, len(tokens), stride):
            chunk = tokens[i : i + max_len]
            if len(chunk) < 50:
                continue

            input_ids = (
                [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
            )
            attention_mask = [1] * len(input_ids)

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
                logits = outputs.logits[0].cpu().numpy()
                chunk_logits.append(logits)

        if chunk_logits:
            avg_logits = np.mean(chunk_logits, axis=0)
            probs = _softmax(avg_logits)
            doc_preds[doc_id] = int(np.argmax(probs))
            doc_confidences[doc_id] = float(np.max(probs))
        else:
            doc_preds[doc_id] = 1  # fallback to LOW_RISK
            doc_confidences[doc_id] = 0.0

        correct = doc_preds[doc_id] == doc_true[doc_id]
        print(
            f"  {doc_id}: {len(chunk_logits)} chunks, "
            f"pred={id2label[doc_preds[doc_id]]}, "
            f"true={id2label[doc_true[doc_id]]}, "
            f"conf={doc_confidences[doc_id]:.3f}, "
            f"{'OK' if correct else 'WRONG'}"
        )

    y_true = [doc_true[d] for d in doc_preds]
    y_pred = [doc_preds[d] for d in doc_preds]

    acc = accuracy_score(y_true, y_pred)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "split": split_name,
        "n_docs": len(docs),
        "accuracy": float(acc),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
        "f1_macro": float(f1_m),
        "per_class": {
            id2label[i]: {
                "precision": float(per_class[0][i]),
                "recall": float(per_class[1][i]),
                "f1": float(per_class[2][i]),
                "support": int(per_class[3][i]),
            }
            for i in range(2)
        },
        "confusion_matrix": cm.tolist(),
        "predictions": {
            doc_id: {
                "predicted": id2label[doc_preds[doc_id]],
                "true": id2label[doc_true[doc_id]],
                "correct": doc_preds[doc_id] == doc_true[doc_id],
                "confidence": doc_confidences[doc_id],
            }
            for doc_id in doc_preds
        },
    }


# Evaluate val and test splits
print("\n--- Validation Split ---")
val_results = evaluate_split(labeled["splits"]["val"], "val")

print("\n--- Test Split ---")
test_results = evaluate_split(labeled["splits"]["test"], "test")

# Combined evaluation
all_docs = labeled["splits"]["val"] + labeled["splits"]["test"]
print("\n--- Combined (Val + Test) ---")
combined_results = evaluate_split(all_docs, "combined")

# Print summary
print(f"\n{'='*60}")
print(f"BINARY EVALUATION RESULTS")
print(f"{'='*60}\n")

for res in [val_results, test_results, combined_results]:
    print(f"  {res['split'].upper()} ({res['n_docs']} docs):")
    print(f"    Accuracy:  {res['accuracy']*100:.1f}%")
    print(f"    F1 (weighted): {res['f1_weighted']:.4f}")
    print(f"    F1 (macro):    {res['f1_macro']:.4f}")
    for cls_name in ["HIGH_RISK", "LOW_RISK"]:
        pc = res["per_class"][cls_name]
        print(
            f"    {cls_name}: P={pc['precision']:.3f} R={pc['recall']:.3f} "
            f"F1={pc['f1']:.3f} (n={pc['support']})"
        )
    print(f"    Confusion matrix: {res['confusion_matrix']}")
    print()

# Save results
os.makedirs("evaluation_reports", exist_ok=True)

results = {
    "model": "models/nap_binary/best",
    "dataset": "Data/processed/labeled_dataset_binary.json",
    "label_map": {"HIGH_RISK": 0, "LOW_RISK": 1},
    "val": val_results,
    "test": test_results,
    "combined": combined_results,
}

with open("evaluation_reports/binary_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Results saved to: evaluation_reports/binary_results.json")
print(f"\n{'='*60}")
print(f"BINARY EVALUATION COMPLETE")
print(f"{'='*60}")
