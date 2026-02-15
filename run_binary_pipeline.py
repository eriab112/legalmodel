"""
Binary Reclassification + Retraining Pipeline for NAP Legal AI.

Step 1 (default): Generate binary reclassification suggestions for all 44 decisions.
Step 2 (--step2): Create binary dataset, retrain model, and evaluate.

Usage:
    python run_binary_pipeline.py          # Step 1: generate suggestions
    python run_binary_pipeline.py --step2  # Step 3: retrain and evaluate
"""
import json
import os
import sys


# ============================================================
# STEP 1: Generate binary reclassification suggestions
# ============================================================

def load_all_decisions(labeled_path):
    """Load all 44 decisions from labeled_dataset.json, preserving split info."""
    with open(labeled_path, "r", encoding="utf-8") as f:
        labeled = json.load(f)

    decisions = []
    for split_name in ["train", "val", "test"]:
        for d in labeled["splits"][split_name]:
            d["_split"] = split_name
            decisions.append(d)
    return decisions, labeled


def classify_medium_to_binary(decision):
    """
    Apply reclassification rules (in order) to a MEDIUM_RISK decision.
    Returns (binary_label, rule_applied, needs_review).
    """
    sd = decision["scoring_details"]
    domslut = sd.get("domslut_measures", [])
    n_domslut = len(domslut)
    outcome = sd.get("outcome_desc", "")
    max_cost = sd.get("max_cost_sek", 0) or 0

    # Rule 1: 3+ domslut_measures → HIGH_RISK
    if n_domslut >= 3:
        return "HIGH_RISK", "rule1_3plus_domslut_measures", False

    # Rule 2: 1-2 domslut_measures AND outcome contains "Villkor/dom ändras" or "Tillstånd meddelas"
    if 1 <= n_domslut <= 2:
        if "Villkor/dom ändras" in outcome or "Tillstånd meddelas" in outcome:
            return "HIGH_RISK", "rule2_1to2_measures_plus_permit_or_condition_change", False

    # Rule 3: 1-2 domslut_measures AND outcome is "Kan ej avgöras automatiskt" or "Målet återförvisas"
    if 1 <= n_domslut <= 2:
        if outcome == "Kan ej avgöras automatiskt" or outcome == "Målet återförvisas":
            return "HIGH_RISK", "rule3_1to2_measures_plus_unclear_or_remanded", False

    # Rule 4: 0 measures AND outcome "Kan ej avgöras automatiskt"
    if n_domslut == 0 and outcome == "Kan ej avgöras automatiskt":
        return "LOW_RISK", "rule4_zero_measures_unclear_outcome", False

    # Rule 5: 0 measures AND outcome "Målet återförvisas"
    if n_domslut == 0 and outcome == "Målet återförvisas":
        return "LOW_RISK", "rule5_zero_measures_remanded", False

    # Rule 6: 0 measures AND outcome contains "avslås" or "Avvisning"
    if n_domslut == 0 and ("avslås" in outcome or "Avvisning" in outcome):
        return "LOW_RISK", "rule6_zero_measures_rejected", False

    # Rule 7: 0 measures AND outcome contains "Villkor/dom ändras" or "Ändrar dom" → ambiguous
    if n_domslut == 0 and ("Villkor/dom ändras" in outcome or "Ändrar dom" in outcome):
        if max_cost > 1_000_000:
            return "HIGH_RISK", "rule7_zero_measures_condition_change_high_cost", False
        else:
            return "LOW_RISK", "rule7_zero_measures_condition_change_low_cost", False

    # Rule 8: Remaining edge cases
    return "LOW_RISK", "rule8_edge_case", True


def build_evidence(decision):
    """Build evidence dict for a decision."""
    sd = decision["scoring_details"]
    return {
        "outcome": sd.get("outcome_desc", ""),
        "n_measures": len(sd.get("domslut_measures", [])),
        "measures": sd.get("domslut_measures", []),
        "all_mentioned_measures": sd.get("measures", []),
        "max_cost_sek": sd.get("max_cost_sek", 0),
        "has_fishway": any(
            m in sd.get("measures", [])
            for m in ["fiskväg", "faunapassage", "omlöp"]
        ),
        "monitoring_required": sd.get("is_relevant", False),
    }


def generate_reclassification(labeled_path):
    """Generate binary_reclassification.json with suggested labels for all 44 decisions."""
    decisions, labeled_data = load_all_decisions(labeled_path)

    results = []
    summary = {
        "total": 0,
        "kept_high": 0,
        "kept_low": 0,
        "medium_to_high": 0,
        "medium_to_low": 0,
        "needs_review": 0,
        "final_high": 0,
        "final_low": 0,
    }

    for d in decisions:
        did = d["id"]
        original_label = d["label"]
        confidence = d.get("confidence", 0.5)
        case_number = d.get("metadata", {}).get("case_number", did)

        if original_label == "HIGH_RISK":
            binary_label = "HIGH_RISK"
            rule = "kept_original"
            needs_review = False
            reclassified = False
            summary["kept_high"] += 1
        elif original_label == "LOW_RISK":
            binary_label = "LOW_RISK"
            rule = "kept_original"
            needs_review = False
            reclassified = False
            summary["kept_low"] += 1
        else:
            # MEDIUM_RISK → apply reclassification rules
            binary_label, rule, needs_review = classify_medium_to_binary(d)
            reclassified = True
            if binary_label == "HIGH_RISK":
                summary["medium_to_high"] += 1
            else:
                summary["medium_to_low"] += 1

        if needs_review:
            summary["needs_review"] += 1

        if binary_label == "HIGH_RISK":
            summary["final_high"] += 1
        else:
            summary["final_low"] += 1

        summary["total"] += 1

        evidence = build_evidence(d)

        results.append({
            "id": did,
            "case_number": case_number,
            "original_label": original_label,
            "original_confidence": confidence,
            "binary_label": binary_label,
            "reclassified": reclassified,
            "rule_applied": rule,
            "evidence": evidence,
            "needs_review": needs_review,
        })

    output = {
        "description": "Binary reclassification of all 44 NAP decisions. Review suggested labels before training.",
        "reclassification_rules": (
            "Rules applied in order to MEDIUM_RISK decisions: "
            "(1) 3+ domslut_measures → HIGH_RISK; "
            "(2) 1-2 domslut_measures + outcome Villkor/dom ändras or Tillstånd meddelas → HIGH_RISK; "
            "(3) 1-2 domslut_measures + outcome Kan ej avgöras or Målet återförvisas → HIGH_RISK; "
            "(4) 0 measures + outcome Kan ej avgöras → LOW_RISK; "
            "(5) 0 measures + outcome Målet återförvisas → LOW_RISK; "
            "(6) 0 measures + outcome contains avslås or Avvisning → LOW_RISK; "
            "(7) 0 measures + outcome Villkor/dom ändras or Ändrar dom → if cost>1M HIGH_RISK else LOW_RISK; "
            "(8) Remaining edge cases → LOW_RISK with needs_review=true."
        ),
        "decisions": results,
        "summary": summary,
    }

    return output


def print_reclassification_summary(reclass_data):
    """Print detailed summary of reclassification results."""
    summary = reclass_data["summary"]

    print("=" * 60)
    print("BINARY RECLASSIFICATION SUMMARY")
    print("=" * 60)
    print()
    print(f"Total decisions:        {summary['total']}")
    print(f"Kept HIGH_RISK:         {summary['kept_high']}")
    print(f"Kept LOW_RISK:          {summary['kept_low']}")
    print(f"MEDIUM -> HIGH_RISK:    {summary['medium_to_high']}")
    print(f"MEDIUM -> LOW_RISK:     {summary['medium_to_low']}")
    print(f"Needs manual review:    {summary['needs_review']}")
    print()
    print(f"Final HIGH_RISK:        {summary['final_high']}")
    print(f"Final LOW_RISK:         {summary['final_low']}")
    print()

    # Print all MEDIUM_RISK decisions with their suggested labels
    medium_decisions = [d for d in reclass_data["decisions"] if d["reclassified"]]

    print("-" * 80)
    print("MEDIUM_RISK RECLASSIFICATION DETAILS")
    print("-" * 80)
    print(f"{'ID':<16} {'Binary Label':<14} {'Rule':<55} {'Review?'}")
    print("-" * 80)

    for d in medium_decisions:
        review_flag = " ** REVIEW" if d["needs_review"] else ""
        print(
            f"{d['id']:<16} {d['binary_label']:<14} "
            f"{d['rule_applied']:<55} {review_flag}"
        )
        ev = d["evidence"]
        print(
            f"  -- outcome: {ev['outcome']}, "
            f"domslut_measures: {ev['n_measures']} {ev['measures']}, "
            f"cost: {ev['max_cost_sek']:,.0f} SEK"
        )

    print()


# ============================================================
# STEP 3: Create binary dataset, retrain, evaluate
# ============================================================

def create_binary_dataset(reclass_path, labeled_path, output_path):
    """Create labeled_dataset_binary.json from reclassification and original data."""
    import random

    with open(reclass_path, "r", encoding="utf-8") as f:
        reclass = json.load(f)

    with open(labeled_path, "r", encoding="utf-8") as f:
        labeled = json.load(f)

    # Build lookup from original labeled data
    original_lookup = {}
    for split_name in ["train", "val", "test"]:
        for d in labeled["splits"][split_name]:
            original_lookup[d["id"]] = d

    # Build list of all decisions with binary labels
    all_decisions = []
    for rd in reclass["decisions"]:
        did = rd["id"]
        orig = original_lookup[did]
        entry = {
            "id": did,
            "filename": orig["filename"],
            "label": rd["binary_label"],
            "original_label": rd["original_label"],
            "confidence": rd["original_confidence"],
            "key_text": orig["key_text"],
            "metadata": orig["metadata"],
            "scoring_details": orig["scoring_details"],
        }
        all_decisions.append(entry)

    # Stratified split: ~34 train, 5 val, 5 test
    random.seed(42)

    high_risk = [d for d in all_decisions if d["label"] == "HIGH_RISK"]
    low_risk = [d for d in all_decisions if d["label"] == "LOW_RISK"]

    random.shuffle(high_risk)
    random.shuffle(low_risk)

    n_high = len(high_risk)
    n_low = len(low_risk)

    # Calculate split sizes proportionally
    # Target: ~5 val, ~5 test, rest train
    # Stratified: each class contributes proportionally
    n_val_high = max(1, round(n_high * 5 / 44))
    n_test_high = max(1, round(n_high * 5 / 44))
    n_train_high = n_high - n_val_high - n_test_high

    n_val_low = max(1, round(n_low * 5 / 44))
    n_test_low = max(1, round(n_low * 5 / 44))
    n_train_low = n_low - n_val_low - n_test_low

    train_split = high_risk[:n_train_high] + low_risk[:n_train_low]
    val_split = high_risk[n_train_high:n_train_high + n_val_high] + low_risk[n_train_low:n_train_low + n_val_low]
    test_split = high_risk[n_train_high + n_val_high:] + low_risk[n_train_low + n_val_low:]

    random.shuffle(train_split)
    random.shuffle(val_split)
    random.shuffle(test_split)

    # Count labels
    label_dist = {"HIGH_RISK": n_high, "LOW_RISK": n_low}

    binary_dataset = {
        "version": "1.0-binary",
        "created": "2026-02-14",
        "total_decisions": len(all_decisions),
        "label_distribution": label_dist,
        "split_ratios": {
            "train": len(train_split) / len(all_decisions),
            "val": len(val_split) / len(all_decisions),
            "test": len(test_split) / len(all_decisions),
        },
        "split_sizes": {
            "train": len(train_split),
            "val": len(val_split),
            "test": len(test_split),
        },
        "random_seed": 42,
        "note": "Binary reclassification of original 3-class dataset. MEDIUM_RISK mapped to HIGH/LOW based on domslut_measures and outcome.",
        "splits": {
            "train": train_split,
            "val": val_split,
            "test": test_split,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(binary_dataset, f, indent=2, ensure_ascii=False)

    print(f"Binary dataset saved to: {output_path}")
    print(f"  Total: {len(all_decisions)}")
    print(f"  HIGH_RISK: {n_high}, LOW_RISK: {n_low}")
    print(f"  Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_split)}")

    # Print split distributions
    for split_name, split_data in [("train", train_split), ("val", val_split), ("test", test_split)]:
        h = sum(1 for d in split_data if d["label"] == "HIGH_RISK")
        l = sum(1 for d in split_data if d["label"] == "LOW_RISK")
        print(f"  {split_name}: HIGH_RISK={h}, LOW_RISK={l}")

    return binary_dataset


def create_finetune_binary_script(script_path):
    """Create run_finetune_binary.py for binary classification training."""
    script = '''"""
Fine-Tuning with Weighted Loss for NAP Legal AI - Binary Classification
Trains a HIGH_RISK vs LOW_RISK classifier using all 44 court decisions.
Optimized for NVIDIA T500 (4GB VRAM).
"""
import gc
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

print("=" * 60)
print("BINARY FINE-TUNING: HIGH_RISK vs LOW_RISK")
print("=" * 60)

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")


class NAPDataset(TorchDataset):
    """Dataset with sliding window for court docs - binary classification."""

    def __init__(self, samples=None, tokenizer=None, max_len=512, stride=256):
        self.samples = []
        if tokenizer is not None:
            self.cls_id = tokenizer.cls_token_id
            self.sep_id = tokenizer.sep_token_id
            self.pad_id = tokenizer.pad_token_id
        if samples is None:
            return
        label_map = {"HIGH_RISK": 0, "LOW_RISK": 1}
        for s in samples:
            label = label_map[s["label"]]
            tokens = tokenizer.encode(s["text"], add_special_tokens=False)
            for i in range(0, len(tokens), stride):
                chunk = tokens[i : i + max_len]
                if len(chunk) >= 50:
                    self.samples.append(
                        {
                            "tokens": chunk,
                            "label": label,
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        input_ids = [self.cls_id] + s["tokens"] + [self.sep_id]
        input_ids = input_ids[:512]
        pad_len = 512 - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [self.pad_id] * pad_len
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(s["label"]),
        }


class WeightedTrainer(Trainer):
    """Custom trainer with class weights for imbalanced binary dataset."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = F.cross_entropy(
            logits.view(-1, 2), labels.view(-1), reduction="none"
        )

        # Apply class weights
        if self.class_weights is not None:
            cw = self.class_weights.to(loss.device)
            class_w = cw[labels.view(-1)]
            loss = loss * class_w

        weighted_loss = loss.mean()
        return (weighted_loss, outputs) if return_outputs else weighted_loss


# Load DAPT model for binary sequence classification
dapt_path = "models/nap_dapt/final"
print(f"Loading DAPT model from {dapt_path}...")
model = AutoModelForSequenceClassification.from_pretrained(dapt_path, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(dapt_path)

# Set label mapping
model.config.id2label = {0: "HIGH_RISK", 1: "LOW_RISK"}
model.config.label2id = {"HIGH_RISK": 0, "LOW_RISK": 1}

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Enable gradient checkpointing to reduce VRAM usage
model.gradient_checkpointing_enable()
print("Gradient checkpointing: ENABLED (saves ~1GB VRAM)")

# Load binary dataset - only 44 strong-labeled court decisions
print("Building datasets from binary labeled data...")

labeled = json.load(
    open("Data/processed/labeled_dataset_binary.json", encoding="utf-8")
)

train_samples = [
    {"text": d["key_text"], "label": d["label"]}
    for d in labeled["splits"]["train"]
]

train_ds = NAPDataset(train_samples, tokenizer)

# Compute class weights from training chunk distribution
from collections import Counter

label_counts = Counter(s["label"] for s in train_ds.samples)
total = sum(label_counts.values())
n_classes = 2
class_weights = torch.tensor(
    [total / (n_classes * label_counts.get(i, 1)) for i in range(n_classes)],
    dtype=torch.float,
)
print(f"Loaded: {len(train_ds)} train chunks")
print(
    f"Train label distribution: HIGH_RISK={label_counts.get(0, 0)}, "
    f"LOW_RISK={label_counts.get(1, 0)}"
)
print(
    f"Class weights: HIGH_RISK={class_weights[0]:.2f}, "
    f"LOW_RISK={class_weights[1]:.2f}"
)

# Create output directories
os.makedirs("models/nap_binary", exist_ok=True)
os.makedirs("logs/finetune_binary", exist_ok=True)

# Clear GPU cache before training
torch.cuda.empty_cache()
gc.collect()

# Training arguments - ultra-conservative for 4GB VRAM
args = TrainingArguments(
    output_dir="models/nap_binary",
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_steps=25,
    logging_dir="logs/finetune_binary",
    eval_strategy="no",
    save_strategy="no",
    report_to="none",
    dataloader_num_workers=0,
    max_grad_norm=1.0,
    remove_unused_columns=False,
)

print(f"\\nTraining Configuration:")
print(f"  Epochs: 4")
print(f"  Batch size: 1 (per device)")
print(f"  Gradient accumulation: 8 steps")
print(f"  Effective batch size: 8")
print(f"  Learning rate: 2e-5")
print(f"  Class-weighted loss: Yes")
print(f"  FP16: {torch.cuda.is_available()}")
print(f"  Gradient checkpointing: True")
print(f"  Eval during training: DISABLED (saves memory)")
print(f"  Checkpoint saving: DISABLED (saves memory)")

trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=args,
    train_dataset=train_ds,
)

print(f"\\nStarting binary fine-tuning...")
start_time = time.time()

trainer.train()

elapsed = time.time() - start_time
print(f"\\nFine-tuning complete in {elapsed/60:.1f} minutes")

# Clear cache before saving
torch.cuda.empty_cache()
gc.collect()

# Save final model
os.makedirs("models/nap_binary/best", exist_ok=True)
model.save_pretrained("models/nap_binary/best")
tokenizer.save_pretrained("models/nap_binary/best")

print(f"Model saved to: models/nap_binary/best/")
print("\\n" + "=" * 60)
print("BINARY FINE-TUNING COMPLETE")
print("=" * 60)
'''
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"Created: {script_path}")


def create_evaluate_binary_script(script_path):
    """Create run_evaluate_binary.py for binary classification evaluation."""
    script = '''"""
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
print("\\n--- Validation Split ---")
val_results = evaluate_split(labeled["splits"]["val"], "val")

print("\\n--- Test Split ---")
test_results = evaluate_split(labeled["splits"]["test"], "test")

# Combined evaluation
all_docs = labeled["splits"]["val"] + labeled["splits"]["test"]
print("\\n--- Combined (Val + Test) ---")
combined_results = evaluate_split(all_docs, "combined")

# Print summary
print(f"\\n{'='*60}")
print(f"BINARY EVALUATION RESULTS")
print(f"{'='*60}\\n")

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
print(f"\\n{'='*60}")
print(f"BINARY EVALUATION COMPLETE")
print(f"{'='*60}")
'''
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"Created: {script_path}")


def run_step2():
    """Step 3: Create binary dataset, generate scripts, retrain, evaluate."""
    import subprocess

    reclass_path = "Data/processed/binary_reclassification.json"
    labeled_path = "Data/processed/labeled_dataset.json"
    binary_path = "Data/processed/labeled_dataset_binary.json"

    if not os.path.exists(reclass_path):
        print(f"ERROR: {reclass_path} not found. Run Step 1 first.")
        sys.exit(1)

    print("=" * 60)
    print("STEP 3: CREATE BINARY DATASET + RETRAIN + EVALUATE")
    print("=" * 60)

    # a) Create binary dataset
    print("\n--- (a) Creating binary dataset ---")
    create_binary_dataset(reclass_path, labeled_path, binary_path)

    # b) Create run_finetune_binary.py
    print("\n--- (b) Creating run_finetune_binary.py ---")
    create_finetune_binary_script("run_finetune_binary.py")

    # c) Create run_evaluate_binary.py
    print("\n--- (c) Creating run_evaluate_binary.py ---")
    create_evaluate_binary_script("run_evaluate_binary.py")

    # d) Run training
    print("\n--- (d) Running binary fine-tuning ---")
    result = subprocess.run(
        [sys.executable, "run_finetune_binary.py"],
        cwd=os.getcwd(),
    )
    if result.returncode != 0:
        print(f"ERROR: Training failed with return code {result.returncode}")
        sys.exit(1)

    # e) Run evaluation
    print("\n--- (e) Running binary evaluation ---")
    result = subprocess.run(
        [sys.executable, "run_evaluate_binary.py"],
        cwd=os.getcwd(),
    )
    if result.returncode != 0:
        print(f"ERROR: Evaluation failed with return code {result.returncode}")
        sys.exit(1)

    # f) Report results
    print("\n--- (f) Final Results ---")
    if os.path.exists("evaluation_reports/binary_results.json"):
        with open("evaluation_reports/binary_results.json", "r", encoding="utf-8") as f:
            results = json.load(f)

        combined = results.get("combined", {})
        print(f"\nBinary Classification Results (Combined Val+Test):")
        print(f"  Accuracy:     {combined.get('accuracy', 0)*100:.1f}%")
        print(f"  F1 (weighted): {combined.get('f1_weighted', 0):.4f}")
        print(f"  F1 (macro):    {combined.get('f1_macro', 0):.4f}")

        for cls_name in ["HIGH_RISK", "LOW_RISK"]:
            pc = combined.get("per_class", {}).get(cls_name, {})
            print(
                f"  {cls_name}: P={pc.get('precision', 0):.3f} "
                f"R={pc.get('recall', 0):.3f} F1={pc.get('f1', 0):.3f}"
            )

    print("\n" + "=" * 60)
    print("BINARY PIPELINE COMPLETE")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if "--step2" in sys.argv:
        run_step2()
    else:
        # Step 1: Generate reclassification suggestions
        labeled_path = "Data/processed/labeled_dataset.json"
        output_path = "Data/processed/binary_reclassification.json"

        print("=" * 60)
        print("STEP 1: BINARY RECLASSIFICATION SUGGESTIONS")
        print("=" * 60)
        print()

        reclass_data = generate_reclassification(labeled_path)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(reclass_data, f, indent=2, ensure_ascii=False)

        print(f"Reclassification saved to: {output_path}")
        print()

        print_reclassification_summary(reclass_data)

        # Step 2: Print review message
        print()
        print("=" * 42)
        print("REVIEW REQUIRED")
        print("=" * 42)
        print(f"Please review {output_path}")
        print("Edit any binary_label values you disagree with.")
        print("When done, save the file and re-run this script with --step2 flag.")
        print()
        print("To proceed immediately with current suggestions, run:")
        print("  python run_binary_pipeline.py --step2")
        print("=" * 42)
