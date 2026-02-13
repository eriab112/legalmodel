#!/usr/bin/env python3
"""
Task 3: Fine-tune KB-BERT for Legal Risk Classification

Fine-tunes KB/bert-base-swedish-cased on Swedish court decisions for
3-class risk prediction (HIGH_RISK / MEDIUM_RISK / LOW_RISK) using:
- Sliding window (512 tokens, stride 256) to use full documents
- 5-fold stratified cross-validation
- Class-weighted loss to handle imbalanced labels (8H/23M/9L)
- Document-level evaluation (aggregate chunk predictions)

Outputs:
- models/nap_legalbert_cv/fold_N/  (best checkpoint per fold)
- models/nap_legalbert_cv/training_metrics.json
- models/nap_legalbert_cv/confusion_matrices.png
"""

import json
import os
import ssl
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

# SSL workaround for corporate proxy
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
ssl._create_default_https_context = ssl._create_unverified_context

import httpx
_orig_client_init = httpx.Client.__init__
def _patched_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _orig_client_init(self, *args, **kwargs)
httpx.Client.__init__ = _patched_init

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# === Constants ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "processed" / "labeled_dataset.json"
OUTPUT_DIR = BASE_DIR / "models" / "nap_legalbert_cv"
MODEL_NAME = "KB/bert-base-swedish-cased"

LABEL_MAP = {"HIGH_RISK": 0, "MEDIUM_RISK": 1, "LOW_RISK": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = 3

CHUNK_SIZE = 512
STRIDE = 256
N_FOLDS = 5
SEED = 42

# Hyperparameters
LEARNING_RATE = 2e-5
EPOCHS = 5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
GRAD_ACCUM = 8
PER_DEVICE_BATCH = 1


# === Dataset ===
class ChunkDataset(torch.utils.data.Dataset):
    """Dataset of sliding-window chunks from court decision documents."""

    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        c = self.chunks[idx]
        return {
            "input_ids": torch.tensor(c["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(c["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(c["label"], dtype=torch.long),
        }


# === Weighted Trainer ===
class WeightedTrainer(Trainer):
    """Trainer with class-weighted CrossEntropyLoss."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# === Helpers ===
def load_documents():
    """Load all 40 labeled documents from the dataset splits."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for split_name in ["train", "val", "test"]:
        for entry in data["splits"][split_name]:
            docs.append({
                "id": entry["id"],
                "text": entry["key_text"],
                "label": LABEL_MAP[entry["label"]],
                "label_name": entry["label"],
            })

    print(f"Loaded {len(docs)} documents")
    label_counts = defaultdict(int)
    for d in docs:
        label_counts[d["label_name"]] += 1
    for lbl in ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"]:
        print(f"  {lbl}: {label_counts[lbl]}")
    return docs


def create_chunks(docs, tokenizer):
    """Tokenize documents and split into overlapping 512-token chunks.

    Returns list of chunks, each with input_ids, attention_mask, label, doc_id.
    """
    all_chunks = []

    for doc in docs:
        encoded = tokenizer(
            doc["text"],
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        token_ids = encoded["input_ids"]

        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        max_content = CHUNK_SIZE - 2  # reserve [CLS] and [SEP]

        if len(token_ids) <= max_content:
            # Single chunk - pad or just use as-is
            chunk_ids = [cls_id] + token_ids + [sep_id]
            attn_mask = [1] * len(chunk_ids)
            # Pad to CHUNK_SIZE
            pad_len = CHUNK_SIZE - len(chunk_ids)
            chunk_ids += [tokenizer.pad_token_id] * pad_len
            attn_mask += [0] * pad_len

            all_chunks.append({
                "input_ids": chunk_ids,
                "attention_mask": attn_mask,
                "label": doc["label"],
                "doc_id": doc["id"],
            })
        else:
            # Sliding window
            content_stride = STRIDE - 2 if STRIDE > 2 else STRIDE
            # Use stride on content tokens (without special tokens)
            content_stride = STRIDE
            start = 0
            while start < len(token_ids):
                end = min(start + max_content, len(token_ids))
                content = token_ids[start:end]

                chunk_ids = [cls_id] + content + [sep_id]
                attn_mask = [1] * len(chunk_ids)
                # Pad if last chunk is shorter
                pad_len = CHUNK_SIZE - len(chunk_ids)
                if pad_len > 0:
                    chunk_ids += [tokenizer.pad_token_id] * pad_len
                    attn_mask += [0] * pad_len

                all_chunks.append({
                    "input_ids": chunk_ids,
                    "attention_mask": attn_mask,
                    "label": doc["label"],
                    "doc_id": doc["id"],
                })

                if end >= len(token_ids):
                    break
                start += content_stride

    return all_chunks


def compute_class_weights(labels):
    """Compute balanced class weights for imbalanced dataset."""
    classes = np.array(sorted(set(labels)))
    weights = compute_class_weight("balanced", classes=classes, y=np.array(labels))
    print(f"Class weights: {dict(zip([ID2LABEL[c] for c in classes], weights.round(3)))}")
    return weights.tolist()


def aggregate_doc_predictions(chunk_preds, chunk_doc_ids, chunk_labels):
    """Aggregate chunk-level logits to document-level predictions.

    For each document, average the logits across all its chunks,
    then argmax for the final prediction.
    """
    doc_logits = defaultdict(list)
    doc_true = {}

    for logits, doc_id, label in zip(chunk_preds, chunk_doc_ids, chunk_labels):
        doc_logits[doc_id].append(logits)
        doc_true[doc_id] = label

    doc_ids_ordered = sorted(doc_logits.keys())
    doc_preds = []
    doc_labels = []

    for doc_id in doc_ids_ordered:
        avg_logits = np.mean(doc_logits[doc_id], axis=0)
        doc_preds.append(np.argmax(avg_logits))
        doc_labels.append(doc_true[doc_id])

    return doc_ids_ordered, np.array(doc_preds), np.array(doc_labels)


def evaluate_fold(trainer, val_dataset, val_doc_ids, val_labels):
    """Evaluate a fold at both chunk and document level."""
    predictions = trainer.predict(val_dataset)
    chunk_logits = predictions.predictions  # (n_chunks, 3)
    chunk_preds = np.argmax(chunk_logits, axis=1)
    chunk_labels = predictions.label_ids

    # Chunk-level metrics
    chunk_acc = accuracy_score(chunk_labels, chunk_preds)
    chunk_f1 = f1_score(chunk_labels, chunk_preds, average="macro", zero_division=0)

    # Document-level metrics
    doc_ids, doc_preds, doc_labels = aggregate_doc_predictions(
        chunk_logits, val_doc_ids, val_labels
    )
    doc_acc = accuracy_score(doc_labels, doc_preds)
    doc_f1 = f1_score(doc_labels, doc_preds, average="macro", zero_division=0)
    doc_precision = precision_score(doc_labels, doc_preds, average="macro", zero_division=0)
    doc_recall = recall_score(doc_labels, doc_preds, average="macro", zero_division=0)
    doc_cm = confusion_matrix(doc_labels, doc_preds, labels=[0, 1, 2])

    report = classification_report(
        doc_labels, doc_preds,
        labels=[0, 1, 2],
        target_names=["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"],
        zero_division=0,
        output_dict=True,
    )

    return {
        "chunk_accuracy": float(chunk_acc),
        "chunk_f1_macro": float(chunk_f1),
        "doc_accuracy": float(doc_acc),
        "doc_f1_macro": float(doc_f1),
        "doc_precision_macro": float(doc_precision),
        "doc_recall_macro": float(doc_recall),
        "doc_confusion_matrix": doc_cm.tolist(),
        "doc_classification_report": report,
        "doc_predictions": {
            doc_id: {"pred": ID2LABEL[int(p)], "true": ID2LABEL[int(t)]}
            for doc_id, p, t in zip(doc_ids, doc_preds, doc_labels)
        },
        "n_val_docs": len(doc_ids),
        "n_val_chunks": len(chunk_preds),
    }


def plot_confusion_matrices(fold_results, output_path):
    """Plot per-fold and aggregate confusion matrices."""
    n_folds = len(fold_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    labels = ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"]

    agg_cm = np.zeros((3, 3), dtype=int)

    for i, result in enumerate(fold_results):
        cm = np.array(result["doc_confusion_matrix"])
        agg_cm += cm

        # Normalize for display
        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums

        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            ax=axes[i], vmin=0, vmax=1,
        )
        axes[i].set_title(f"Fold {i} (acc={result['doc_accuracy']:.2f})")
        axes[i].set_ylabel("True")
        axes[i].set_xlabel("Predicted")

    # Aggregate
    agg_norm = agg_cm.astype(float)
    row_sums = agg_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    agg_norm = agg_norm / row_sums

    sns.heatmap(
        agg_norm, annot=True, fmt=".2f", cmap="Oranges",
        xticklabels=labels, yticklabels=labels,
        ax=axes[n_folds], vmin=0, vmax=1,
    )
    total_correct = np.trace(agg_cm)
    total = agg_cm.sum()
    axes[n_folds].set_title(f"Aggregate ({total_correct}/{total} = {total_correct/total:.2f})")
    axes[n_folds].set_ylabel("True")
    axes[n_folds].set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices to {output_path}")


def main():
    print("=" * 60)
    print("Task 3: Fine-tune KB-BERT for Legal Risk Classification")
    print("=" * 60)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: {device} ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        print(f"Device: {device}")

    # Create output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n--- Loading data ---")
    docs = load_documents()
    doc_labels = np.array([d["label"] for d in docs])
    doc_ids = [d["id"] for d in docs]

    # Compute class weights from all docs
    class_weights = compute_class_weights(doc_labels.tolist())

    # Load tokenizer (once, reused across folds)
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create chunks for all documents (to report stats)
    all_chunks = create_chunks(docs, tokenizer)
    chunks_per_doc = defaultdict(int)
    for c in all_chunks:
        chunks_per_doc[c["doc_id"]] += 1
    avg_chunks = np.mean(list(chunks_per_doc.values()))
    print(f"Total chunks: {len(all_chunks)} from {len(docs)} docs (avg {avg_chunks:.1f} chunks/doc)")

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    all_train_losses = []

    print(f"\n{'=' * 60}")
    print(f"Starting {N_FOLDS}-fold stratified cross-validation")
    print(f"{'=' * 60}")

    total_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(doc_ids, doc_labels)):
        fold_start = time.time()
        print(f"\n--- Fold {fold_idx} ---")

        # Split documents
        train_docs = [docs[i] for i in train_idx]
        val_docs = [docs[i] for i in val_idx]

        train_labels = [d["label_name"] for d in train_docs]
        val_labels = [d["label_name"] for d in val_docs]
        print(f"  Train: {len(train_docs)} docs ({', '.join(f'{l}: {train_labels.count(l)}' for l in ['HIGH_RISK','MEDIUM_RISK','LOW_RISK'])})")
        print(f"  Val:   {len(val_docs)} docs ({', '.join(f'{l}: {val_labels.count(l)}' for l in ['HIGH_RISK','MEDIUM_RISK','LOW_RISK'])})")

        # Create chunks
        train_chunks = create_chunks(train_docs, tokenizer)
        val_chunks = create_chunks(val_docs, tokenizer)
        print(f"  Train chunks: {len(train_chunks)}, Val chunks: {len(val_chunks)}")

        # Track doc_ids for val chunks (for document-level aggregation)
        val_doc_ids = [c["doc_id"] for c in val_chunks]
        val_chunk_labels = [c["label"] for c in val_chunks]

        # Create datasets
        train_dataset = ChunkDataset(train_chunks)
        val_dataset = ChunkDataset(val_chunks)

        # Load fresh model for each fold
        print(f"  Loading fresh model: {MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL_MAP,
        )

        # Enable gradient checkpointing for memory savings
        model.gradient_checkpointing_enable()

        fold_output = OUTPUT_DIR / f"fold_{fold_idx}"

        training_args = TrainingArguments(
            output_dir=str(fold_output),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=PER_DEVICE_BATCH,
            per_device_eval_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            fp16=(device == "cuda"),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            logging_steps=25,
            report_to="none",
            seed=SEED,
            dataloader_pin_memory=False,
        )

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # Train
        print(f"  Training...")
        try:
            train_result = trainer.train()
            train_loss = train_result.training_loss
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  GPU OOM! Falling back to CPU...")
                torch.cuda.empty_cache()
                model = model.cpu()
                training_args.fp16 = False
                training_args.output_dir = str(fold_output)

                trainer = WeightedTrainer(
                    class_weights=class_weights,
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
                )
                train_result = trainer.train()
                train_loss = train_result.training_loss
            else:
                raise

        all_train_losses.append(train_loss)
        print(f"  Train loss: {train_loss:.4f}")

        # Evaluate
        print(f"  Evaluating...")
        fold_metrics = evaluate_fold(
            trainer, val_dataset, val_doc_ids, val_chunk_labels,
        )
        fold_metrics["train_loss"] = float(train_loss)
        fold_metrics["fold"] = fold_idx

        print(f"  Doc accuracy: {fold_metrics['doc_accuracy']:.3f}")
        print(f"  Doc F1 macro: {fold_metrics['doc_f1_macro']:.3f}")

        # Save best model for this fold
        best_model_path = fold_output / "best_model"
        trainer.save_model(str(best_model_path))
        tokenizer.save_pretrained(str(best_model_path))

        fold_results.append(fold_metrics)

        # Clean up GPU memory
        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        fold_time = time.time() - fold_start
        print(f"  Fold {fold_idx} completed in {fold_time:.0f}s")

    total_time = time.time() - total_start

    # === Aggregate results ===
    print(f"\n{'=' * 60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'=' * 60}")

    doc_accs = [r["doc_accuracy"] for r in fold_results]
    doc_f1s = [r["doc_f1_macro"] for r in fold_results]
    doc_precs = [r["doc_precision_macro"] for r in fold_results]
    doc_recs = [r["doc_recall_macro"] for r in fold_results]
    train_losses = [r["train_loss"] for r in fold_results]

    print(f"\nDocument-level metrics (mean +/- std):")
    print(f"  Accuracy:  {np.mean(doc_accs):.3f} +/- {np.std(doc_accs):.3f}")
    print(f"  F1 macro:  {np.mean(doc_f1s):.3f} +/- {np.std(doc_f1s):.3f}")
    print(f"  Precision: {np.mean(doc_precs):.3f} +/- {np.std(doc_precs):.3f}")
    print(f"  Recall:    {np.mean(doc_recs):.3f} +/- {np.std(doc_recs):.3f}")
    print(f"  Train loss: {np.mean(train_losses):.4f} +/- {np.std(train_losses):.4f}")
    print(f"\nTotal training time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Per-fold table
    print(f"\nPer-fold breakdown:")
    print(f"{'Fold':>4}  {'Acc':>6}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'Loss':>8}")
    for r in fold_results:
        print(f"  {r['fold']:>2}  {r['doc_accuracy']:>6.3f}  {r['doc_f1_macro']:>6.3f}  "
              f"{r['doc_precision_macro']:>6.3f}  {r['doc_recall_macro']:>6.3f}  "
              f"{r['train_loss']:>8.4f}")

    # Save metrics
    metrics_output = {
        "model": MODEL_NAME,
        "n_folds": N_FOLDS,
        "n_documents": len(docs),
        "n_total_chunks": len(all_chunks),
        "chunk_size": CHUNK_SIZE,
        "stride": STRIDE,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "per_device_batch_size": PER_DEVICE_BATCH,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "effective_batch_size": PER_DEVICE_BATCH * GRAD_ACCUM,
        },
        "class_weights": {ID2LABEL[i]: w for i, w in enumerate(class_weights)},
        "label_distribution": {
            ID2LABEL[i]: int((doc_labels == i).sum()) for i in range(NUM_LABELS)
        },
        "aggregate": {
            "doc_accuracy_mean": float(np.mean(doc_accs)),
            "doc_accuracy_std": float(np.std(doc_accs)),
            "doc_f1_macro_mean": float(np.mean(doc_f1s)),
            "doc_f1_macro_std": float(np.std(doc_f1s)),
            "doc_precision_macro_mean": float(np.mean(doc_precs)),
            "doc_precision_macro_std": float(np.std(doc_precs)),
            "doc_recall_macro_mean": float(np.mean(doc_recs)),
            "doc_recall_macro_std": float(np.std(doc_recs)),
            "train_loss_mean": float(np.mean(train_losses)),
            "train_loss_std": float(np.std(train_losses)),
        },
        "per_fold": fold_results,
        "total_time_seconds": total_time,
        "device": device,
    }

    metrics_path = OUTPUT_DIR / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved metrics to {metrics_path}")

    # Plot confusion matrices
    cm_path = OUTPUT_DIR / "confusion_matrices.png"
    plot_confusion_matrices(fold_results, cm_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
