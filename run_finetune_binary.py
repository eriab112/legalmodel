"""
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

print(f"\nTraining Configuration:")
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

print(f"\nStarting binary fine-tuning...")
start_time = time.time()

trainer.train()

elapsed = time.time() - start_time
print(f"\nFine-tuning complete in {elapsed/60:.1f} minutes")

# Clear cache before saving
torch.cuda.empty_cache()
gc.collect()

# Save final model
os.makedirs("models/nap_binary/best", exist_ok=True)
model.save_pretrained("models/nap_binary/best")
tokenizer.save_pretrained("models/nap_binary/best")

print(f"Model saved to: models/nap_binary/best/")
print("\n" + "=" * 60)
print("BINARY FINE-TUNING COMPLETE")
print("=" * 60)
