"""
Fine-Tuning with Weighted Loss for NAP Legal AI
Runs after DAPT (Step 2). Optimized for NVIDIA T500 (4GB VRAM).
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
print("STEP 3: FINE-TUNING WITH WEIGHTED LOSS")
print("=" * 60)

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")


# NAPDataset must be defined here so torch.load can unpickle it
class NAPDataset(TorchDataset):
    """Dataset with sliding window for court docs."""

    def __init__(self, samples=None, tokenizer=None, max_len=512, stride=256):
        self.samples = []
        if tokenizer is not None:
            self.cls_id = tokenizer.cls_token_id
            self.sep_id = tokenizer.sep_token_id
            self.pad_id = tokenizer.pad_token_id
        if samples is None:
            return
        label_map = {"HIGH_RISK": 0, "MEDIUM_RISK": 1, "LOW_RISK": 2}
        for s in samples:
            label = label_map[s["label"]]
            weight = s["weight"]
            source = s["source"]
            if source == "court":
                tokens = tokenizer.encode(s["text"], add_special_tokens=False)
                for i in range(0, len(tokens), stride):
                    chunk = tokens[i : i + max_len]
                    if len(chunk) >= 50:
                        self.samples.append(
                            {
                                "tokens": chunk,
                                "label": label,
                                "weight": weight,
                                "source": source,
                            }
                        )
            else:
                tokens = tokenizer.encode(
                    s["text"],
                    max_length=max_len,
                    truncation=True,
                    add_special_tokens=False,
                )
                self.samples.append(
                    {
                        "tokens": tokens,
                        "label": label,
                        "weight": weight,
                        "source": source,
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
            "weights": torch.tensor(s["weight"], dtype=torch.float),
            "source": s["source"],
        }


class WeightedTrainer(Trainer):
    """Custom trainer with per-sample weights and class weights."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        weights = inputs.pop("weights")
        inputs.pop("source", None)

        outputs = model(**inputs)
        logits = outputs.logits

        loss = F.cross_entropy(logits.view(-1, 3), labels.view(-1), reduction="none")

        # Apply class weights to counteract MEDIUM_RISK dominance
        if self.class_weights is not None:
            cw = self.class_weights.to(loss.device)
            class_w = cw[labels.view(-1)]
            loss = loss * class_w

        weighted_loss = (loss * weights.view(-1)).mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss


def custom_collate_fn(features):
    """Custom collator that handles string 'source' field."""
    batch = {}
    batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
    batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
    batch["labels"] = torch.stack([f["labels"] for f in features])
    batch["weights"] = torch.stack([f["weights"] for f in features])
    batch["source"] = [f["source"] for f in features]
    return batch


# Load DAPT model for sequence classification
dapt_path = "models/nap_dapt/final"
print(f"Loading DAPT model from {dapt_path}...")
model = AutoModelForSequenceClassification.from_pretrained(dapt_path, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(dapt_path)

# Set label mapping
model.config.id2label = {0: "HIGH_RISK", 1: "MEDIUM_RISK", 2: "LOW_RISK"}
model.config.label2id = {"HIGH_RISK": 0, "MEDIUM_RISK": 1, "LOW_RISK": 2}

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Enable gradient checkpointing to reduce VRAM usage
model.gradient_checkpointing_enable()
print("Gradient checkpointing: ENABLED (saves ~1GB VRAM)")

# Rebuild datasets from source instead of using torch.load (avoids pickle issues)
print("Building datasets from source data...")

label_map = {"HIGH_RISK": 0, "MEDIUM_RISK": 1, "LOW_RISK": 2}
labeled = json.load(open("Data/processed/labeled_dataset.json", encoding="utf-8"))
weak_data = json.load(
    open("Data/processed/weakly_labeled_applications.json", encoding="utf-8")
)

train_strong = [
    {"text": i["key_text"], "label": i["label"], "weight": 1.0, "source": "court"}
    for i in labeled["splits"]["train"]
]
val_samples = [
    {"text": i["key_text"], "label": i["label"], "weight": 1.0, "source": "court"}
    for i in labeled["splits"]["val"]
]
weak_samples = [
    {
        "text": a["text"],
        "label": a["weak_label"],
        "weight": a["confidence"],
        "source": "application",
    }
    for a in weak_data["applications"]
]

train_all = train_strong + weak_samples

train_ds = NAPDataset(train_all, tokenizer)
val_ds = NAPDataset(val_samples, tokenizer)

# Compute class weights from training chunk distribution
from collections import Counter
label_counts = Counter(s["label"] for s in train_ds.samples)
total = sum(label_counts.values())
n_classes = 3
class_weights = torch.tensor([
    total / (n_classes * label_counts.get(i, 1)) for i in range(n_classes)
], dtype=torch.float)
print(f"Loaded: {len(train_ds)} train chunks, {len(val_ds)} val chunks")
print(f"Train label distribution: HIGH_RISK={label_counts.get(0,0)}, "
      f"MEDIUM_RISK={label_counts.get(1,0)}, LOW_RISK={label_counts.get(2,0)}")
print(f"Class weights: HIGH_RISK={class_weights[0]:.2f}, "
      f"MEDIUM_RISK={class_weights[1]:.2f}, LOW_RISK={class_weights[2]:.2f}")

# Create output directories
os.makedirs("models/nap_final", exist_ok=True)
os.makedirs("logs/finetune", exist_ok=True)

# Clear GPU cache before training
torch.cuda.empty_cache()
gc.collect()

# Training arguments - ultra-conservative for 4GB VRAM
# NO eval or checkpoint saving during training to prevent OOM crash
args = TrainingArguments(
    output_dir="models/nap_final",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # effective batch = 8
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_steps=25,
    logging_dir="logs/finetune",
    eval_strategy="no",    # NO eval during training - prevents OOM
    save_strategy="no",    # NO checkpoints during training - prevents OOM
    report_to="none",
    dataloader_num_workers=0,  # safer on Windows
    max_grad_norm=1.0,
    remove_unused_columns=False,  # keep weights/source for custom loss
)

print(f"\nTraining Configuration:")
print(f"  Epochs: 3 (middle ground: 2=underfit, 5=overfit)")
print(f"  Batch size: 1 (per device)")
print(f"  Gradient accumulation: 8 steps")
print(f"  Effective batch size: 8")
print(f"  Learning rate: 2e-5")
print(f"  Weighted loss: Yes (per-sample + class weights)")
print(f"  Class weights: counteract MEDIUM_RISK dominance")
print(f"  FP16: {torch.cuda.is_available()}")
print(f"  Gradient checkpointing: True")
print(f"  Eval during training: DISABLED (saves memory)")
print(f"  Checkpoint saving: DISABLED (saves memory)")

trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=custom_collate_fn,
)

print(f"\nStarting fine-tuning...")
start_time = time.time()

trainer.train()

elapsed = time.time() - start_time
print(f"\nFine-tuning complete in {elapsed/60:.1f} minutes")

# Clear cache before saving
torch.cuda.empty_cache()
gc.collect()

# Save final model (weights only, no optimizer state)
os.makedirs("models/nap_final/best", exist_ok=True)
model.save_pretrained("models/nap_final/best")
tokenizer.save_pretrained("models/nap_final/best")

print(f"Model saved to: models/nap_final/best/")
print(f"\n{'='*60}")
print("FINE-TUNING COMPLETE - Ready for evaluation (Step 4)")
print(f"{'='*60}")
