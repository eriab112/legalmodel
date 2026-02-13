"""
DAPT (Domain-Adaptive Pre-Training) for NAP Legal AI
Optimized for NVIDIA T500 (4GB VRAM)
"""
import os
import time
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk

print("=" * 60)
print("STEP 2: DOMAIN-ADAPTIVE PRE-TRAINING (DAPT)")
print("=" * 60)

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print("WARNING: No GPU detected, training will be very slow")

# Load base model
print("Loading KB/bert-base-swedish-cased...")
model = AutoModelForMaskedLM.from_pretrained("KB/bert-base-swedish-cased")
tokenizer = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Load datasets
train_ds = load_from_disk("Data/processed/dapt_train")
val_ds = load_from_disk("Data/processed/dapt_val")
print(f"Loaded datasets: {len(train_ds)} train, {len(val_ds)} val")


# Tokenize with chunking for long documents
def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False,
        return_special_tokens_mask=True,
    )


train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])
print(f"Tokenized: {len(train_ds)} train, {len(val_ds)} val")

# Data collator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# Create output directories
os.makedirs("models/nap_dapt", exist_ok=True)
os.makedirs("logs/dapt", exist_ok=True)

# Training arguments - conservative for 4GB VRAM
# Baseline used batch_size=1, grad_accum=8 for classification
# MLM head is larger (50k vocab vs 3 classes), so stay conservative
args = TrainingArguments(
    output_dir="models/nap_dapt",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # effective batch = 16
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    logging_dir="logs/dapt",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    dataloader_num_workers=0,  # safer on Windows
    max_grad_norm=1.0,
)

print(f"\nTraining Configuration:")
print(f"  Epochs: 3")
print(f"  Batch size: 2 (per device)")
print(f"  Gradient accumulation: 8 steps")
print(f"  Effective batch size: 16")
print(f"  Learning rate: 2e-5")
print(f"  MLM probability: 0.15")
print(f"  FP16: {torch.cuda.is_available()}")

trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

print(f"\nStarting DAPT training...")
start_time = time.time()

trainer.train()

elapsed = time.time() - start_time
print(f"\nDAPT training complete in {elapsed/3600:.1f} hours")

# Save final model
model.save_pretrained("models/nap_dapt/final")
tokenizer.save_pretrained("models/nap_dapt/final")

print(f"DAPT model saved to: models/nap_dapt/final/")

# Log final eval
eval_results = trainer.evaluate()
print(f"Final eval loss: {eval_results['eval_loss']:.4f}")

print(f"\n{'='*60}")
print("DAPT COMPLETE - Ready for fine-tuning (Step 3)")
print(f"{'='*60}")
