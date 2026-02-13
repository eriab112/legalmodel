"""
LegalBERT risk predictor using fold_4 (best fold: Acc=0.750, F1=0.739).

Sliding window inference matches scripts/05_finetune_legalbert.py exactly:
- CHUNK_SIZE=512, STRIDE=256
- Tokenize without special tokens, then add [CLS]/[SEP] per chunk
- Aggregate by averaging logits across chunks, then argmax
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import streamlit as st
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
_default_model = BASE_DIR / "models" / "nap_legalbert_cv" / "fold_4" / "best_model"
_env_model = os.getenv("MODEL_PATH")
MODEL_DIR = Path(_env_model) if _env_model and Path(_env_model).is_absolute() else (BASE_DIR / _env_model if _env_model else _default_model)

LABEL_MAP = {"HIGH_RISK": 0, "MEDIUM_RISK": 1, "LOW_RISK": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = 3

CHUNK_SIZE = 512
STRIDE = 256


@dataclass
class PredictionResult:
    predicted_label: str
    probabilities: Dict[str, float]
    confidence: float
    num_chunks: int
    chunk_predictions: List[Dict]
    ground_truth: Optional[str] = None


class LegalBERTPredictor:
    """Risk prediction using fine-tuned KB-BERT (fold_4)."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load_model(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_DIR), num_labels=NUM_LABELS
        )

        if self._device.type == "cuda":
            self._model = self._model.half()  # fp16 for GPU

        self._model.to(self._device)
        self._model.eval()
        print(f"LegalBERT loaded from: {MODEL_DIR}")

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    @property
    def device(self):
        if self._device is None:
            self._load_model()
        return self._device

    def _create_chunks(self, text: str) -> List[Dict]:
        """Tokenize text and split into overlapping 512-token chunks.

        Matches create_chunks() from scripts/05_finetune_legalbert.py:162-226 exactly.
        """
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        token_ids = encoded["input_ids"]

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        max_content = CHUNK_SIZE - 2  # reserve [CLS] and [SEP]

        chunks = []

        if len(token_ids) <= max_content:
            # Single chunk
            chunk_ids = [cls_id] + token_ids + [sep_id]
            attn_mask = [1] * len(chunk_ids)
            pad_len = CHUNK_SIZE - len(chunk_ids)
            chunk_ids += [self.tokenizer.pad_token_id] * pad_len
            attn_mask += [0] * pad_len

            chunks.append({
                "input_ids": chunk_ids,
                "attention_mask": attn_mask,
            })
        else:
            # Sliding window
            content_stride = STRIDE
            start = 0
            while start < len(token_ids):
                end = min(start + max_content, len(token_ids))
                content = token_ids[start:end]

                chunk_ids = [cls_id] + content + [sep_id]
                attn_mask = [1] * len(chunk_ids)
                pad_len = CHUNK_SIZE - len(chunk_ids)
                if pad_len > 0:
                    chunk_ids += [self.tokenizer.pad_token_id] * pad_len
                    attn_mask += [0] * pad_len

                chunks.append({
                    "input_ids": chunk_ids,
                    "attention_mask": attn_mask,
                })

                if end >= len(token_ids):
                    break
                start += content_stride

        return chunks

    def predict_text(self, text: str, ground_truth: Optional[str] = None) -> PredictionResult:
        """Run inference on text using sliding window + logit aggregation.

        Matches aggregate_doc_predictions() from scripts/05_finetune_legalbert.py:237-259.
        """
        chunks = self._create_chunks(text)

        all_logits = []
        chunk_predictions = []

        with torch.no_grad():
            for i, chunk in enumerate(chunks):
                input_ids = torch.tensor([chunk["input_ids"]], device=self.device)
                attention_mask = torch.tensor([chunk["attention_mask"]], device=self.device)

                if self.device.type == "cuda":
                    input_ids = input_ids.long()
                    attention_mask = attention_mask.long()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.float().cpu().numpy()[0]
                all_logits.append(logits)

                # Per-chunk prediction
                chunk_probs = _softmax(logits)
                chunk_pred = int(np.argmax(chunk_probs))
                chunk_predictions.append({
                    "chunk_index": i,
                    "predicted_label": ID2LABEL[chunk_pred],
                    "probabilities": {
                        ID2LABEL[j]: round(float(chunk_probs[j]), 4)
                        for j in range(NUM_LABELS)
                    },
                })

        # Aggregate: average logits across chunks, then softmax
        avg_logits = np.mean(all_logits, axis=0)
        probabilities = _softmax(avg_logits)
        predicted_idx = int(np.argmax(probabilities))
        predicted_label = ID2LABEL[predicted_idx]

        return PredictionResult(
            predicted_label=predicted_label,
            probabilities={
                ID2LABEL[i]: round(float(probabilities[i]), 4)
                for i in range(NUM_LABELS)
            },
            confidence=round(float(probabilities[predicted_idx]), 4),
            num_chunks=len(chunks),
            chunk_predictions=chunk_predictions,
            ground_truth=ground_truth,
        )

    def predict_decision(self, decision) -> PredictionResult:
        """Run prediction on a DecisionRecord, using key_text."""
        text = decision.key_text if decision.key_text else decision.full_text[:10000]
        return self.predict_text(text, ground_truth=decision.label)


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


@st.cache_resource
def get_predictor():
    return LegalBERTPredictor()
