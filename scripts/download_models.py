"""
Pre-download all models needed by NAP Legal AI Advisor.
Run this on a machine with unrestricted internet access,
then copy the cache to the target machine.

Usage:
    python scripts/download_models.py

Cache location (copy this entire folder):
    Windows: %USERPROFILE%\\.cache\\huggingface\\
    Linux/Mac: ~/.cache/huggingface/
"""

import os
import sys
from pathlib import Path

# Add project for ssl_fix
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nap-legal-ai-advisor"))
import utils.ssl_fix  # noqa: F401

print("Downloading sentence-transformers model...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print(f"[OK] Model loaded: {model.get_sentence_embedding_dimension()}d embeddings")

print("\nDownloading KB-BERT tokenizer (for risk predictor)...")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")
print(f"[OK] Tokenizer loaded: {tok.vocab_size} tokens")

print("\n=== All models cached successfully ===")
print("Cache location:")
cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
print(f"  {cache_dir}")
print("Copy this folder to the target machine if running behind a proxy.")
