"""
SSL workaround for corporate proxy environments.

Must be imported before any HuggingFace / httpx calls.
Matches the pattern from scripts/05_finetune_legalbert.py.
"""

import os
import ssl

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
