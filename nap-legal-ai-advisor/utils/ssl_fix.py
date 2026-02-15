"""
SSL workaround for corporate proxy environments.

Must be imported before any HuggingFace / httpx / google-generativeai calls.
Patches httpx (sync + async) and requests.Session to skip SSL verification.
Sets gRPC/Google API env vars so any gRPC code paths are less strict; Gemini
is configured to use REST transport in llm_engine so API calls go through the
patched requests library instead of gRPC.
"""

import os
import ssl

# Environment-level SSL bypass
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# gRPC / Gemini SSL bypass — must be set before any google imports
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = ""
os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"

ssl._create_default_https_context = ssl._create_unverified_context

# Patch httpx — both sync and async clients
import httpx

_orig_client_init = httpx.Client.__init__
_orig_async_client_init = httpx.AsyncClient.__init__


def _patched_client_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _orig_client_init(self, *args, **kwargs)


def _patched_async_client_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _orig_async_client_init(self, *args, **kwargs)


httpx.Client.__init__ = _patched_client_init
httpx.AsyncClient.__init__ = _patched_async_client_init

# Patch requests.Session — used by google-generativeai REST transport
import requests

_orig_session_request = requests.Session.request


def _patched_session_request(self, *args, **kwargs):
    kwargs["verify"] = False
    return _orig_session_request(self, *args, **kwargs)


requests.Session.request = _patched_session_request

# Also patch urllib3 if present (used by requests/transformers)
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass
