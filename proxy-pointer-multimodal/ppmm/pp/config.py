"""Central configuration. Override any value via environment variables / .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from the ppmm root directory relative to this config file
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Defaulting to OpenAI/Azure models.
GEN_MODEL = os.getenv("PP_GEN_MODEL") or os.getenv("AZURE_OPENAI_LLM_MODEL_LLM_MODEL") or "gpt-4o"
EMBED_MODEL = os.getenv("PP_EMBED_MODEL", "text-embedding-3-large")
EMBED_DIM = int(os.getenv("PP_EMBED_DIM", "3072"))


# Azure OpenAI Embedding (separate from regular OpenAI)
OPENAI_API_KEY           = os.getenv("OPENAI_API_KEY", "")
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_LLM_MODEL_API_KEY") or ""

AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_LLM_MODEL_API_BASE") or ""

AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_LLM_MODEL_API_VERSION") or "2024-02-01"

# We use the deploy names if provided, else the model name.
AZURE_EMBED_DEPLOYMENT   = os.getenv("AZURE_EMBED_DEPLOYMENT") or os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_EMBEDDING_MODEL") or ""

# Separate Azure OpenAI Embedding Configurations (if different from LLM resource)
AZURE_OPENAI_EMBEDDING_API_KEY     = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_API_KEY") or ""
AZURE_OPENAI_EMBEDDING_ENDPOINT    = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_API_BASE") or ""
AZURE_OPENAI_EMBEDDING_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION") or "2024-02-01"

# Aliases for pp/embedding_model.py compatibility
AZURE_OPENAI_EMBEDDING_API_BASE    = AZURE_OPENAI_EMBEDDING_ENDPOINT
AZURE_OPENAI_EMBEDDING_MODEL       = AZURE_EMBED_DEPLOYMENT

# Azure AI Document Intelligence
DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT") or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT") or ""
DOCUMENT_INTELLIGENCE_KEY      = (
    os.getenv("DOCUMENTINTELLIGENCE_API_KEY")
    or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")
    or os.getenv("DOCUMENTINTELLIGENCE_KEY")
    or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    or ""
)


def has_document_intelligence() -> bool:
    return bool(DOCUMENT_INTELLIGENCE_ENDPOINT and DOCUMENT_INTELLIGENCE_KEY)

# ---------------------------------------------------------------------------
# Retrieval knobs (mirror the original Proxy-Pointer MultiModal pipeline)
# ---------------------------------------------------------------------------
RECALL_K = int(os.getenv("PP_RECALL_K", "200"))      # broad vector recall
CANDIDATE_K = int(os.getenv("PP_CANDIDATE_K", "50"))  # unique candidates to re-rank
FINALIST_K = int(os.getenv("PP_FINALIST_K", "5"))     # nodes loaded for synthesis
SNIPPET_CHARS = int(os.getenv("PP_SNIPPET_CHARS", "150"))
MAX_IMAGES_PER_ANSWER = int(os.getenv("PP_MAX_IMAGES", "6"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent  # project root (ppmm/)
DATA_DIR = Path(os.getenv("PP_DATA_DIR", _DEFAULT_ROOT / "data"))

PAPERS_DIR = Path(os.getenv("PP_PAPERS_DIR", DATA_DIR / "extracted_papers"))
TREES_DIR = Path(os.getenv("PP_TREES_DIR", DATA_DIR / "trees"))
INDEX_DIR = Path(os.getenv("PP_INDEX_DIR", DATA_DIR / "index"))
PROXIES_DIR = Path(os.getenv("PP_PROXIES_DIR", DATA_DIR / "proxies"))

# The "diff file" that stores every proxy-pointer as it is created.
PROXY_FILE = Path(os.getenv("PP_PROXY_FILE", PROXIES_DIR / "proxy_pointers.jsonl"))

INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.json"

for _d in (PAPERS_DIR, TREES_DIR, INDEX_DIR, PROXIES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def require_api_key() -> str:
    # Use OpenAI API key as the default required key
    if not (OPENAI_API_KEY or AZURE_OPENAI_API_KEY):
        raise RuntimeError(
            "No OpenAI or Azure API key found. Put OPENAI_API_KEY=... "
            "in a .env file next to app.py, or set it in your environment."
        )
    return OPENAI_API_KEY or AZURE_OPENAI_API_KEY
