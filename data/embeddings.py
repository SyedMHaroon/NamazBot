import os
import time
import math
from typing import List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# -------------------- ENV & Defaults --------------------
# Accept multiple token var names to avoid 403s due to "missing token"
HF_API_TOKEN = (
    os.getenv("HF_API_TOKEN")
    or os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or ""
).strip()

if not HF_API_TOKEN:
    raise ValueError(
        "HF API token missing. Set HF_API_TOKEN or HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) in your .env"
    )

# Base URL for Hugging Face Inference API (do not add trailing slash)
HF_API_BASE = os.getenv("HF_API_BASE", "https://api-inference.huggingface.co")

# Choose the model to use
EMBED_MODEL = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-base-en-v1.5")

# Known dims (we default to 768 for the selected model)
_DEFAULT_DIMS = {
    "BAAI/bge-base-en-v1.5": 768,
}

EMBED_DIM = int(
    os.getenv("EMBEDDINGS_DIM") or str(_DEFAULT_DIMS.get(EMBED_MODEL, 768))
)

EMBED_TIMEOUT = float(os.getenv("EMBED_TIMEOUT", "25"))
EMBED_MAX_BATCH = int(os.getenv("EMBED_MAX_BATCH", "64"))

# -------------------- URL builder --------------------
def _embedding_url(model: str) -> str:
    return f"{HF_API_BASE}/models/{model}"

# -------------------- HTTP client --------------------
_client = httpx.Client(
    timeout=httpx.Timeout(EMBED_TIMEOUT),
    http2=True,
    headers={
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    },
)

# -------------------- Helpers --------------------
def _l2_normalize(vec: List[float]) -> List[float]:
    """Normalize the vector to unit length for cosine similarity."""
    s = math.sqrt(sum(x * x for x in vec)) or 1.0
    inv = 1.0 / s
    return [x * inv for x in vec]

# -------------------- Client --------------------
class EmbeddingsClient:
    """
    Client to interact with Hugging Face's inference API for embeddings.
    Features:
      - Supports batching
      - Retries on 429/503
      - L2 normalization
    """

    def __init__(
        self,
        model: str = EMBED_MODEL,
        api_token: str = HF_API_TOKEN,
        dim: int = EMBED_DIM,
        max_batch: int = EMBED_MAX_BATCH,
        timeout: float = EMBED_TIMEOUT,
        normalize: bool = True,
    ):
        if not api_token:
            raise ValueError("HF_API_TOKEN (or HUGGINGFACEHUB_API_TOKEN / HF_TOKEN) is not set.")
        self.model = model
        self.dim = dim
        self.max_batch = max_batch
        self.normalize = normalize
        self.url = _embedding_url(model)

    def embed(self, texts: List[str], retry: int = 2) -> List[List[float]]:
        """Create embeddings for a list of texts."""
        out: List[List[float]] = []
        for i in range(0, len(texts), self.max_batch):
            chunk = texts[i : i + self.max_batch]
            vecs = self._embed_chunk(chunk, retry=retry)
            out.extend(vecs)
        return out

    def _embed_chunk(self, texts: List[str], retry: int = 2) -> List[List[float]]:
        """Helper function to embed a chunk of texts."""
        payload = {"inputs": texts}
        backoff = 1.0
        last_err: Optional[Exception] = None

        for _ in range(retry + 1):
            try:
                r = _client.post(self.url, json=payload)

                # Handle transient issues like rate limiting or model loading
                if r.status_code in (429, 503):
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8)
                    continue

                r.raise_for_status()

                data = r.json()

                # Ensure data is in expected format
                if isinstance(data, list) and data and isinstance(data[0], list):
                    vecs = data
                elif isinstance(data, list) and data and isinstance(data[0], dict) and "embedding" in data[0]:
                    vecs = [d["embedding"] for d in data]
                else:
                    raise RuntimeError(f"Unexpected HF response format: {str(data)[:200]}")

                # Normalize the vectors if required
                if self.normalize:
                    vecs = [_l2_normalize(v) for v in vecs]

                # Sanity check dimensions
                if any(len(v) != self.dim for v in vecs):
                    raise RuntimeError(
                        f"Embedding dim mismatch. Got {[len(v) for v in vecs][:3]} (first 3), expected {self.dim}. "
                        "Set EMBEDDINGS_DIM to the correct size for your model."
                    )
                return vecs

            except Exception as e:
                last_err = e
                time.sleep(backoff)
                backoff = min(backoff * 2, 8)

        raise RuntimeError(f"HF embedding failed after retries: {last_err}")

# -------------------- Convenience API --------------------
emb = EmbeddingsClient()

def embed_query(text: str) -> List[float]:
    """Convenient wrapper to embed a single query."""
    return emb.embed([text])[0]

def embed_many(texts: List[str]) -> List[List[float]]:
    """Convenient wrapper to embed multiple queries."""
    return emb.embed(texts)
