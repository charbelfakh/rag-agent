"""Local GPU sentence-transformer embedder (``EMBED_PROVIDER=gpu``)."""
import os
from sentence_transformers import SentenceTransformer
from providers.base import EmbedProvider

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    # Singleton avoids reloading a multi-GB model on every embed call.
    if _model is None:
        model_name = os.getenv("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1")
        device = os.getenv("EMBED_DEVICE", "cuda")
        _model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=device,
        )
    return _model


class GPUEmbedder(EmbedProvider):
    """Encode text batches on CUDA via sentence-transformers."""

    def __init__(self):
        self.batch_size = int(os.getenv("EMBED_ENCODE_BATCH_SIZE", "512"))
        self.model = _get_model()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if len(texts) > 1:
            print(f"  Encoding {len(texts)} chunks on GPU...", flush=True)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()
