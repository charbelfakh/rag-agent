"""Provider factory: instantiate LLM, embedder, vector store, and reranker from ``.env``."""
import os
from dotenv import load_dotenv
from providers.base import LLMProvider, EmbedProvider, VectorStore, Reranker

load_dotenv()

_llm: LLMProvider | None = None
_embedder: EmbedProvider | None = None
_vector_store: VectorStore | None = None


_image_embedder = None
_vlm = None


def reset_providers() -> None:
    """Clear cached provider instances (for tests)."""
    global _llm, _embedder, _vector_store, _reranker, _image_embedder, _vlm
    _llm = None
    _embedder = None
    _vector_store = None
    _reranker = None
    _image_embedder = None
    _vlm = None


def get_llm() -> LLMProvider:
    """Return the configured LLM provider (``LLM_PROVIDER`` env var)."""
    global _llm
    if _llm is not None:
        return _llm
    provider = os.getenv("LLM_PROVIDER", "ollama")
    if provider == "ollama":
        from providers.ollama_llm import OllamaLLM
        _llm = OllamaLLM()
    elif provider in ("vllm", "openai_compatible", "tgi"):
        from providers.openai_chat_llm import OpenAIChatLLM
        _llm = OpenAIChatLLM()
    elif provider == "vlm":
        from providers.vlm_llm import VLMLLM
        _llm = VLMLLM()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
    return _llm


def get_embedder() -> EmbedProvider:
    """Return the configured embedding provider (``EMBED_PROVIDER`` env var)."""
    global _embedder
    if _embedder is not None:
        return _embedder
    provider = os.getenv("EMBED_PROVIDER", "ollama")
    if provider == "ollama":
        from providers.ollama_embed import OllamaEmbedder
        _embedder = OllamaEmbedder()
    elif provider == "gpu":
        from providers.gpu_embed import GPUEmbedder
        _embedder = GPUEmbedder()
    elif provider == "tei":
        from providers.tei_embed import TEIEmbedder
        _embedder = TEIEmbedder()
    else:
        raise ValueError(f"Unknown EMBED_PROVIDER: {provider}")
    return _embedder


def get_vector_store() -> VectorStore:
    """Return the configured vector store (``VECTOR_STORE`` env var)."""
    global _vector_store
    if _vector_store is not None:
        return _vector_store
    store = os.getenv("VECTOR_STORE", "qdrant_local")
    if store == "qdrant_local":
        from providers.qdrant_sharded_store import is_vendor_sharding_enabled

        if is_vendor_sharding_enabled():
            from providers.qdrant_sharded_store import VendorShardedQdrantStore

            _vector_store = VendorShardedQdrantStore()
        else:
            from providers.qdrant_store import QdrantLocalStore

            _vector_store = QdrantLocalStore()
    elif store == "qdrant_cloud":
        from providers.qdrant_cloud_store import QdrantCloudStore
        _vector_store = QdrantCloudStore()
    else:
        raise ValueError(f"Unknown VECTOR_STORE: {store}")
    return _vector_store


_reranker: Reranker | None = None


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("true", "1", "yes")


def get_image_embedder():
    """Return SigLIP image embedder for hybrid multimodal retrieval."""
    global _image_embedder
    if _image_embedder is None:
        from providers.siglip_embed import SigLIPEmbedder

        _image_embedder = SigLIPEmbedder()
    return _image_embedder


def get_vlm():
    """Return vision-language model provider."""
    global _vlm
    if _vlm is None:
        from providers.vlm_llm import VLMLLM

        _vlm = VLMLLM()
    return _vlm


def get_reranker() -> Reranker:
    """Return cross-encoder reranker or a no-op passthrough based on ``RERANKER_ENABLED``."""
    global _reranker
    if _reranker is None:
        if _env_bool("RERANKER_ENABLED"):
            from providers.reranker import CrossEncoderReranker

            _reranker = CrossEncoderReranker()
        else:
            from providers.reranker import NoOpReranker

            _reranker = NoOpReranker()
    return _reranker