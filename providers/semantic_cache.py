"""Redis-backed semantic answer cache keyed by question embedding similarity."""
import json
import logging
import os
import time

logger = logging.getLogger(__name__)

CACHE_KEY = "semantic_cache:entries"
CACHE_STATS_KEY = "semantic_cache:stats"


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("true", "1", "yes")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class LegacySemanticCacheBackend:
    def __init__(self, redis_client, *, threshold: float, max_size: int):
        self._redis = redis_client
        self.threshold = threshold
        self.max_size = max_size

    def _load_entries(self) -> list[dict]:
        raw_entries = self._redis.lrange(CACHE_KEY, 0, -1)
        return [json.loads(item) for item in raw_entries]

    def _incr_stat(self, field: str) -> None:
        try:
            self._redis.hincrby(CACHE_STATS_KEY, field, 1)
        except Exception as exc:
            logger.debug("Cache stat increment failed: %s", exc)

    def lookup(self, vector: list[float]) -> tuple[str, float] | None:
        entries = self._load_entries()
        best_sim = 0.0
        best_answer = None
        for entry in entries:
            sim = cosine_similarity(vector, entry["vector"])
            if sim >= self.threshold and sim > best_sim:
                best_sim = sim
                best_answer = entry["answer"]
        if best_answer is not None:
            self._incr_stat("hits")
            return best_answer, best_sim
        self._incr_stat("misses")
        return None

    def store(self, question: str, vector: list[float], answer: str) -> None:
        payload = json.dumps(
            {"question": question, "vector": vector, "answer": answer}
        )
        pipe = self._redis.pipeline()
        while self._redis.llen(CACHE_KEY) >= self.max_size:
            pipe.lpop(CACHE_KEY)
        pipe.rpush(CACHE_KEY, payload)
        pipe.execute()


class SemanticCache:
    """Lookup and store answers by cosine similarity to cached question vectors."""

    def __init__(self):
        self.enabled = _env_bool("SEMANTIC_CACHE_ENABLED")
        self.threshold = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
        self.max_size = int(os.getenv("SEMANTIC_CACHE_MAX_SIZE", "200"))
        self.backend_name = os.getenv("SEMANTIC_CACHE_BACKEND", "legacy").lower()
        self._redis = None
        self._backend = None
        self.last_lookup_ms: int | None = None
        self.last_lookup_result: str | None = None
        if self.enabled:
            self._connect()

    def _connect(self) -> None:
        try:
            import redis

            url = os.getenv("REDIS_URL", "redis://localhost:6379")
            client = redis.from_url(url, decode_responses=True)
            client.ping()
            self._redis = client
            self._backend = self._create_backend(client)
        except Exception as exc:
            logger.warning("Semantic cache disabled: Redis unreachable (%s)", exc)
            self._redis = None
            self._backend = None

    def _create_backend(self, client):
        if self.backend_name == "redisvl":
            try:
                from providers.semantic_cache_redisvl import RedisVLSemanticCacheBackend

                return RedisVLSemanticCacheBackend(
                    client,
                    threshold=self.threshold,
                    max_size=self.max_size,
                )
            except ImportError:
                logger.warning(
                    "redisvl not installed; falling back to legacy semantic cache"
                )
            except Exception as exc:
                logger.warning(
                    "RedisVL semantic cache unavailable (%s); using legacy backend",
                    exc,
                )
        return LegacySemanticCacheBackend(
            client,
            threshold=self.threshold,
            max_size=self.max_size,
        )

    def lookup(self, vector: list[float]) -> tuple[str, float] | None:
        if not self.enabled or self._backend is None:
            self.last_lookup_result = "disabled"
            return None
        t0 = time.perf_counter()
        try:
            result = self._backend.lookup(vector)
        except Exception as exc:
            logger.warning("Semantic cache lookup failed: %s", exc)
            self.last_lookup_result = "error"
            return None
        finally:
            self.last_lookup_ms = int((time.perf_counter() - t0) * 1000)

        if result is not None:
            self.last_lookup_result = "hit"
            return result
        self.last_lookup_result = "miss"
        return None

    def record_skip(self, reason: str) -> None:
        self.last_lookup_result = reason
        if reason == "history" and self._redis is not None:
            try:
                self._redis.hincrby(CACHE_STATS_KEY, "skipped_history", 1)
            except Exception as exc:
                logger.debug("Cache stat increment failed: %s", exc)

    def store(self, question: str, vector: list[float], answer: str) -> None:
        if not self.enabled or self._backend is None:
            return
        try:
            self._backend.store(question, vector, answer)
        except Exception as exc:
            logger.warning("Semantic cache store failed: %s", exc)

    def get_stats(self) -> dict:
        if self._redis is None:
            return {}
        try:
            raw = self._redis.hgetall(CACHE_STATS_KEY)
            stats = {k: int(v) for k, v in raw.items()}
            stats["backend"] = self.backend_name if self._backend else "disabled"
            return stats
        except Exception:
            return {}


_cache: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    """Return the process-wide semantic cache singleton."""
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache


def reset_semantic_cache() -> None:
    """Clear cached instance (tests)."""
    global _cache
    _cache = None
