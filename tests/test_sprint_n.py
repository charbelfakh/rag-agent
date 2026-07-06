"""Sprint N tests: VLM LLM, ColPali stub."""
import json
from unittest.mock import MagicMock

import httpx

from providers.colpali_embed import ColPaliEmbedder
from providers.vlm_llm import VLMLLM


class TestVLMLLM:
    def test_generate_with_images(self, monkeypatch):
        monkeypatch.setenv("VLM_BASE_URL", "http://vlm.test/v1")
        llm = VLMLLM()

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            content = body["messages"][0]["content"]
            assert any(part.get("type") == "image_url" for part in content)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Wiring answer"}}]},
            )

        transport = httpx.MockTransport(handler)
        llm._client = httpx.Client(transport=transport)
        answer = llm.generate_with_images(
            "Describe wiring",
            image_uris=["http://localhost/media/a.png"],
        )
        assert answer == "Wiring answer"


class TestColPali:
    def test_embed_page_images_deterministic(self):
        embedder = ColPaliEmbedder()
        vectors = embedder.embed_page_images([b"page-a", b"page-b"])
        assert len(vectors) == 2
        assert len(vectors[0]) == embedder.dimensions


class TestFactoryVLM:
    def test_factory_selects_vlm_provider(self, monkeypatch):
        from unittest.mock import patch

        from providers import factory

        factory.reset_providers()
        monkeypatch.setenv("LLM_PROVIDER", "vlm")
        with patch("providers.vlm_llm.VLMLLM") as constructor:
            constructor.return_value = MagicMock(name="vlm")
            llm = factory.get_llm()
        assert llm is not None
        factory.reset_providers()
