"""Sprint M tests: PDF image extract, SigLIP embed, hybrid RRF fusion."""


from providers.hybrid_retrieval import merge_text_and_image_hits, reciprocal_rank_fusion
from providers.pdf_images import extract_pdf_images
from providers.siglip_embed import SigLIPEmbedder


class TestHybridRetrieval:
    def test_rrf_merges_lists(self):
        text_hits = [
            {"chunk_id": "a", "source": "a.pdf", "score": 0.9},
            {"chunk_id": "b", "source": "b.pdf", "score": 0.5},
        ]
        image_hits = [
            {"chunk_id": "b", "source": "b.pdf", "score": 0.8},
            {"chunk_id": "c", "source": "c.pdf", "score": 0.7},
        ]
        fused = reciprocal_rank_fusion(text_hits, image_hits)
        assert len(fused) == 3
        assert fused[0]["chunk_id"] in {"a", "b", "c"}

    def test_merge_without_images(self):
        text = [{"chunk_id": "a", "score": 0.9}]
        assert merge_text_and_image_hits(text, [], top_k=1)[0]["chunk_id"] == "a"

    def test_rrf_keeps_timestamped_video_segments_distinct(self):
        video_hits = [
            {
                "source": "Random Bin Picking Tutorial Introduction",
                "start_seconds": 0.12,
                "score": 0.7,
            },
            {
                "source": "Random Bin Picking Tutorial Introduction",
                "start_seconds": 93.4,
                "score": 0.6,
            },
        ]
        fused = reciprocal_rank_fusion(video_hits)
        assert len(fused) == 2


class TestSigLIPEmbedder:
    def test_hash_fallback_vectors(self):
        embedder = SigLIPEmbedder()
        embedder._model = None
        vectors = embedder.embed_bytes([b"abc", b"def"])
        assert len(vectors) == 2
        assert len(vectors[0]) == embedder.dimensions


class TestPdfImages:
    def test_extract_pdf_images_empty_when_unavailable(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "pymupdf", None)
        assert extract_pdf_images("missing.pdf") == []

    def test_image_payload_builder_for_ingest(self):
        from providers.metadata import build_image_chunk_payload, resolve_metadata

        meta = resolve_metadata("docs/a.pdf")
        payload = build_image_chunk_payload(
            metadata=meta,
            chunk_index=2,
            page=1,
            media_uri="/media/abc/1_0.png",
            media_hash="hash",
            image_class="diagram",
            width=100,
            height=80,
        )
        assert payload["content_type"] == "image"
        assert payload["chunk_index"] == 2
