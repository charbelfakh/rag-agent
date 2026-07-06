"""Sprint A unit tests: reranker input truncation."""
from unittest.mock import MagicMock, patch


from providers.reranker import CrossEncoderReranker, truncate_for_rerank


class TestTruncateForRerank:
    def test_short_text_unchanged(self):
        text = "short chunk"
        assert truncate_for_rerank(text, max_chars=512) == text

    def test_long_text_truncated(self):
        text = "x" * 1000
        assert len(truncate_for_rerank(text, max_chars=512)) == 512

    def test_default_max_chars_is_512(self, monkeypatch):
        monkeypatch.setenv("RERANKER_MAX_CHARS", "512")
        # Re-import to pick up env if module was loaded with different value
        text = "a" * 600
        assert len(truncate_for_rerank(text)) == 512


class TestCrossEncoderRerankerTruncation:
    def test_rerank_passes_truncated_text_to_model(self):
        long_text = "x" * 600
        chunks = [{"text": long_text, "source": "s.pdf", "page": 0, "score": 0.5}]

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.99]
        mock_st = MagicMock()
        mock_st.CrossEncoder.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            reranker = CrossEncoderReranker()
            result = reranker.rerank("query", chunks, top_n=1)

        mock_model.predict.assert_called_once()
        pairs = mock_model.predict.call_args[0][0]
        assert pairs[0][1] == "x" * 512
        assert result[0]["text"] == long_text
