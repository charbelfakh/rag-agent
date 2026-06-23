"""Sprint B unit tests: dual vector_score / rerank_score preservation."""
import sys
from unittest.mock import MagicMock, patch

import pytest

from providers.reranker import CrossEncoderReranker, NoOpReranker, _attach_rerank_scores


class TestAttachRerankScores:
    def test_preserves_vector_and_sets_rerank(self):
        chunks = [{"text": "a", "source": "s.pdf", "page": 0, "score": 0.7}]
        result = _attach_rerank_scores(chunks, [0.95], top_n=1)
        assert result[0]["vector_score"] == 0.7
        assert result[0]["rerank_score"] == 0.95
        assert result[0]["score"] == 0.95


class TestNoOpRerankerDualScores:
    def test_noop_sets_both_scores_to_vector(self):
        chunks = [{"text": "a", "score": 0.66}]
        result = NoOpReranker().rerank("q", chunks, top_n=1)
        assert result[0]["vector_score"] == 0.66
        assert result[0]["rerank_score"] == 0.66


class TestCrossEncoderDualScores:
    def test_cross_encoder_preserves_vector_score(self):
        chunks = [{"text": "x" * 20, "source": "s.pdf", "page": 0, "score": 0.55}]
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.99]
        mock_st = MagicMock()
        mock_st.CrossEncoder.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            result = CrossEncoderReranker().rerank("query", chunks, top_n=1)

        assert result[0]["vector_score"] == 0.55
        assert result[0]["rerank_score"] == 0.99
