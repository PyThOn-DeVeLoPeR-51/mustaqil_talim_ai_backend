from __future__ import annotations

import unittest

import numpy as np

from app.rag.embeddings import GeminiEmbeddingProvider


class StubGeminiEmbeddingProvider(GeminiEmbeddingProvider):
    def __init__(self, responses):
        super().__init__(
            api_key="test-key",
            model_name="gemini-embedding-2",
            dimensions=4,
            batch_size=2,
            base_url="https://example.invalid/v1beta",
            max_retries=0,
        )
        self.responses = list(responses)
        self.calls = []

    def _post_json(self, url, payload):
        self.calls.append((url, payload))
        return self.responses.pop(0)


class GeminiEmbeddingProviderTestCase(unittest.TestCase):
    def test_document_batch_uses_retrieval_document_and_dimension(self) -> None:
        provider = StubGeminiEmbeddingProvider(
            [
                {
                    "embeddings": [
                        {"values": [3.0, 4.0, 0.0, 0.0]},
                        {"values": [0.0, 0.0, 5.0, 0.0]},
                    ]
                }
            ]
        )
        vectors = provider.embed_documents(["bir", "ikki"])
        self.assertEqual(len(vectors), 2)
        self.assertTrue(np.allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0]))
        url, payload = provider.calls[0]
        self.assertTrue(url.endswith("/models/gemini-embedding-2:batchEmbedContents"))
        request = payload["requests"][0]
        self.assertEqual(request["model"], "models/gemini-embedding-2")
        self.assertEqual(
            request["embedContentConfig"],
            {
                "taskType": "RETRIEVAL_DOCUMENT",
                "outputDimensionality": 4,
                "autoTruncate": True,
            },
        )

    def test_query_uses_retrieval_query(self) -> None:
        provider = StubGeminiEmbeddingProvider(
            [{"embedding": {"values": [0.0, 3.0, 4.0, 0.0]}}]
        )
        vector = provider.embed_query("savol")
        self.assertEqual(len(vector), 4)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)
        url, payload = provider.calls[0]
        self.assertTrue(url.endswith("/models/gemini-embedding-2:embedContent"))
        self.assertEqual(
            payload["embedContentConfig"]["taskType"], "RETRIEVAL_QUERY"
        )

    def test_large_document_set_is_split_by_batch_size(self) -> None:
        provider = StubGeminiEmbeddingProvider(
            [
                {
                    "embeddings": [
                        {"values": [1.0, 0.0, 0.0, 0.0]},
                        {"values": [1.0, 0.0, 0.0, 0.0]},
                    ]
                },
                {"embeddings": [{"values": [1.0, 0.0, 0.0, 0.0]}]},
            ]
        )
        vectors = provider.embed_documents(["a", "b", "c"])
        self.assertEqual(len(vectors), 3)
        self.assertEqual(len(provider.calls), 2)


if __name__ == "__main__":
    unittest.main()
