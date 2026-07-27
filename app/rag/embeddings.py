from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol, Sequence

import numpy as np

from app.core.config import settings


class EmbeddingProviderError(RuntimeError):
    """Embedding provider yuklash yoki inference xatosi."""


class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str
    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class EmbeddingProviderStatus:
    provider: str
    model: str
    dimensions: int
    loaded: bool
    cache_dir: str


class LocalONNXE5EmbeddingProvider:
    """Multilingual E5 modelini lokal ONNX Runtime orqali ishlatadi.

    PyTorch talab qilinmaydi. Model birinchi real embedding so‘rovida Hugging Face'dan
    bir marta yuklanadi va keyin lokal cache'dan ishlaydi.
    """

    provider_name = "local_onnx_e5"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        dimensions: int | None = None,
        onnx_file: str | None = None,
        cache_dir: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.RAG_EMBEDDING_MODEL
        self.dimensions = dimensions or settings.RAG_EMBEDDING_DIMENSIONS
        self.onnx_file = onnx_file or settings.RAG_EMBEDDING_ONNX_FILE
        self.cache_dir = Path(cache_dir or settings.RAG_EMBEDDING_CACHE_DIR)
        self.batch_size = batch_size or settings.RAG_EMBEDDING_BATCH_SIZE
        self.max_length = max_length or settings.RAG_EMBEDDING_MAX_LENGTH

        self._session = None
        self._tokenizer = None
        self._load_lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._session is not None and self._tokenizer is not None

    def status(self) -> EmbeddingProviderStatus:
        return EmbeddingProviderStatus(
            provider=self.provider_name,
            model=self.model_name,
            dimensions=self.dimensions,
            loaded=self.loaded,
            cache_dir=str(self.cache_dir),
        )

    def _ensure_loaded(self) -> None:
        if self.loaded:
            return

        with self._load_lock:
            if self.loaded:
                return

            try:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise EmbeddingProviderError(
                    "Lokal embedding dependency'lari o‘rnatilmagan. "
                    "requirements.txt ni qayta o‘rnating."
                ) from exc

            model_dir = self.cache_dir / self.model_name.replace("/", "--")
            model_dir.mkdir(parents=True, exist_ok=True)

            # Tokenizer root fayllari va ONNX modelni lokal papkaga yuklaymiz.
            files = [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "sentencepiece.bpe.model",
                self.onnx_file,
            ]
            try:
                for filename in files:
                    hf_hub_download(
                        repo_id=self.model_name,
                        filename=filename,
                        local_dir=model_dir,
                    )

                tokenizer = AutoTokenizer.from_pretrained(
                    model_dir,
                    local_files_only=True,
                    use_fast=True,
                )
                model_path = model_dir / self.onnx_file
                session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
            except Exception as exc:
                raise EmbeddingProviderError(
                    f"Embedding modelini yuklash/ochish muvaffaqiyatsiz: {exc}"
                ) from exc

            self._tokenizer = tokenizer
            self._session = session

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return vectors / norms

    @staticmethod
    def _average_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask = attention_mask.astype(np.float32)[..., None]
        summed = (last_hidden_state * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        return summed / counts

    def _encode(self, texts: Sequence[str], *, prefix: str) -> list[list[float]]:
        cleaned = [text.strip() for text in texts if text and text.strip()]
        if len(cleaned) != len(texts):
            raise EmbeddingProviderError("Embedding uchun bo‘sh matn yuborib bo‘lmaydi.")
        if not cleaned:
            return []

        self._ensure_loaded()
        assert self._session is not None
        assert self._tokenizer is not None

        results: list[list[float]] = []
        input_names = {item.name for item in self._session.get_inputs()}

        for start in range(0, len(cleaned), self.batch_size):
            batch = [f"{prefix}{text}" for text in cleaned[start : start + self.batch_size]]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )

            ort_inputs: dict[str, np.ndarray] = {}
            for name in input_names:
                if name in encoded:
                    ort_inputs[name] = np.asarray(encoded[name], dtype=np.int64)
                elif name == "token_type_ids":
                    ort_inputs[name] = np.zeros_like(
                        np.asarray(encoded["input_ids"], dtype=np.int64)
                    )

            try:
                outputs = self._session.run(None, ort_inputs)
            except Exception as exc:
                raise EmbeddingProviderError(f"ONNX embedding inference xatosi: {exc}") from exc

            last_hidden_state = np.asarray(outputs[0], dtype=np.float32)
            attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int64)
            pooled = self._average_pool(last_hidden_state, attention_mask)
            normalized = self._normalize(pooled)

            if normalized.shape[1] != self.dimensions:
                raise EmbeddingProviderError(
                    "Embedding dimension mos emas: "
                    f"kutilgan={self.dimensions}, olingan={normalized.shape[1]}"
                )
            results.extend(normalized.astype(np.float32).tolist())

        return results

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        # Multilingual E5 retrieval modeli passage/query prefikslarini talab qiladi.
        return self._encode(texts, prefix="passage: ")

    def embed_query(self, text: str) -> list[float]:
        vectors = self._encode([text], prefix="query: ")
        return vectors[0]


_provider: EmbeddingProvider | None = None
_provider_lock = Lock()


def get_embedding_provider() -> EmbeddingProvider:
    global _provider
    if _provider is not None:
        return _provider

    with _provider_lock:
        if _provider is not None:
            return _provider

        provider_name = settings.RAG_EMBEDDING_PROVIDER.strip().lower()
        if provider_name != "local_onnx_e5":
            raise EmbeddingProviderError(
                f"Qo‘llab-quvvatlanmaydigan RAG_EMBEDDING_PROVIDER: {provider_name}"
            )
        _provider = LocalONNXE5EmbeddingProvider()
        return _provider


def reset_embedding_provider_for_tests() -> None:
    global _provider
    _provider = None
