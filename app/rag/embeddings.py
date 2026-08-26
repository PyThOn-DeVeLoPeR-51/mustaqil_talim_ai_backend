from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import time
from typing import Any, Protocol, Sequence

import httpx
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


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Cosine-search uchun vektorlarni L2 normalize qiladi."""

    if vectors.ndim != 2:
        raise EmbeddingProviderError(
            f"Embedding matrix shape noto‘g‘ri: {vectors.shape}"
        )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


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
        return _normalize_vectors(vectors)

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


class GeminiEmbeddingProvider:
    """Gemini Embedding API orqali low-memory remote embedding provider.

    Render kabi RAM cheklangan production muhitida lokal ONNX modelni process
    ichida yuklamasdan, faqat matn va embedding vektorlarini HTTP orqali uzatadi.
    """

    provider_name = "gemini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model_name = model_name or settings.RAG_EMBEDDING_MODEL
        self.dimensions = dimensions or settings.RAG_EMBEDDING_DIMENSIONS
        self.batch_size = max(1, batch_size or settings.RAG_EMBEDDING_BATCH_SIZE)
        self.base_url = (base_url or settings.RAG_GEMINI_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds or settings.RAG_EMBEDDING_TIMEOUT_SECONDS
        self.max_retries = (
            settings.RAG_EMBEDDING_MAX_RETRIES if max_retries is None else max_retries
        )

    @property
    def _model_resource(self) -> str:
        clean = self.model_name.strip()
        return clean if clean.startswith("models/") else f"models/{clean}"

    def status(self) -> EmbeddingProviderStatus:
        return EmbeddingProviderStatus(
            provider=self.provider_name,
            model=self.model_name,
            dimensions=self.dimensions,
            loaded=True,
            cache_dir="",
        )

    def _require_api_key(self) -> str:
        key = (self.api_key or "").strip()
        if not key:
            raise EmbeddingProviderError(
                "GEMINI_API_KEY topilmadi. Render Environment'da kalitni sozlang."
            )
        return key

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        api_key = self._require_api_key()
        last_error: Exception | None = None

        for attempt in range(max(0, self.max_retries) + 1):
            try:
                response = httpx.post(
                    url,
                    headers={
                        "x-goog-api-key": api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    if attempt < self.max_retries:
                        time.sleep(min(2.0 ** attempt, 4.0))
                        continue
                if response.is_error:
                    body = response.text.strip().replace("\n", " ")[:800]
                    raise EmbeddingProviderError(
                        "Gemini embedding API xatosi: "
                        f"HTTP {response.status_code}: {body or response.reason_phrase}"
                    )
                data = response.json()
                if not isinstance(data, dict):
                    raise EmbeddingProviderError("Gemini embedding javobi JSON object emas.")
                return data
            except EmbeddingProviderError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2.0 ** attempt, 4.0))
                    continue
                break

        raise EmbeddingProviderError(
            f"Gemini embedding API bilan aloqa xatosi: {last_error}"
        ) from last_error

    def _config(self, *, task_type: str) -> dict[str, Any]:
        return {
            "taskType": task_type,
            "outputDimensionality": self.dimensions,
            "autoTruncate": True,
        }

    def _validate_and_normalize(self, vectors: Sequence[Sequence[float]]) -> list[list[float]]:
        try:
            matrix = np.asarray(vectors, dtype=np.float32)
        except Exception as exc:
            raise EmbeddingProviderError(f"Gemini embedding vektori noto‘g‘ri: {exc}") from exc

        if matrix.ndim != 2 or matrix.shape[1] != self.dimensions:
            raise EmbeddingProviderError(
                "Gemini embedding dimension mos emas: "
                f"kutilgan={self.dimensions}, olingan={matrix.shape}"
            )
        return _normalize_vectors(matrix).astype(np.float32).tolist()

    def _embed_batch(self, texts: Sequence[str], *, task_type: str) -> list[list[float]]:
        cleaned = [text.strip() for text in texts if text and text.strip()]
        if len(cleaned) != len(texts):
            raise EmbeddingProviderError("Embedding uchun bo‘sh matn yuborib bo‘lmaydi.")
        if not cleaned:
            return []

        resource = self._model_resource
        model_id = resource.removeprefix("models/")
        url = f"{self.base_url}/models/{model_id}:batchEmbedContents"
        results: list[list[float]] = []

        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start : start + self.batch_size]
            payload = {
                "requests": [
                    {
                        "model": resource,
                        "content": {"parts": [{"text": text}]},
                        "embedContentConfig": self._config(task_type=task_type),
                    }
                    for text in batch
                ]
            }
            data = self._post_json(url, payload)
            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise EmbeddingProviderError(
                    "Gemini batch embedding javobidagi embedding soni mos emas."
                )
            vectors: list[list[float]] = []
            for item in embeddings:
                values = item.get("values") if isinstance(item, dict) else None
                if not isinstance(values, list):
                    raise EmbeddingProviderError(
                        "Gemini batch embedding javobida values topilmadi."
                    )
                vectors.append(values)
            results.extend(self._validate_and_normalize(vectors))

        return results

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        clean = text.strip()
        if not clean:
            raise EmbeddingProviderError("Embedding uchun bo‘sh matn yuborib bo‘lmaydi.")

        resource = self._model_resource
        model_id = resource.removeprefix("models/")
        url = f"{self.base_url}/models/{model_id}:embedContent"
        payload = {
            "content": {"parts": [{"text": clean}]},
            "embedContentConfig": self._config(task_type="RETRIEVAL_QUERY"),
        }
        data = self._post_json(url, payload)
        embedding = data.get("embedding")
        values = embedding.get("values") if isinstance(embedding, dict) else None
        if not isinstance(values, list):
            raise EmbeddingProviderError(
                "Gemini query embedding javobida embedding.values topilmadi."
            )
        return self._validate_and_normalize([values])[0]


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
        if provider_name == "local_onnx_e5":
            _provider = LocalONNXE5EmbeddingProvider()
        elif provider_name in {"gemini", "gemini_embedding"}:
            _provider = GeminiEmbeddingProvider()
        else:
            raise EmbeddingProviderError(
                f"Qo‘llab-quvvatlanmaydigan RAG_EMBEDDING_PROVIDER: {provider_name}"
            )
        return _provider


def reset_embedding_provider_for_tests() -> None:
    global _provider
    _provider = None
