from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import logging
from typing import Iterator

from app.rag.document import ParsedDocument

logger = logging.getLogger(__name__)

METADATA_DICTIONARY_FIELDS = {"category", "dish_name", "difficulty"}
DOCUMENT_DATA_SOURCES = {"recipes", "personal"}
GLOBAL_USER_ID = "GLOBAL"


class MetadataDictionaryCache:
    def load(self) -> dict[str, dict] | None:
        raise NotImplementedError

    def save(self, global_cache: dict[str, list[str]], user_cache: dict[str, dict[str, list[str]]]) -> None:
        raise NotImplementedError


class RedisMetadataDictionaryCache(MetadataDictionaryCache):
    key = "rag:metadata:dictionary"

    def __init__(self, redis_url: str) -> None:
        if not isinstance(redis_url, str) or not redis_url.strip():
            raise ValueError("Redis metadata cache URL must be text")
        import redis

        self.client = redis.Redis.from_url(redis_url.strip())

    def load(self) -> dict[str, dict] | None:
        payload = self.client.get(self.key)
        if not payload:
            return None
        text = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
        return dict(json.loads(text))

    def save(self, global_cache: dict[str, list[str]], user_cache: dict[str, dict[str, list[str]]]) -> None:
        self.client.set(
            self.key,
            json.dumps({"global": global_cache, "user": user_cache}, ensure_ascii=False),
        )


class BaseDocumentRepository:
    _global_cache: dict[str, list[str]] = {}
    _user_cache: dict[str, dict[str, list[str]]] = {}

    def create_schema(self) -> None:
        raise NotImplementedError

    def upsert_documents(self, documents: list[ParsedDocument]) -> None:
        raise NotImplementedError

    def list_documents(self, data_source: str | None = None) -> list[ParsedDocument]:
        raise NotImplementedError

    def get_parent_documents(self, parent_ids: list[str]) -> dict[str, ParsedDocument]:
        raise NotImplementedError

    def init_all_metadata_cache(self) -> None:
        raise NotImplementedError

    def get_metadata_for_filter(self, user_id: str | None = None) -> dict[str, list[str]]:
        merged = {key: list(values) for key, values in self._global_cache.items()}
        if user_id and user_id in self._user_cache:
            for key, values in self._user_cache[user_id].items():
                merged.setdefault(key, [])
                merged[key] = sorted(set(merged[key]) | set(values))
        return merged


class PostgresDocumentRepository(BaseDocumentRepository):
    def __init__(self, dsn: str, metadata_cache: MetadataDictionaryCache | None = None) -> None:
        if not isinstance(dsn, str) or not dsn.strip():
            raise ValueError("PostgreSQL DSN must be text")
        self.dsn = dsn.strip()
        self.metadata_cache = metadata_cache

    def _connect(self):
        import psycopg

        return psycopg.connect(self.dsn)

    @contextmanager
    def _wrap_database_errors(self) -> Iterator[None]:
        try:
            yield
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"PostgreSQL operation failed: {exc}") from exc

    def create_schema(self) -> None:
        with self._wrap_database_errors():
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        doc_id TEXT PRIMARY KEY,
                        dish_name TEXT NOT NULL,
                        category TEXT NOT NULL,
                        difficulty TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source TEXT NOT NULL,
                        data_source TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        is_dish_index BOOLEAN NOT NULL DEFAULT FALSE,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_documents_source_user
                        ON documents(data_source, user_id);
                    CREATE INDEX IF NOT EXISTS idx_documents_metadata
                        ON documents(category, difficulty, dish_name);
                    """
                )

    def upsert_documents(self, documents: list[ParsedDocument]) -> None:
        if not documents:
            return
        rows = [asdict(self._normalize_document_for_storage(document)) for document in documents]
        with self._wrap_database_errors():
            with self._connect() as conn, conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO documents (
                        doc_id, dish_name, category, difficulty, content, source,
                        data_source, source_type, user_id, is_dish_index, updated_at
                    ) VALUES (
                        %(doc_id)s, %(dish_name)s, %(category)s, %(difficulty)s,
                        %(content)s, %(source)s, %(data_source)s, %(source_type)s,
                        %(user_id)s, %(is_dish_index)s, NOW()
                    )
                    ON CONFLICT (doc_id) DO UPDATE SET
                        dish_name = EXCLUDED.dish_name,
                        category = EXCLUDED.category,
                        difficulty = EXCLUDED.difficulty,
                        content = EXCLUDED.content,
                        source = EXCLUDED.source,
                        data_source = EXCLUDED.data_source,
                        source_type = EXCLUDED.source_type,
                        user_id = EXCLUDED.user_id,
                        is_dish_index = EXCLUDED.is_dish_index,
                        updated_at = NOW()
                    """,
                    rows,
                )
        self.init_all_metadata_cache()

    def list_documents(self, data_source: str | None = None) -> list[ParsedDocument]:
        normalized_data_source = self._normalize_data_source_filter(data_source)
        where = "WHERE data_source = %s" if normalized_data_source else ""
        params = (normalized_data_source,) if normalized_data_source else ()
        with self._wrap_database_errors():
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT doc_id, dish_name, category, difficulty, content, source,
                           data_source, source_type, user_id, is_dish_index
                    FROM documents {where}
                    ORDER BY source
                    """,
                    params,
                )
                rows = cur.fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_parent_documents(self, parent_ids: list[str]) -> dict[str, ParsedDocument]:
        if not isinstance(parent_ids, list):
            raise ValueError("parent_ids must be a list")
        if not parent_ids:
            return {}
        normalized_parent_ids = [
            self._required_document_text(parent_id, "parent_id") for parent_id in parent_ids
        ]
        placeholders = ",".join(["%s"] * len(normalized_parent_ids))
        with self._wrap_database_errors():
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT doc_id, dish_name, category, difficulty, content, source,
                           data_source, source_type, user_id, is_dish_index
                    FROM documents
                    WHERE doc_id IN ({placeholders})
                    """,
                    tuple(normalized_parent_ids),
                )
                rows = cur.fetchall()
        return {document.doc_id: document for document in map(self._row_to_document, rows)}

    def init_all_metadata_cache(self) -> None:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT user_id, category, dish_name, difficulty
                    FROM documents
                    """
                )
                rows = cur.fetchall()
        except Exception as exc:
            if self.metadata_cache:
                try:
                    snapshot = self.metadata_cache.load()
                except Exception:
                    logger.exception("metadata dictionary cache load failed")
                    snapshot = None
                if snapshot:
                    try:
                        self._apply_metadata_cache_snapshot(snapshot)
                    except ValueError:
                        logger.exception("metadata dictionary cache snapshot invalid")
                    else:
                        return
            raise RuntimeError(f"PostgreSQL operation failed: {exc}") from exc

        global_values: dict[str, set[str]] = defaultdict(set)
        user_values: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for row in rows:
            user_id, category, dish_name, difficulty = self._normalize_metadata_database_row(row)
            bucket = global_values if user_id == "GLOBAL" else user_values[user_id]
            bucket["category"].add(category)
            bucket["dish_name"].add(dish_name)
            bucket["difficulty"].add(difficulty)
        self.__class__._global_cache = {key: sorted(values) for key, values in global_values.items()}
        self.__class__._user_cache = {
            user_id: {key: sorted(values) for key, values in values_by_key.items()}
            for user_id, values_by_key in user_values.items()
        }
        if self.metadata_cache:
            try:
                self.metadata_cache.save(self.__class__._global_cache, self.__class__._user_cache)
            except Exception:
                logger.exception("metadata dictionary cache write failed")

    def _apply_metadata_cache_snapshot(self, snapshot: dict[str, dict]) -> None:
        if not isinstance(snapshot, dict):
            raise ValueError("metadata cache snapshot must be an object")
        global_cache = self._normalize_metadata_snapshot_bucket(
            snapshot.get("global", {})
        )
        raw_user_cache = snapshot.get("user", {})
        if not isinstance(raw_user_cache, dict):
            raise ValueError("metadata cache user snapshot must be an object")
        user_cache: dict[str, dict[str, list[str]]] = {}
        for user_id, values_by_key in raw_user_cache.items():
            normalized_user_id = self._required_metadata_text(user_id, "user_id")
            user_cache[normalized_user_id] = self._normalize_metadata_snapshot_bucket(
                values_by_key
            )
        self.__class__._global_cache = global_cache
        self.__class__._user_cache = user_cache

    @staticmethod
    def _normalize_document_for_storage(document: ParsedDocument) -> ParsedDocument:
        doc_id = PostgresDocumentRepository._required_document_text(document.doc_id, "doc_id")
        dish_name = PostgresDocumentRepository._required_document_text(document.dish_name, "dish_name")
        category = PostgresDocumentRepository._required_document_text(document.category, "category")
        difficulty = PostgresDocumentRepository._required_document_text(document.difficulty, "difficulty")
        content = PostgresDocumentRepository._required_document_text(document.content, "content")
        source = PostgresDocumentRepository._required_document_text(document.source, "source")
        data_source = PostgresDocumentRepository._required_document_text(document.data_source, "data_source")
        if data_source not in DOCUMENT_DATA_SOURCES:
            raise ValueError(f"unknown document source: {data_source}")
        if not isinstance(document.source_type, str):
            raise ValueError("source_type must be text")
        source_type = document.source_type.strip() or data_source
        if source_type != data_source:
            raise ValueError("source_type must match data_source")
        user_id = PostgresDocumentRepository._required_document_text(document.user_id, "user_id")
        if data_source == "personal" and user_id == GLOBAL_USER_ID:
            raise ValueError("personal documents require a non-GLOBAL user_id")
        if data_source == "recipes" and user_id != GLOBAL_USER_ID:
            raise ValueError("recipe documents must use GLOBAL user_id")
        if not isinstance(document.is_dish_index, bool):
            raise ValueError("is_dish_index must be a boolean")
        return replace(
            document,
            doc_id=doc_id,
            dish_name=dish_name,
            category=category,
            difficulty=difficulty,
            content=content,
            source=source,
            data_source=data_source,
            source_type=source_type,
            user_id=user_id,
        )

    @staticmethod
    def _normalize_data_source_filter(data_source: str | None) -> str | None:
        if data_source is None:
            return None
        normalized_data_source = PostgresDocumentRepository._required_document_text(
            data_source, "data_source"
        )
        if normalized_data_source not in DOCUMENT_DATA_SOURCES:
            raise ValueError(f"unknown document source: {normalized_data_source}")
        return normalized_data_source

    @staticmethod
    def _required_document_text(value: object, field_name: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be text")
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} is required")
        return normalized

    @staticmethod
    def _normalize_metadata_database_row(row: object) -> tuple[str, str, str, str]:
        try:
            user_id, category, dish_name, difficulty = row
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "metadata row must contain user_id, category, dish_name and difficulty"
            ) from exc
        return (
            PostgresDocumentRepository._required_metadata_text(user_id, "user_id"),
            PostgresDocumentRepository._required_metadata_text(category, "category"),
            PostgresDocumentRepository._required_metadata_text(dish_name, "dish_name"),
            PostgresDocumentRepository._required_metadata_text(difficulty, "difficulty"),
        )

    @staticmethod
    def _required_metadata_text(value: object, field: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"metadata field {field} must be text")
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"metadata field {field} contains a blank value")
        return normalized

    @staticmethod
    def _normalize_metadata_snapshot_bucket(bucket: object) -> dict[str, list[str]]:
        if not isinstance(bucket, dict):
            raise ValueError("metadata cache bucket must be an object")
        normalized: dict[str, list[str]] = {}
        for field, values in bucket.items():
            if field not in METADATA_DICTIONARY_FIELDS:
                raise ValueError(f"metadata cache field is not supported: {field}")
            if not isinstance(values, list):
                raise ValueError(f"metadata cache field {field} must be a list")
            clean_values: list[str] = []
            for value in values:
                if not isinstance(value, str):
                    raise ValueError(
                        f"metadata cache field {field} contains a non-string value"
                    )
                normalized_value = value.strip()
                if not normalized_value:
                    raise ValueError(
                        f"metadata cache field {field} contains a blank value"
                    )
                clean_values.append(normalized_value)
            normalized[str(field)] = sorted(set(clean_values))
        return normalized

    @staticmethod
    def _row_to_document(row) -> ParsedDocument:
        try:
            (
                doc_id,
                dish_name,
                category,
                difficulty,
                content,
                source,
                data_source,
                source_type,
                user_id,
                is_dish_index,
            ) = row
        except (TypeError, ValueError) as exc:
            raise ValueError("document row must contain all document fields") from exc
        doc_id = PostgresDocumentRepository._required_document_text(doc_id, "doc_id")
        dish_name = PostgresDocumentRepository._required_document_text(dish_name, "dish_name")
        category = PostgresDocumentRepository._required_document_text(category, "category")
        difficulty = PostgresDocumentRepository._required_document_text(difficulty, "difficulty")
        content = PostgresDocumentRepository._required_document_text(content, "content")
        source = PostgresDocumentRepository._required_document_text(source, "source")
        data_source = PostgresDocumentRepository._required_document_text(data_source, "data_source")
        if data_source not in DOCUMENT_DATA_SOURCES:
            raise ValueError(f"unknown document source: {data_source}")
        source_type = PostgresDocumentRepository._required_document_text(source_type, "source_type")
        if source_type != data_source:
            raise ValueError("source_type must match data_source")
        user_id = PostgresDocumentRepository._required_document_text(user_id, "user_id")
        if data_source == "personal" and user_id == GLOBAL_USER_ID:
            raise ValueError("personal documents require a non-GLOBAL user_id")
        if data_source == "recipes" and user_id != GLOBAL_USER_ID:
            raise ValueError("recipe documents must use GLOBAL user_id")
        if not isinstance(is_dish_index, bool):
            raise ValueError("is_dish_index must be a boolean")
        return ParsedDocument(
            doc_id=doc_id,
            dish_name=dish_name,
            category=category,
            difficulty=difficulty,
            content=content,
            source=source,
            data_source=data_source,
            source_type=source_type,
            user_id=user_id,
            is_dish_index=is_dish_index,
        )


DocumentRepository = PostgresDocumentRepository
