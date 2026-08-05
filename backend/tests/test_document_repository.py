import sys
from types import SimpleNamespace

import pytest
from app.database.document_repository import (
    PostgresDocumentRepository,
    RedisMetadataDictionaryCache,
)
from app.rag.document import ParsedDocument


@pytest.mark.parametrize("dsn", ["", "   ", 123])
def test_postgres_document_repository_rejects_invalid_dsn(dsn) -> None:
    with pytest.raises(ValueError, match="PostgreSQL DSN must be text"):
        PostgresDocumentRepository(dsn)


@pytest.mark.parametrize("redis_url", ["", "   ", 123])
def test_redis_metadata_dictionary_cache_rejects_invalid_redis_url_before_client_creation(
    redis_url,
    monkeypatch,
) -> None:
    calls = []

    class FakeRedisModule:
        @staticmethod
        def from_url(value: str):
            calls.append(value)
            raise AssertionError("invalid Redis URL should not create Redis client")

    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=FakeRedisModule))

    with pytest.raises(ValueError, match="Redis metadata cache URL must be text"):
        RedisMetadataDictionaryCache(redis_url)

    assert calls == []


class FakeCursor:
    def __init__(self) -> None:
        self.executed_many = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def executemany(self, sql, params):
        self.executed_many = params


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.cursor_obj = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self.cursor_obj


def test_postgres_repository_upserts_slot_parsed_documents(monkeypatch) -> None:
    cursor = FakeCursor()
    repository = PostgresDocumentRepository("postgresql://example")
    monkeypatch.setattr(repository, "_connect", lambda: FakeConnection(cursor))
    monkeypatch.setattr(repository, "init_all_metadata_cache", lambda: None)

    repository.upsert_documents([
        ParsedDocument(
            doc_id="doc-1",
            dish_name="番茄炒蛋",
            category="素菜",
            difficulty="简单",
            content="# 番茄炒蛋",
            source="vegetable_dish/番茄炒蛋.md",
        )
    ])

    assert cursor.executed_many[0]["doc_id"] == "doc-1"
    assert cursor.executed_many[0]["user_id"] == "GLOBAL"


def test_postgres_repository_rejects_invalid_document_before_database_connect(monkeypatch) -> None:
    repository = PostgresDocumentRepository("postgresql://example")

    def fail_connect():
        raise AssertionError("invalid documents must not touch PostgreSQL")

    monkeypatch.setattr(repository, "_connect", fail_connect)

    with pytest.raises(ValueError, match="doc_id is required"):
        repository.upsert_documents([
            ParsedDocument(
                doc_id="   ",
                dish_name="番茄炒蛋",
                category="素菜",
                difficulty="简单",
                content="# 番茄炒蛋",
                source="vegetable_dish/番茄炒蛋.md",
            )
        ])


def test_postgres_repository_rejects_non_text_source_type_before_database_connect(monkeypatch) -> None:
    repository = PostgresDocumentRepository("postgresql://example")

    def fail_connect():
        raise AssertionError("invalid documents must not touch PostgreSQL")

    monkeypatch.setattr(repository, "_connect", fail_connect)

    with pytest.raises(ValueError, match="source_type must be text"):
        repository.upsert_documents([
            ParsedDocument(
                doc_id="doc-1",
                dish_name="番茄炒蛋",
                category="素菜",
                difficulty="简单",
                content="# 番茄炒蛋",
                source="vegetable_dish/番茄炒蛋.md",
                source_type=123,
            )
        ])


def test_postgres_repository_rejects_non_text_parent_ids_before_database_connect(monkeypatch) -> None:
    repository = PostgresDocumentRepository("postgresql://example")

    def fail_connect():
        raise AssertionError("invalid parent ids must not touch PostgreSQL")

    monkeypatch.setattr(repository, "_connect", fail_connect)

    with pytest.raises(ValueError, match="parent_id must be text"):
        repository.get_parent_documents([123])


class MetadataCursor(FakeCursor):
    def __init__(self, rows=None, error: Exception | None = None) -> None:
        super().__init__()
        self.rows = rows or []
        self.error = error

    def execute(self, sql, params=None):
        if self.error:
            raise self.error

    def fetchall(self):
        return self.rows


class FakeMetadataCache:
    def __init__(self, snapshot=None) -> None:
        self.saved = None
        self.snapshot = snapshot

    def load(self):
        return self.snapshot

    def save(self, global_cache, user_cache) -> None:
        self.saved = {"global": global_cache, "user": user_cache}


def test_metadata_dictionary_is_mirrored_to_cache(monkeypatch) -> None:
    cursor = MetadataCursor(rows=[
        ("GLOBAL", "素菜", "番茄炒蛋", "简单"),
        ("u1", "个人饮食", "训练日晚餐偏好", "普通"),
    ])
    cache = FakeMetadataCache()
    repository = PostgresDocumentRepository("postgresql://example", metadata_cache=cache)
    monkeypatch.setattr(repository, "_connect", lambda: FakeConnection(cursor))

    repository.init_all_metadata_cache()

    assert repository.get_metadata_for_filter("u1") == {
        "category": ["个人饮食", "素菜"],
        "dish_name": ["番茄炒蛋", "训练日晚餐偏好"],
        "difficulty": ["普通", "简单"],
    }
    assert cache.saved == {
        "global": {"category": ["素菜"], "dish_name": ["番茄炒蛋"], "difficulty": ["简单"]},
        "user": {"u1": {"category": ["个人饮食"], "dish_name": ["训练日晚餐偏好"], "difficulty": ["普通"]}},
    }


def test_metadata_dictionary_can_fall_back_to_cache_when_database_unavailable(monkeypatch) -> None:
    snapshot = {
        "global": {"category": ["素菜"], "dish_name": ["番茄炒蛋"], "difficulty": ["简单"]},
        "user": {},
    }
    cache = FakeMetadataCache(snapshot=snapshot)
    cursor = MetadataCursor(error=RuntimeError("database unavailable"))
    repository = PostgresDocumentRepository("postgresql://example", metadata_cache=cache)
    monkeypatch.setattr(repository, "_connect", lambda: FakeConnection(cursor))

    repository.init_all_metadata_cache()

    assert repository.get_metadata_for_filter() == snapshot["global"]


def test_malformed_metadata_cache_snapshot_is_not_applied_when_database_unavailable(
    monkeypatch,
) -> None:
    PostgresDocumentRepository._global_cache = {}
    PostgresDocumentRepository._user_cache = {}
    snapshot = {
        "global": {
            "category": "素菜",
            "dish_name": ["番茄炒蛋"],
            "difficulty": ["简单"],
        },
        "user": {},
    }
    cache = FakeMetadataCache(snapshot=snapshot)
    cursor = MetadataCursor(error=RuntimeError("database unavailable"))
    repository = PostgresDocumentRepository("postgresql://example", metadata_cache=cache)
    monkeypatch.setattr(repository, "_connect", lambda: FakeConnection(cursor))

    with pytest.raises(RuntimeError, match="PostgreSQL operation failed"):
        repository.init_all_metadata_cache()

    assert repository.get_metadata_for_filter() == {}


def test_metadata_cache_snapshot_rejects_non_text_user_ids_when_database_unavailable(
    monkeypatch,
) -> None:
    PostgresDocumentRepository._global_cache = {}
    PostgresDocumentRepository._user_cache = {}
    snapshot = {
        "global": {"category": ["素菜"], "dish_name": ["番茄炒蛋"], "difficulty": ["简单"]},
        "user": {
            123: {"category": ["个人饮食"], "dish_name": ["训练日晚餐偏好"], "difficulty": ["普通"]}
        },
    }
    cache = FakeMetadataCache(snapshot=snapshot)
    cursor = MetadataCursor(error=RuntimeError("database unavailable"))
    repository = PostgresDocumentRepository("postgresql://example", metadata_cache=cache)
    monkeypatch.setattr(repository, "_connect", lambda: FakeConnection(cursor))

    with pytest.raises(RuntimeError, match="PostgreSQL operation failed"):
        repository.init_all_metadata_cache()

    assert repository.get_metadata_for_filter("123") == {}


class FailingSaveMetadataCache(FakeMetadataCache):
    def save(self, global_cache, user_cache) -> None:
        raise RuntimeError("redis unavailable")


def test_metadata_cache_write_failure_does_not_block_database_loaded_metadata(monkeypatch) -> None:
    cursor = MetadataCursor(rows=[("GLOBAL", "素菜", "番茄炒蛋", "简单")])
    repository = PostgresDocumentRepository(
        "postgresql://example", metadata_cache=FailingSaveMetadataCache()
    )
    monkeypatch.setattr(repository, "_connect", lambda: FakeConnection(cursor))

    repository.init_all_metadata_cache()

    assert repository.get_metadata_for_filter() == {
        "category": ["素菜"],
        "dish_name": ["番茄炒蛋"],
        "difficulty": ["简单"],
    }


class RecordingMetadataCursor(MetadataCursor):
    def __init__(self, rows=None, error: Exception | None = None) -> None:
        super().__init__(rows=rows, error=error)
        self.executed_sql = ""

    def execute(self, sql, params=None):
        self.executed_sql = str(sql)
        super().execute(sql, params)


def test_metadata_dictionary_query_uses_postgres_distinct(monkeypatch) -> None:
    cursor = RecordingMetadataCursor(rows=[])
    repository = PostgresDocumentRepository("postgresql://example")
    monkeypatch.setattr(repository, "_connect", lambda: FakeConnection(cursor))

    repository.init_all_metadata_cache()

    normalized_sql = " ".join(cursor.executed_sql.split())
    assert "SELECT DISTINCT user_id, category, dish_name, difficulty" in normalized_sql


def test_metadata_dictionary_rejects_blank_database_metadata_values(monkeypatch) -> None:
    PostgresDocumentRepository._global_cache = {}
    PostgresDocumentRepository._user_cache = {}
    cursor = MetadataCursor(rows=[("GLOBAL", "   ", "番茄炒蛋", "简单")])
    cache = FakeMetadataCache()
    repository = PostgresDocumentRepository("postgresql://example", metadata_cache=cache)
    monkeypatch.setattr(repository, "_connect", lambda: FakeConnection(cursor))

    with pytest.raises(ValueError, match="metadata field category contains a blank value"):
        repository.init_all_metadata_cache()

    assert repository.get_metadata_for_filter() == {}
    assert cache.saved is None


class DatabaseUnavailable(Exception):
    pass


def test_postgres_document_repository_wraps_database_failures_for_api_error_mapping(monkeypatch) -> None:
    repository = PostgresDocumentRepository("postgresql://example")

    def fail_connect():
        raise DatabaseUnavailable("postgres down")

    monkeypatch.setattr(repository, "_connect", fail_connect)

    operations = [
        lambda: repository.create_schema(),
        lambda: repository.upsert_documents([
            ParsedDocument(
                doc_id="doc-1",
                dish_name="番茄炒蛋",
                category="素菜",
                difficulty="简单",
                content="# 番茄炒蛋",
                source="vegetable_dish/番茄炒蛋.md",
            )
        ]),
        lambda: repository.list_documents(),
        lambda: repository.get_parent_documents(["doc-1"]),
    ]

    for operation in operations:
        with pytest.raises(RuntimeError, match="PostgreSQL operation failed") as exc_info:
            operation()
        assert isinstance(exc_info.value.__cause__, DatabaseUnavailable)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            ("   ", "番茄炒蛋", "素菜", "简单", "# 番茄炒蛋", "vegetable_dish/番茄炒蛋.md", "recipes", "recipes", "GLOBAL", False),
            "doc_id is required",
        ),
        (
            ("doc-1", "番茄炒蛋", "素菜", "简单", "# 番茄炒蛋", "vegetable_dish/番茄炒蛋.md", "recipes", "   ", "GLOBAL", False),
            "source_type is required",
        ),
        (
            ("doc-1", "番茄炒蛋", "素菜", "简单", "# 番茄炒蛋", "vegetable_dish/番茄炒蛋.md", "recipes", "recipes", "u1", False),
            "recipe documents must use GLOBAL user_id",
        ),
        (
            ("doc-1", "训练日晚餐偏好", "个人饮食", "普通", "# 偏好", "personal/u1.md", "personal", "personal", "GLOBAL", False),
            "personal documents require a non-GLOBAL user_id",
        ),
        (
            ("doc-1", "番茄炒蛋", "素菜", "简单", "# 番茄炒蛋", "vegetable_dish/番茄炒蛋.md", "recipes", "recipes", "GLOBAL", 1),
            "is_dish_index must be a boolean",
        ),
    ],
)
def test_postgres_repository_rejects_malformed_document_rows_from_database(monkeypatch, row, message) -> None:
    cursor = MetadataCursor(rows=[row])
    repository = PostgresDocumentRepository("postgresql://example")
    monkeypatch.setattr(repository, "_connect", lambda: FakeConnection(cursor))

    with pytest.raises(ValueError, match=message):
        repository.list_documents()
