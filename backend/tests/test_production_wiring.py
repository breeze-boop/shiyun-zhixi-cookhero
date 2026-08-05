import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.llm import LLMConfigurationError
from app.main import build_application_state
from app.database.document_repository import PostgresDocumentRepository, RedisMetadataDictionaryCache
from app.database.user_repository import UserRepository
from app.rag.cache.backends import MilvusVectorCache, RedisKeywordCache
from app.rag.pipeline.retrieval import MilvusHybridRetriever
from app.services.diet_service import DietService


def test_application_state_uses_real_external_adapters() -> None:
    settings = Settings(
        postgres_dsn="postgresql://cookhero:cookhero@localhost:5432/cookhero",
        redis_url="redis://localhost:6379/0",
        milvus_uri="http://localhost:19530",
        llm_api_key="test-key",
        llm_base_url="https://api.example.test/v1",
        embedding_api_key="test-key",
        embedding_base_url="https://api.example.test/v1",
        siliconflow_api_key="test-key",
        rerank_min_score=0.2,
        auto_seed_howtocook_data=False,
    )

    state = build_application_state(settings)

    assert isinstance(state.repository, PostgresDocumentRepository)
    assert isinstance(state.repository.metadata_cache, RedisMetadataDictionaryCache)
    assert isinstance(state.user_repository, UserRepository)
    assert isinstance(state.diet_service, DietService)
    assert isinstance(state.rag_service.retrieval, MilvusHybridRetriever)
    assert state.rag_service.reranker.min_score == 0.2
    assert state.rag_service.retrieval.ranker_strategy == settings.ranker_strategy
    assert isinstance(state.rag_service.cache_manager.keyword_backend, RedisKeywordCache)
    assert isinstance(state.rag_service.cache_manager.vector_backend, MilvusVectorCache)



def test_application_state_degrades_rerank_when_siliconflow_key_is_missing() -> None:
    settings = Settings(
        postgres_dsn="postgresql://cookhero:cookhero@localhost:5432/cookhero",
        redis_url="redis://localhost:6379/0",
        milvus_uri="http://localhost:19530",
        llm_api_key="test-key",
        embedding_api_key="test-key",
        siliconflow_api_key=None,
        rerank_enabled=True,
        auto_seed_howtocook_data=False,
    )

    state = build_application_state(settings)

    assert state.rag_service.reranker.enabled is False


def test_application_state_registers_food_image_tool_when_modelscope_key_is_configured() -> None:
    settings = Settings(
        postgres_dsn="postgresql://cookhero:cookhero@localhost:5432/cookhero",
        redis_url="redis://localhost:6379/0",
        milvus_uri="http://localhost:19530",
        llm_api_key="test-key",
        embedding_api_key="test-key",
        siliconflow_api_key="test-key",
        modelscope_api_key="modelscope-key",
        auto_seed_howtocook_data=False,
    )

    state = build_application_state(settings)

    tools = {tool["name"]: tool for tool in state.tool_registry.list_tools()}
    assert tools["food_image_analysis"]["provider"] == "local"
    assert "image_url" in tools["meal_checkin"]["input_schema"]["properties"]


@pytest.mark.asyncio
async def test_auto_seed_uses_howtocook_and_never_sample_recipes(tmp_path, monkeypatch) -> None:
    from app.main import seed_local_knowledge

    sample_recipe = (
        tmp_path
        / "data"
        / "sample_recipes"
        / "dishes"
        / "vegetable_dish"
        / "样例菜.md"
    )
    sample_recipe.parent.mkdir(parents=True)
    sample_recipe.write_text("# 样例菜\n\n预估烹饪难度：★★\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    class FakeRAGService:
        def __init__(self) -> None:
            self.index_calls = 0

        async def index_parsed_documents(self, documents):
            self.index_calls += 1

    state = type("State", (), {"rag_service": FakeRAGService()})()

    with pytest.raises(FileNotFoundError, match="HowToCook dishes directory not found"):
        await seed_local_knowledge(state)

    assert state.rag_service.index_calls == 0


@pytest.mark.asyncio
async def test_ingest_howtocook_rejects_sample_recipes_before_building_state(
    tmp_path, monkeypatch
) -> None:
    from scripts import ingest_howtocook

    sample_recipe = (
        tmp_path
        / "data"
        / "sample_recipes"
        / "dishes"
        / "vegetable_dish"
        / "样例菜.md"
    )
    sample_recipe.parent.mkdir(parents=True)
    sample_recipe.write_text("# 样例菜\n\n预估烹饪难度：★★\n", encoding="utf-8")

    def fail_if_state_is_built(_settings):
        raise AssertionError("application state was built before validating the HowToCook source")

    monkeypatch.setattr(ingest_howtocook, "build_application_state", fail_if_state_is_built)

    with pytest.raises(ValueError, match="sample_recipes"):
        await ingest_howtocook.ingest(sample_recipe.parents[1])


def test_ranker_strategy_rejects_unknown_values() -> None:
    import pytest
    from pydantic import ValidationError

    assert Settings(
        llm_api_key="test-key",
        embedding_api_key="test-key",
        ranker_strategy="RRF",
    ).ranker_strategy == "rrf"

    with pytest.raises(ValidationError, match="RANKER_STRATEGY"):
        Settings(
            llm_api_key="test-key",
            embedding_api_key="test-key",
            ranker_strategy="dense_only",
        )


def test_infrastructure_defaults_match_settings_and_env_example() -> None:
    from pathlib import Path

    compose = Path("../infra/docker-compose.yml").read_text(encoding="utf-8")
    readme = Path("../README.md").read_text(encoding="utf-8")
    env_lines = Path(".env.example").read_text(encoding="utf-8").splitlines()
    env = dict(line.split("=", 1) for line in env_lines if line and not line.startswith("#"))
    settings = Settings(llm_api_key="test-key", embedding_api_key="test-key")

    assert "AUTO_SEED_HOWTOCOOK_DATA" in readme
    assert "AUTO_SEED_SAMPLE_DATA" not in readme
    assert "AUTO_SEED_HOWTOCOOK_DATA" in env
    assert "AUTO_SEED_SAMPLE_DATA" not in env
    assert settings.auto_seed_howtocook_data is False
    assert Settings(
        llm_api_key="test-key",
        embedding_api_key="test-key",
        auto_seed_howtocook_data=True,
    ).auto_seed_howtocook_data is True

    assert "postgres:" in compose
    assert "redis:" in compose
    assert "milvus:" in compose
    assert "etcd:" in compose
    assert "minio:" in compose
    assert '"5432:5432"' in compose
    assert '"6379:6379"' in compose
    assert '"19530:19530"' in compose
    assert "ETCD_ENDPOINTS: etcd:2379" in compose
    assert "MINIO_ADDRESS: minio:9000" in compose
    assert env["POSTGRES_DSN"] == settings.postgres_dsn
    assert env["REDIS_URL"] == settings.redis_url
    assert env["MILVUS_URI"] == settings.milvus_uri


def test_env_example_placeholder_api_keys_are_not_accepted_as_real_credentials() -> None:
    settings = Settings(
        llm_api_key="replace-with-qwen-or-deepseek-compatible-key",
        embedding_api_key="replace-with-embedding-key",
        modelscope_api_key="replace-with-modelscope-key",
        siliconflow_api_key="replace-with-siliconflow-key",
    )

    with pytest.raises(LLMConfigurationError, match="LLM_API_KEY is required"):
        build_application_state(settings)

    assert settings.llm_api_key is None
    assert settings.embedding_api_key is None
    assert settings.modelscope_api_key is None
    assert settings.siliconflow_api_key is None


@pytest.mark.parametrize(
    "field_name",
    [
        "llm_api_key",
        "embedding_api_key",
        "modelscope_api_key",
        "siliconflow_api_key",
    ],
)
def test_api_key_settings_reject_non_string_values_before_external_calls(
    field_name: str,
) -> None:
    values = {"llm_api_key": "test-key", "embedding_api_key": "test-key"}
    values[field_name] = 123

    with pytest.raises(ValidationError):
        Settings(**values)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("embedding_dim", 0),
        ("default_top_k", 0),
        ("fetch_multiplier", 0),
        ("rrf_k", 0),
        ("cache_ttl_seconds", 0),
        ("l2_similarity_threshold", 0.0),
        ("l2_similarity_threshold", 1.5),
        ("rerank_min_score", -0.01),
        ("rerank_min_score", 1.01),
    ],
)
def test_rag_runtime_numeric_settings_reject_values_that_break_production_wiring(
    field_name: str, invalid_value: int | float
) -> None:
    with pytest.raises(ValidationError):
        Settings(
            llm_api_key="test-key",
            embedding_api_key="test-key",
            **{field_name: invalid_value},
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "llm_provider",
        "llm_base_url",
        "llm_model_fast",
        "llm_model_reasoning",
        "embedding_base_url",
        "embedding_model",
        "modelscope_base_url",
        "vision_model",
        "siliconflow_rerank_model",
    ],
)
def test_runtime_text_settings_reject_blank_values_before_external_calls(field_name: str) -> None:
    with pytest.raises(ValidationError):
        Settings(
            llm_api_key="test-key",
            embedding_api_key="test-key",
            **{field_name: "   "},
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "postgres_dsn",
        "redis_url",
        "milvus_uri",
        "recipes_collection",
        "personal_collection",
        "retrieval_cache_collection",
    ],
)
def test_infrastructure_text_settings_reject_blank_values_before_adapter_construction(
    field_name: str,
) -> None:
    with pytest.raises(ValidationError):
        Settings(
            llm_api_key="test-key",
            embedding_api_key="test-key",
            **{field_name: "   "},
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "postgres_dsn",
        "redis_url",
        "milvus_uri",
        "llm_provider",
        "llm_base_url",
        "llm_model_fast",
        "llm_model_reasoning",
        "embedding_base_url",
        "embedding_model",
        "modelscope_base_url",
        "vision_model",
        "siliconflow_rerank_model",
        "recipes_collection",
        "personal_collection",
        "retrieval_cache_collection",
    ],
)
def test_runtime_text_settings_reject_non_string_values_before_external_calls(
    field_name: str,
) -> None:
    with pytest.raises(ValidationError):
        Settings(
            llm_api_key="test-key",
            embedding_api_key="test-key",
            **{field_name: 123},
        )


def test_llm_provider_rejects_unsupported_values_before_building_clients() -> None:
    with pytest.raises(ValidationError, match="LLM_PROVIDER"):
        Settings(
            llm_api_key="test-key",
            embedding_api_key="test-key",
            llm_provider="local-stub",
        )


def test_blank_optional_mcp_config_is_treated_as_disabled() -> None:
    settings = Settings(
        llm_api_key="test-key",
        embedding_api_key="test-key",
        mcp_servers_config="   ",
    )

    assert settings.mcp_servers_config is None
