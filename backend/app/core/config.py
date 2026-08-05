from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CookHero"
    api_prefix: str = "/api/v1"
    environment: str = "production"

    postgres_dsn: str = Field(
        default="postgresql://cookhero:cookhero@localhost:5432/cookhero"
    )
    redis_url: str = Field(default="redis://localhost:6379/0")
    milvus_uri: str = Field(default="http://localhost:19530")

    llm_provider: str = Field(default="openai-compatible")
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model_fast: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_model_reasoning: str = "deepseek-ai/DeepSeek-V3"

    embedding_api_key: str | None = None
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_dim: int = Field(default=1024, gt=0)

    modelscope_api_key: str | None = None
    modelscope_base_url: str = "https://api-inference.modelscope.cn/v1"
    vision_model: str = "Qwen/Qwen2.5-VL-72B-Instruct"

    rerank_enabled: bool = True
    siliconflow_api_key: str | None = None
    siliconflow_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_min_score: float = Field(default=0.05, ge=0.0, le=1.0, allow_inf_nan=False)

    cache_enabled: bool = True
    cache_l2_enabled: bool = True
    cache_ttl_seconds: int = Field(default=3600, gt=0)
    l2_similarity_threshold: float = Field(default=0.92, gt=0.0, le=1.0, allow_inf_nan=False)

    default_top_k: int = Field(default=6, gt=0)
    fetch_multiplier: int = Field(default=4, gt=0)
    ranker_strategy: str = "weighted"
    rrf_k: int = Field(default=60, gt=0)
    auto_seed_howtocook_data: bool = False
    mcp_servers_config: str | None = None

    recipes_collection: str = "cook_hero_recipes"
    personal_collection: str = "cook_hero_personal"
    retrieval_cache_collection: str = "cookhero_retrieval_cache"

    @field_validator(
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
        mode="before",
    )
    @classmethod
    def strip_required_runtime_text(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("runtime text setting must be text")
        normalized = value.strip()
        if not normalized:
            raise ValueError("runtime text setting must not be blank")
        return normalized

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        normalized = value.lower()
        if normalized != "openai-compatible":
            raise ValueError("LLM_PROVIDER must be openai-compatible")
        return normalized

    @field_validator(
        "llm_api_key",
        "embedding_api_key",
        "modelscope_api_key",
        "siliconflow_api_key",
        mode="before",
    )
    @classmethod
    def normalize_secret_placeholder(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("API key setting must be text")
        normalized = value.strip()
        if not normalized or normalized.startswith("replace-with-"):
            return None
        return normalized

    @field_validator("mcp_servers_config", mode="before")
    @classmethod
    def normalize_optional_runtime_path(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("runtime path setting must be text")
        normalized = value.strip()
        return normalized or None

    @field_validator("ranker_strategy")
    @classmethod
    def validate_ranker_strategy(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"weighted", "rrf"}:
            raise ValueError("RANKER_STRATEGY must be one of: weighted, rrf")
        return normalized

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
