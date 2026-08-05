from __future__ import annotations

from app.core.llm import OpenAICompatibleClient


class GenerationIntegrationModule:
    def __init__(self, llm_client: OpenAICompatibleClient) -> None:
        self.llm_client = llm_client

    async def rewrite_query(self, query: str) -> str:
        system_prompt = (
            "你是饮食健康平台的查询改写模块。只输出改写后的中文自然语言查询，"
            "不要解释。改写要保留菜名、食材、难度、口味等硬约束。"
        )
        user_prompt = f"原始查询：{query}"
        rewritten = await self.llm_client.complete_text(
            system_prompt, user_prompt, model="fast", temperature=0.1
        )
        return rewritten.strip().strip("\"'") or query
