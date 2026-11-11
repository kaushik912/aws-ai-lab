"""Thin Bedrock wrappers. boto3 is sync -> run in threads so FastAPI's event loop stays free."""
import asyncio
import json
from collections.abc import AsyncIterator

from .clients import bedrock
from .config import get_settings


def embed_sync(text: str) -> list[float]:
    s = get_settings()
    resp = bedrock().invoke_model(
        modelId=s.bedrock_embed_model,
        body=json.dumps({"inputText": text, "dimensions": s.embed_dims, "normalize": True}),
    )
    return json.loads(resp["body"].read())["embedding"]


async def embed(text: str) -> list[float]:
    return await asyncio.to_thread(embed_sync, text)


def _kwargs(messages: list[dict], system: str, tools: list[dict] | None) -> dict:
    s = get_settings()
    kw: dict = dict(
        modelId=s.bedrock_chat_model,
        messages=messages,
        system=[{"text": system}],
        inferenceConfig={"maxTokens": s.max_output_tokens, "temperature": 0.1},
    )
    if tools:
        kw["toolConfig"] = {"tools": tools}
    return kw


async def converse(messages: list[dict], system: str, tools: list[dict] | None = None) -> dict:
    """Returns the raw Converse response (output.message, stopReason, usage)."""
    return await asyncio.to_thread(bedrock().converse, **_kwargs(messages, system, tools))


async def converse_stream(messages: list[dict], system: str) -> AsyncIterator[dict]:
    """Yields {"text": str} deltas, then a final {"usage": {...}}."""
    resp = await asyncio.to_thread(bedrock().converse_stream, **_kwargs(messages, system, None))
    it = iter(resp["stream"])
    done = object()
    while (ev := await asyncio.to_thread(next, it, done)) is not done:
        if "contentBlockDelta" in ev:
            delta = ev["contentBlockDelta"]["delta"]
            if "text" in delta:
                yield {"text": delta["text"]}
        elif "metadata" in ev:
            yield {"usage": ev["metadata"]["usage"]}
