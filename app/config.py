from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    aws_region: str = "us-east-1"
    # Inference profile id; Claude on Bedrock needs one for on-demand use.
    bedrock_chat_model: str = "us.anthropic.claude-haiku-5-5"
    bedrock_embed_model: str = "amazon.titan-embed-text-v2:0"
    embed_dims: int = 512  # Titan v2: 256 | 512 | 1024. Smaller = smaller index, same price.
    index_dir: str = "index"
    # Grounding gate: best dense cosine below this => decline without an LLM call (saves tokens).
    min_cosine: float = 0.30
    top_k: int = 4
    max_output_tokens: int = 400  # hard cost cap per answer
    max_agent_steps: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
