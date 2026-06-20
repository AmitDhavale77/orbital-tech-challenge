from __future__ import annotations

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://orbital:orbital@db:5432/orbital_takehome"
    anthropic_api_key: str = ""
    upload_dir: str = "uploads"
    max_upload_size: int = 25 * 1024 * 1024  # 25MB

    # --- Models ---------------------------------------------------------- #

    qa_model: str = "claude-sonnet-4-6"
    card_model: str = "anthropic:claude-haiku-4-5-20251001"
    map_model: str = "anthropic:claude-sonnet-4-6"
    reduce_model: str = "anthropic:claude-sonnet-4-6"

    # --- Token / consumption knobs --------------------------------------- #

    chat_total_tokens_limit: int = 600_000
    compaction_token_threshold: int = 300_000
    agent_retries: int = 2
    # Cap concurrent per-doc map agents so a 50-doc bundle doesn't open 50 LLM
    map_concurrency: int = 5
    # A sample is enough for a routing card; keeps the call cheap.
    card_sample_chars: int = 6000

    model_config = {"env_file": ".env"}


settings = Settings()

# Ensure the Anthropic API key is available as an environment variable
# so that pydantic-ai's Anthropic integration can pick it up.
if settings.anthropic_api_key:
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
