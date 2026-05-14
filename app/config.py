from functools import lru_cache
import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "MediaMind AI"
    mongodb_url: str = "mongodb://localhost:27017"
    mongo_uri: str | None = None
    mongodb_database: str = "mediamind"
    upload_dir: Path = Path("uploads")
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.mongo_uri and settings.mongodb_url == "mongodb://localhost:27017":
        settings.mongodb_url = settings.mongo_uri
    if not settings.groq_api_key:
        settings.groq_api_key = os.getenv("GROQ__API_KEY")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
