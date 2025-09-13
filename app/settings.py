import os
from functools import lru_cache
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

load_dotenv()

class Settings(BaseModel):
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    hf_token: str | None = os.getenv("HF_TOKEN")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

@lru_cache
def get_settings() -> Settings:
    return Settings()

def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
