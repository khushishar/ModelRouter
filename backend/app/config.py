import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
)

DATA_DIR = BACKEND_DIR / "data"
CATALOGUE_PATH = DATA_DIR / "openrouter_models.json"
MODEL_OVERRIDES_PATH = DATA_DIR / "model_overrides.json"
MODEL_PREFERENCES_PATH = DATA_DIR / "model_preferences.json"
ROUTING_LOG_PATH = DATA_DIR / "routing_logs.jsonl"
