
import json

import httpx

from app.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    CATALOGUE_PATH,
)


def download_catalogue() -> int:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing from backend/.env"
        )

    url = f"{OPENROUTER_BASE_URL.rstrip('/')}/models"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    }

    response = httpx.get(
        url,
        headers=headers,
        params={"limit": 1000},
        timeout=60.0,
    )
    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload.get("data"), list):
        raise ValueError("Unexpected OpenRouter catalogue response")

    CATALOGUE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save the complete API response, including pagination metadata.
    CATALOGUE_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    count = len(payload["data"])

    print(f"Saved {count} models to {CATALOGUE_PATH}")
    print(f"Has more pages: {payload.get('has_more', 'unknown')}")

    return count


if __name__ == "__main__":
    download_catalogue()
