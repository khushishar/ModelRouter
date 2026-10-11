
import asyncio
import statistics
import time
from typing import Any

import httpx

from app.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

import json
from datetime import datetime, timezone

from app.config import CATALOGUE_PATH 


async def measure_model_latency(
    client: httpx.AsyncClient,
    model_id: str,
) -> float | None:
    import time

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": "Reply with only the word OK.",
            }
        ],
        "max_tokens": 5,
        "temperature": 0,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    started = time.perf_counter()

    try:
        response = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )

        if response.is_error:
            print(
                f"[LATENCY ERROR] {model_id} | "
                f"HTTP {response.status_code} | "
                f"{response.text[:500]}"
            )
            return None

        data = response.json()

        if not data.get("choices"):
            print(
                f"[LATENCY ERROR] {model_id} | "
                f"No choices returned: {str(data)[:300]}"
            )
            return None

        return round(
            (time.perf_counter() - started) * 1000,
            2,
        )

    except httpx.TimeoutException:
        print(f"[LATENCY ERROR] {model_id} | Request timed out")
        return None

    except httpx.RequestError as exc:
        print(f"[LATENCY ERROR] {model_id} | Network error: {exc}")
        return None

    except ValueError as exc:
        print(f"[LATENCY ERROR] {model_id} | Invalid JSON: {exc}")
        return None



async def benchmark_models(
    models: list[dict[str, Any]],
    repetitions: int = 3,
) -> list[dict[str, Any]]:
    """Measure models and rank them by median latency."""

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")

    results = []

    async with httpx.AsyncClient(timeout=45.0) as client:
        for model in models:
            measurements = []

            for _ in range(repetitions):
                latency = await measure_model_latency(
                    client,
                    model["id"],
                )

                if latency is not None:
                    measurements.append(latency)

                # Avoid hammering providers during benchmarking.
                await asyncio.sleep(0.3)

            median_ms = (
                round(statistics.median(measurements), 2)
                if measurements
                else None
            )

            model["measured_latency_ms"] = median_ms

            results.append({
                "id": model["id"],
                "measurements_ms": measurements,
                "median_latency_ms": median_ms,
                "successful_runs": len(measurements),
            })

    results.sort(
        key=lambda item: (
            item["median_latency_ms"] is None,
            item["median_latency_ms"]
            if item["median_latency_ms"] is not None
            else float("inf"),
        )
    )
    return results



def save_latency_to_catalogue(results):
    with CATALOGUE_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # Match results by exact model ID.
    profiles = {result["id"]: result for result in results}
    timestamp = datetime.now(timezone.utc).isoformat()

    updated = 0

    for model in payload.get("data", []):
        result = profiles.get(model.get("id"))

        if result is None:
            continue

        model["modelrouter_metrics"] = {
            "median_latency_ms": result["median_latency_ms"],
            "measurements_ms": result["measurements_ms"],
            "successful_runs": result["successful_runs"],
            "measured_at": timestamp,
        }
        updated += 1

    with CATALOGUE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Updated latency metrics for {updated} models.")


    return results
