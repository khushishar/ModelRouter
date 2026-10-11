import time
from typing import Any

import httpx

from app.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL


class InferenceError(Exception):
    """Raised when every valid candidate fails within the request budget."""

    def __init__(self, message: str, attempts: list[dict[str, Any]]):
        super().__init__(message)
        self.attempts = attempts


async def generate_response(
    messages: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    latency_budget_ms: float = 30_000,
) -> dict[str, Any]:
    """Try ranked, already-filtered candidates under one shared deadline."""
    if not OPENROUTER_API_KEY:
        raise InferenceError(
            "OPENROUTER_API_KEY is not configured.",
            [{"error": "OPENROUTER_API_KEY is not configured"}],
        )
    if not candidates:
        raise InferenceError("No eligible models are available.", [])

    started_at = time.monotonic()
    deadline = started_at + latency_budget_ms / 1000
    attempts: list[dict[str, Any]] = []
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5173",
        "X-Title": "ModelRouter",
    }

    async with httpx.AsyncClient() as client:
        for model in candidates:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                break

            model_id = model["id"]
            attempt_started = time.monotonic()
            payload = {
                "model": model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }

            try:
                response = await client.post(
                    f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=remaining_s,
                )
                elapsed_ms = round(
                    (time.monotonic() - attempt_started) * 1000, 2
                )
                if response.is_error:
                    attempts.append({
                        "model": model_id,
                        "status": response.status_code,
                        "latency_ms": elapsed_ms,
                        "error": "Provider returned an error response",
                    })
                    continue

                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    attempts.append({
                        "model": model_id,
                        "latency_ms": elapsed_ms,
                        "error": "Provider returned no response choices",
                    })
                    continue

                content = choices[0].get("message", {}).get("content")
                if isinstance(content, list):
                    content = "\n".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                    )
                if not isinstance(content, str) or not content.strip():
                    attempts.append({
                        "model": model_id,
                        "latency_ms": elapsed_ms,
                        "error": "Provider returned no text content",
                    })
                    continue
                model["measured_latency_ms"] = elapsed_ms
                attempts.append({
                    "model": model_id,
                    "latency_ms": elapsed_ms,
                    "status": response.status_code,
                    "success": True,
                })
                return {
                    "model": model_id,
                    "content": content,
                    "latency_ms": elapsed_ms,
                    "usage": data.get("usage") or {},
                    "attempts": attempts,
                    "total_latency_ms": round(
                        (time.monotonic() - started_at) * 1000, 2
                    ),
                }
            except httpx.TimeoutException:
                elapsed_ms = round(
                    (time.monotonic() - attempt_started) * 1000, 2
                )
                attempts.append({
                    "model": model_id,
                    "latency_ms": elapsed_ms,
                    "error": "Request timed out",
                })
            except httpx.RequestError as exc:
                attempts.append({
                    "model": model_id,
                    "latency_ms": round(
                        (time.monotonic() - attempt_started) * 1000, 2
                    ),
                    "error": f"Network error: {type(exc).__name__}",
                })
            except (ValueError, KeyError, TypeError) as exc:
                attempts.append({
                    "model": model_id,
                    "latency_ms": round(
                        (time.monotonic() - attempt_started) * 1000, 2
                    ),
                    "error": f"Invalid provider response: {type(exc).__name__}",
                })

    raise InferenceError(
        "No eligible model returned a response within the latency budget.",
        attempts,
    )
