import time
from typing import Any

import httpx

from app.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL


class InferenceError(Exception):
    """Raised when every valid candidate fails within the request budget."""

    def __init__(self, message: str, attempts: list[dict[str, Any]]):
        super().__init__(message)
        self.attempts = attempts


def _provider_error(response: httpx.Response) -> str:
    """Extract a concise provider error without retaining the full body."""
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        details = []
        code = error.get("code")
        message = error.get("message")
        if code is not None:
            details.append(f"code {code}")
        if isinstance(message, str) and message.strip():
            details.append(message.strip()[:400])
        if details:
            return f"HTTP {response.status_code}: {' — '.join(details)}"
    if isinstance(error, str) and error.strip():
        return f"HTTP {response.status_code}: {error.strip()[:400]}"
    return f"HTTP {response.status_code}: provider returned an error response"


async def generate_response(
    messages: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    latency_budget_ms: float = 30_000,
    output_modality: str = "text",
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
    deadline_exhausted = False
    headers = {
        "Authorization": "Bearer " + OPENROUTER_API_KEY,
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5173",
        "X-Title": "ModelRouter",
    }

    async with httpx.AsyncClient() as client:
        for model in candidates:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                deadline_exhausted = True
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
            if output_modality == "image":
                payload["modalities"] = ["image", "text"]

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
                    provider_error = _provider_error(response)
                    attempts.append({
                        "model": model_id,
                        "status": response.status_code,
                        "latency_ms": elapsed_ms,
                        "error": provider_error,
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

                message = choices[0].get("message") or {}
                content = message.get("content")
                images = []
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if isinstance(part.get("text"), str):
                            text_parts.append(part["text"])
                        image = part.get("image_url")
                        image_url = (
                            image.get("url") if isinstance(image, dict)
                            else image
                        )
                        if isinstance(image_url, str):
                            images.append(image_url)
                    content = "\n".join(text_parts)
                if not isinstance(content, str):
                    content = ""
                for image in message.get("images") or []:
                    image_url = (
                        image.get("image_url") if isinstance(image, dict)
                        else None
                    )
                    image_url = (
                        image_url.get("url")
                        if isinstance(image_url, dict)
                        else image_url
                    )
                    if isinstance(image_url, str):
                        images.append(image_url)
                images = list(dict.fromkeys(images))
                if output_modality == "image" and not images:
                    attempts.append({
                        "model": model_id,
                        "latency_ms": elapsed_ms,
                        "error": "Provider returned no generated image",
                    })
                    continue
                if not content.strip() and not images:
                    attempts.append({
                        "model": model_id,
                        "latency_ms": elapsed_ms,
                        "error": "Provider returned no text or image content",
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
                    "images": images,
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
                deadline_exhausted = time.monotonic() >= deadline
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

    if deadline_exhausted or time.monotonic() >= deadline:
        message = (
            "The shared latency budget expired before an eligible model "
            "returned a response."
        )
    elif attempts and all("status" in attempt for attempt in attempts):
        errors = "; ".join(
            f"{attempt['model']}: {attempt['error']}"
            for attempt in attempts
        )
        message = f"All attempted models were rejected by the provider: {errors}"
    else:
        errors = "; ".join(
            f"{attempt['model']}: {attempt['error']}"
            for attempt in attempts
        )
        message = (
            "All attempted models failed before returning a response."
            + (f" Failures: {errors}" if errors else "")
        )

    raise InferenceError(message, attempts)
