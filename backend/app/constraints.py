import math
from typing import Any


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError):
        return None


def get_capability(model: dict[str, Any], task: str) -> float | None:
    benchmarks = model.get("benchmarks") or {}
    artificial_analysis = benchmarks.get("artificial_analysis") or {}
    field = {
        "coding": "coding_index",
        "agentic": "agentic_index",
    }.get(task, "intelligence_index")
    value = artificial_analysis.get(field)
    return _number(value)


def get_price(value: Any) -> float | None:
    """OpenRouter prices are commonly returned as strings per token."""
    return _number(value)


def _model_modalities(model: dict[str, Any], key: str) -> list[str]:
    value = model.get(key)
    if value is None:
        architecture = model.get("architecture") or {}
        architecture_key = (
            "input_modalities" if key == "input_modalities"
            else "output_modalities"
        )
        value = architecture.get(architecture_key)
    return [str(item).casefold() for item in (value or [])]


def _matches_modality(supported: list[str], required: str) -> bool:
    aliases = {
        "pdf": {"pdf", "file"},
        "image": {"image"},
        "audio": {"audio"},
        "video": {"video"},
        "text": {"text"},
    }
    return bool(aliases.get(required, {required}) & set(supported))


def _pricing(model: dict[str, Any]) -> tuple[float | None, float | None]:
    pricing = model.get("pricing") or {}
    return (
        get_price(pricing.get("prompt", model.get("input_price_per_token"))),
        get_price(
            pricing.get("completion", model.get("output_price_per_token"))
        ),
    )


def _latency(model: dict[str, Any]) -> float | None:
    metrics = model.get("modelrouter_metrics") or {}
    return _number(
        model.get("measured_latency_ms", metrics.get("median_latency_ms"))
    )


def run_pipeline(
    models: list[dict[str, Any]],
    *,
    task: str = "general",
    min_capability: float | None = None,
    input_modality: str = "text",
    input_modalities: list[str] | None = None,
    output_modality: str = "text",
    free_only: bool = True,
    paid_consent: bool = False,
    max_cost_usd: float = 0.02,
    estimated_input_tokens: int = 2000,
    estimated_output_tokens: int = 500,
    estimated_output_images: int = 0,
    required_context: int = 4096,
    max_latency_ms: float | None = 5000,
) -> dict[str, Any]:
    """Filter models against hard constraints and keep every rejection reason."""
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    after_capability = 0
    after_pricing = 0

    for original in models:
        model = dict(original)
        model_id = str(model.get("id", "unknown"))
        reasons: list[dict[str, str]] = []
        if model.get("available") is False:
            reasons.append({
                "constraint": "availability",
                "reason": "Model is marked unavailable",
            })
        inputs = _model_modalities(model, "input_modalities")
        outputs = _model_modalities(model, "output_modalities")

        required_inputs = input_modalities or [input_modality]
        for required in required_inputs:
            if not _matches_modality(inputs, required):
                reasons.append({
                    "constraint": "modality",
                    "reason": f"Unsupported input modality: {required}",
                })
        if not _matches_modality(outputs, output_modality):
            reasons.append({
                "constraint": "modality",
                "reason": f"Unsupported output modality: {output_modality}",
            })

        capability = get_capability(model, task)
        if min_capability is not None:
            if capability is None:
                reasons.append({
                    "constraint": "capability",
                    "reason": f"No benchmark available for task: {task}",
                })
            elif capability < min_capability:
                reasons.append({
                    "constraint": "capability",
                    "reason": (
                        f"Capability {capability:g} is below minimum "
                        f"{min_capability:g}"
                    ),
                })
        if not reasons:
            after_capability += 1

        prompt_price, completion_price = _pricing(model)
        if prompt_price is None or completion_price is None:
            reasons.append({
                "constraint": "cost",
                "reason": "Prompt or completion price is unknown",
            })
            estimated_cost = None
        else:
            estimated_cost = (
                prompt_price * estimated_input_tokens
                + completion_price * estimated_output_tokens
            )
            image_price = None
            if estimated_output_images:
                image_price = get_price(
                    (model.get("pricing") or {}).get("image_output")
                )
                if image_price is None:
                    reasons.append({
                        "constraint": "cost",
                        "reason": "Image output price is unknown",
                    })
                else:
                    estimated_cost += image_price * estimated_output_images
            if free_only and (
                prompt_price != 0
                or completion_price != 0
                or (estimated_output_images and image_price != 0)
            ):
                reasons.append({
                    "constraint": "cost",
                    "reason": "Model is not verified as free",
                })
            elif not free_only and not paid_consent and estimated_cost > 0:
                reasons.append({
                    "constraint": "cost",
                    "reason": "Paid inference requires user consent",
                })

            if estimated_cost > max_cost_usd:
                reasons.append({
                    "constraint": "cost",
                    "reason": (
                        f"Estimated cost ${estimated_cost:.6f} exceeds "
                        f"budget ${max_cost_usd:.6f}"
                    ),
                })

        if not any(reason["constraint"] == "cost" for reason in reasons):
            after_pricing += 1
        if estimated_cost is not None:
            model["_router_estimated_cost_usd"] = estimated_cost

        context = _number(model.get("context_length"))
        if context is None or context < required_context:
            reasons.append({
                "constraint": "context",
                "reason": (
                    f"Context length is unknown or below the required "
                    f"{required_context:,} tokens"
                ),
            })

        latency = _latency(model)
        if max_latency_ms is not None and latency is not None:
            if latency > max_latency_ms:
                reasons.append({
                    "constraint": "latency",
                    "reason": (
                        f"Measured latency {latency:g} ms exceeds the "
                        f"{max_latency_ms:g} ms budget"
                    ),
                })

        if reasons:
            rejected.append({
                "id": model_id,
                "name": model.get("name", model_id),
                "reasons": reasons,
            })
        else:
            model["_router_capability"] = capability
            eligible.append(model)

    return {
        "eligible": eligible,
        "rejected": rejected,
        "stage_counts": {
            "catalogue": len(models),
            "after_capability": after_capability,
            "after_pricing": after_pricing,
            "eligible": len(eligible),
        },
        "rejections_by_constraint": {
            key: [
                {
                    "id": entry["id"],
                    "name": entry["name"],
                    "reason": reason["reason"],
                }
                for entry in rejected
                for reason in entry["reasons"]
                if reason["constraint"] == key
            ]
            for key in (
                "availability", "modality", "capability", "cost",
                "context", "latency",
            )
        },
    }
