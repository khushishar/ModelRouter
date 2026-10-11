
from typing import Any


def get_capability(model: dict[str, Any], task: str) -> float | None:
    """Extract a task-specific capability signal from OpenRouter benchmarks."""
    benchmarks = model.get("benchmarks") or {}
    aa = benchmarks.get("artificial_analysis") or {}

    fields = {
        "general": "intelligence_index",
        "coding": "coding_index",
        "agentic": "agentic_index",
    }

    field = fields.get(task)
    value = aa.get(field) if field else None

    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def get_price(value: Any) -> float | None:
    """OpenRouter prices are commonly returned as strings per token."""
    try:
        price = float(value)
        return price if price >= 0 else None
    except (ValueError, TypeError):
        return None


def capability_filter(
    models: list[dict[str, Any]],
    *,
    task: str = "general",
    min_capability: float | None = None,
    input_modality: str = "text",
    output_modality: str = "text",
):
    """Stage 1: modality compatibility and optional benchmark threshold."""
    eligible, rejected = [], []

    for model in models:
        reasons = []
        architecture = model.get("architecture") or {}
        inputs = architecture.get("input_modalities") or []
        outputs = architecture.get("output_modalities") or []

        if input_modality not in inputs:
            reasons.append(f"Unsupported input modality: {input_modality}")

        if output_modality not in outputs:
            reasons.append(f"Unsupported output modality: {output_modality}")

        capability = get_capability(model, task)

        if min_capability is not None:
            if capability is None:
                reasons.append(f"No benchmark available for task: {task}")
            elif capability < min_capability:
                reasons.append("Capability benchmark below minimum")

        if reasons:
            rejected.append({
                "id": model.get("id"),
                "stage": "capability",
                "reasons": reasons,
            })
        else:
            # Keep the extracted value available for ranking and explanations.
            model["_router_capability"] = capability
            eligible.append(model)

    if min_capability is not None:
        eligible.sort(
            key=lambda m: m["_router_capability"],
            reverse=True,
        )

    return eligible, rejected


def pricing_filter(
    models: list[dict[str, Any]],
    *,
    free_only: bool = True,
    paid_consent: bool = False,
    max_cost_usd: float = 0.02,
    estimated_input_tokens: int = 2000,
    estimated_output_tokens: int = 500,
):
    """Stage 2: pricing policy and estimated request cost."""
    eligible, rejected = [], []

    for model in models:
        reasons = []
        model_id = model.get("id", "")
        pricing = model.get("pricing") or {}

        prompt_price = get_price(pricing.get("prompt"))
        completion_price = get_price(pricing.get("completion"))

        # A :free suffix is a useful signal, but verify pricing too.
        verified_free = (
            model_id.endswith(":free")
            or (
                prompt_price == 0
                and completion_price == 0
            )
        )

        if prompt_price is None or completion_price is None:
            reasons.append("Prompt or completion price unknown")

        if free_only and not verified_free:
            reasons.append("Model is not verified as free")
        elif not free_only and not verified_free and not paid_consent:
            reasons.append("Paid model requires user consent")

        if prompt_price is not None and completion_price is not None:
            estimated_cost = (
                prompt_price * estimated_input_tokens
                + completion_price * estimated_output_tokens
            )

            if estimated_cost > max_cost_usd:
                reasons.append(
                    f"Estimated cost ${estimated_cost:.6f} "
                    f"exceeds budget ${max_cost_usd:.2f}"
                )
        else:
            estimated_cost = None

        if reasons:
            rejected.append({
                "id": model.get("id"),
                "stage": "pricing",
                "reasons": reasons,
            })
        else:
            model["_router_estimated_cost_usd"] = estimated_cost
            eligible.append(model)

    return eligible, rejected


def speed_context_filter(
    models: list[dict[str, Any]],
    *,
    required_context: int = 4096,
    max_latency_ms: float | None = 5000,
):
    """Stage 3: context requirement and measured latency."""
    eligible, rejected = [], []

    for model in models:
        reasons = []
        context = model.get("context_length")

        # This is an application-measured value, not a catalogue field.
        latency = model.get("measured_latency_ms")

        if context is None or context < required_context:
            reasons.append("Context length unknown or insufficient")

        if max_latency_ms is not None:
            if latency is None:
                reasons.append("Latency has not been measured")
            elif latency > max_latency_ms:
                reasons.append("Latency exceeds maximum")

        if reasons:
            rejected.append({
                "id": model.get("id"),
                "stage": "speed_context",
                "reasons": reasons,
            })
        else:
            eligible.append(model)

    if max_latency_ms is not None:
        eligible.sort(key=lambda m: m["measured_latency_ms"])

    return eligible, rejected


def run_pipeline(
    models: list[dict[str, Any]],
    *,
    task: str = "general",
    min_capability: float | None = None,
    input_modality: str = "text",
    output_modality: str = "text",
    free_only: bool = True,
    paid_consent: bool = False,
    max_cost_usd: float = 0.02,
    required_context: int = 4096,
    max_latency_ms: float | None = 5000,
):
    """Run sequential filters and preserve rejection reasons."""
    stage1, rejected1 = capability_filter(
        models,
        task=task,
        min_capability=min_capability,
        input_modality=input_modality,
        output_modality=output_modality,
    )

    stage2, rejected2 = pricing_filter(
        stage1,
        free_only=free_only,
        paid_consent=paid_consent,
        max_cost_usd=max_cost_usd,
    )

    stage3, rejected3 = speed_context_filter(
        stage2,
        required_context=required_context,
        max_latency_ms=max_latency_ms,
    )

    return {
        "eligible": stage3,
        "rejected": rejected1 + rejected2 + rejected3,
        "stage_counts": {
            "catalogue": len(models),
            "after_capability": len(stage1),
            "after_pricing": len(stage2),
            "after_speed_context": len(stage3),
        },
        "rejections_by_stage": {
            "capability": rejected1,
            "pricing": rejected2,
            "speed_context": rejected3,
        },
    }
