from math import isfinite
from typing import Any


def _numeric(value: Any) -> float | None:
    try:
        parsed = float(value) if value is not None else None
        return parsed if parsed is not None and isfinite(parsed) else None
    except (ValueError, TypeError):
        return None


def _score(model: dict[str, Any], task: str) -> float | None:
    value = model.get("_router_capability")
    if value is None:
        benchmarks = model.get("benchmarks") or {}
        aa = benchmarks.get("artificial_analysis") or {}
        field = {
            "coding": "coding_index",
            "agentic": "agentic_index",
        }.get(task, "intelligence_index")
        value = aa.get(field)
    return _numeric(value)


def _latency(model: dict[str, Any]) -> float | None:
    metrics = model.get("modelrouter_metrics") or {}
    value = model.get(
        "measured_latency_ms", metrics.get("median_latency_ms")
    )
    return _numeric(value)


def _cost(model: dict[str, Any]) -> float:
    value = _numeric(model.get("_router_estimated_cost_usd"))
    return value if value is not None else float("inf")


def _supports_reasoning(model: dict[str, Any]) -> bool:
    params = set(model.get("supported_parameters") or [])
    return bool(
        {"reasoning", "include_reasoning"} & params
        or (model.get("reasoning") or {}).get("default_enabled") is True
    )


def _utility(values: list[float | None], *, higher_is_better: bool) -> list[float]:
    """Normalize one metric to 0-1 utility; missing measurements score 0."""
    known = [value for value in values if value is not None]
    if not known:
        return [0.0] * len(values)

    low, high = min(known), max(known)
    if low == high:
        return [1.0 if value is not None else 0.0 for value in values]

    utilities = []
    for value in values:
        if value is None:
            utilities.append(0.0)
        elif higher_is_better:
            utilities.append((value - low) / (high - low))
        else:
            utilities.append((high - value) / (high - low))
    return utilities


def rank_models(
    models: list[dict[str, Any]],
    mode: str = "fast",
    task: str = "general",
) -> list[dict[str, Any]]:
    """Rank only eligible models; ranking never changes their eligibility."""
    if mode not in {"fast", "thinking"}:
        raise ValueError("mode must be 'fast' or 'thinking'")

    latencies = [_latency(model) for model in models]
    costs = [_cost(model) for model in models]
    capabilities = [_score(model, task) for model in models]
    latency_utility = _utility(latencies, higher_is_better=False)
    cost_utility = _utility(
        [cost if isfinite(cost) else None for cost in costs],
        higher_is_better=False,
    )
    capability_utility = _utility(capabilities, higher_is_better=True)
    reasoning_utility = [
        1.0 if _supports_reasoning(model) else 0.0 for model in models
    ]

    if mode == "fast":
        weights = (0.75, 0.15, 0.10, 0.0)
    else:
        weights = (0.55, 0.05, 0.15, 0.25)

    scored = []
    for index, model in enumerate(models):
        latency_score, cost_score, capability_score, reasoning_score = (
            latency_utility[index],
            cost_utility[index],
            capability_utility[index],
            reasoning_utility[index],
        )
        total = (
            weights[0] * latency_score
            + weights[1] * cost_score
            + weights[2] * capability_score
            + weights[3] * reasoning_score
        )
        scored.append((total, index, model))

    ranked = sorted(
        scored,
        key=lambda item: (
            -item[0],
            latencies[item[1]] is None,
            latencies[item[1]]
            if latencies[item[1]] is not None
            else float("inf"),
            costs[item[1]],
            item[1],
        ),
    )
    return [model for _, _, model in ranked]
