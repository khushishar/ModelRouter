from typing import Any


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
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def _latency(model: dict[str, Any]) -> float | None:
    metrics = model.get("modelrouter_metrics") or {}
    value = model.get(
        "measured_latency_ms", metrics.get("median_latency_ms")
    )
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def _cost(model: dict[str, Any]) -> float:
    try:
        return float(model.get("_router_estimated_cost_usd", float("inf")))
    except (TypeError, ValueError):
        return float("inf")


def rank_models(
    models: list[dict[str, Any]],
    mode: str = "fast",
    task: str = "general",
) -> list[dict[str, Any]]:
    """Rank only eligible models; ranking never changes their eligibility."""
    if mode not in {"fast", "thinking"}:
        raise ValueError("mode must be 'fast' or 'thinking'")

    if mode == "fast":
        return sorted(
            models,
            key=lambda model: (
                _latency(model) is None,
                _latency(model) if _latency(model) is not None else float("inf"),
                _cost(model),
            ),
        )

    def supports_reasoning(model: dict[str, Any]) -> bool:
        params = set(model.get("supported_parameters") or [])
        return bool(
            {"reasoning", "include_reasoning"} & params
            or (model.get("reasoning") or {}).get("default_enabled") is True
        )

    return sorted(
        models,
        key=lambda model: (
            not supports_reasoning(model),
            _score(model, task) is None,
            -(_score(model, task) or 0),
            _latency(model) is None,
            _latency(model) if _latency(model) is not None else float("inf"),
            _cost(model),
        ),
    )
