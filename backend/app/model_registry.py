import json
from typing import Any

from app.config import CATALOGUE_PATH


class ModelRegistry:
    """Reads and normalizes the locally cached OpenRouter catalogue."""

    def __init__(self):
        self.models: list[dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        if not CATALOGUE_PATH.exists():
            raise FileNotFoundError(
                f"Model catalogue not found: {CATALOGUE_PATH}. "
                "Run `python -m app.catalogue` first."
            )

        with CATALOGUE_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        raw_models = payload.get("data", [])

        self.models = [
            model
            for item in raw_models
            if (model := self._normalize(item)) is not None
        ]

        print(f"Loaded {len(self.models)} models from local catalogue.")

    @staticmethod
    def _normalize(item: dict[str, Any]) -> dict[str, Any] | None:
        model_id = item.get("id")
        if not model_id:
            return None

        architecture = item.get("architecture") or {}
        pricing = item.get("pricing") or {}

        input_modalities = architecture.get("input_modalities") or ["text"]
        output_modalities = architecture.get("output_modalities") or ["text"]

        def price(value: Any) -> float | None:
            try:
                return float(value) if value is not None else None
            except (ValueError, TypeError):
                return None

        return {
            "id": model_id,
            "name": item.get("name") or model_id,
            "provider": model_id.split("/")[0] if "/" in model_id else "unknown",
            "context_length": item.get("context_length"),
            "input_modalities": input_modalities,
            "output_modalities": output_modalities,
            "supported_parameters": item.get("supported_parameters") or [],
            "input_price_per_token": price(pricing.get("prompt")),
            "output_price_per_token": price(pricing.get("completion")),
            "is_free": (
                price(pricing.get("prompt")) == 0
                and price(pricing.get("completion")) == 0
            ),
            # These are application-owned fields, not official OpenRouter scores.
            "capability": None,
            "measured_latency_ms": None,
        }

    def list_models(self) -> list[dict[str, Any]]:
        return self.models

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        return next(
            (model for model in self.models if model["id"] == model_id),
            None,
        )