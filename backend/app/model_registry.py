import json
import logging
import math
from typing import Any

from app.config import CATALOGUE_PATH, MODEL_OVERRIDES_PATH

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Reads and normalizes the locally cached OpenRouter catalogue."""

    def __init__(self):
        self.models: list[dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        try:
            with CATALOGUE_PATH.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except FileNotFoundError:
            logger.exception("OpenRouter catalogue is missing: %s", CATALOGUE_PATH)
            raise
        except (OSError, json.JSONDecodeError):
            logger.exception("Unable to read OpenRouter catalogue: %s", CATALOGUE_PATH)
            raise

        raw_models = payload.get("data")
        if not isinstance(raw_models, list):
            raise ValueError("OpenRouter catalogue must contain a data array")

        self.models = [
            model
            for item in raw_models
            if isinstance(item, dict)
            if (model := self._normalize(item)) is not None
        ]
        logger.info("Loaded %d models from local catalogue", len(self.models))

    @staticmethod
    def _normalize(item: dict[str, Any]) -> dict[str, Any] | None:
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            return None

        architecture = item.get("architecture") or {}
        pricing = item.get("pricing") or {}
        return {
            "id": model_id,
            "name": item.get("name") or model_id,
            "provider": model_id.split("/")[0] if "/" in model_id else "unknown",
            "context_length": item.get("context_length"),
            "architecture": architecture,
            "input_modalities": architecture.get("input_modalities") or [],
            "output_modalities": architecture.get("output_modalities") or [],
            "supported_parameters": item.get("supported_parameters") or [],
            "pricing": pricing,
            "input_price_per_token": pricing.get("prompt"),
            "output_price_per_token": pricing.get("completion"),
            "benchmarks": item.get("benchmarks") or {},
            "modelrouter_metrics": item.get("modelrouter_metrics") or {},
            "reasoning": item.get("reasoning") or {},
            "available": item.get("available", True),
        }

    def list_models(self) -> list[dict[str, Any]]:
        try:
            with MODEL_OVERRIDES_PATH.open("r", encoding="utf-8") as file:
                overrides = json.load(file)
        except FileNotFoundError:
            overrides = {}
        except (OSError, json.JSONDecodeError):
            logger.exception(
                "Unable to read model overrides: %s", MODEL_OVERRIDES_PATH
            )
            raise

        if not isinstance(overrides, dict):
            raise ValueError("Model overrides must be a JSON object")

        models = [dict(model) for model in self.models]
        indexed = {model["id"]: model for model in models}
        unknown_ids = set(overrides) - set(indexed)
        if unknown_ids:
            raise ValueError(
                "Model overrides reference unknown model IDs: "
                + ", ".join(sorted(unknown_ids))
            )

        for model_id, override in overrides.items():
            if not isinstance(override, dict):
                raise ValueError(f"Override for {model_id} must be an object")
            model = indexed[model_id]
            if "available" in override:
                if not isinstance(override["available"], bool):
                    raise ValueError(
                        f"Availability override for {model_id} must be boolean"
                    )
                model["available"] = override["available"]
            if "measured_latency_ms" in override:
                latency = override["measured_latency_ms"]
                if latency is not None and (
                    not isinstance(latency, (int, float))
                    or isinstance(latency, bool)
                    or not math.isfinite(latency)
                    or latency < 0
                ):
                    raise ValueError(
                        f"Latency override for {model_id} must be "
                        "a non-negative number or null"
                    )
                model["measured_latency_ms"] = latency
        return models

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        model = next(
            (item for item in self.models if item["id"] == model_id),
            None,
        )
        return dict(model) if model else None
