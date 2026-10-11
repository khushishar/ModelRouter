import json
import logging
import math
import os
import tempfile
from threading import RLock
from typing import Any

from app.config import CATALOGUE_PATH, MODEL_OVERRIDES_PATH

logger = logging.getLogger(__name__)
_OVERRIDE_LOCK = RLock()


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
        try:
            prompt_price = float(pricing["prompt"])
            completion_price = float(pricing["completion"])
            is_free = (
                math.isfinite(prompt_price)
                and math.isfinite(completion_price)
                and prompt_price == 0
                and completion_price == 0
            )
        except (KeyError, TypeError, ValueError):
            is_free = False

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
            "is_free": is_free,
            "benchmarks": item.get("benchmarks") or {},
            "modelrouter_metrics": item.get("modelrouter_metrics") or {},
            "reasoning": item.get("reasoning") or {},
            "available": item.get("available", True),
        }

    def list_models(self) -> list[dict[str, Any]]:
        overrides = self.get_overrides()

        models = [dict(model) for model in self.models]
        indexed = {model["id"]: model for model in models}

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

    def get_overrides(self) -> dict[str, Any]:
        overrides = self._load_overrides()
        known_ids = {model["id"] for model in self.models}
        unknown_ids = set(overrides) - known_ids
        if unknown_ids:
            raise ValueError(
                "Model overrides reference unknown model IDs: "
                + ", ".join(sorted(unknown_ids))
            )
        return overrides

    @staticmethod
    def _load_overrides() -> dict[str, Any]:
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
        return overrides

    def set_override(
        self,
        model_id: str,
        *,
        available: bool,
        measured_latency_ms: float | None,
    ) -> None:
        if model_id not in {model["id"] for model in self.models}:
            raise KeyError(model_id)
        if not isinstance(available, bool):
            raise ValueError("Availability override must be boolean")
        if measured_latency_ms is not None and (
            isinstance(measured_latency_ms, bool)
            or not isinstance(measured_latency_ms, (int, float))
            or not math.isfinite(measured_latency_ms)
            or measured_latency_ms < 0
        ):
            raise ValueError(
                "Latency override must be a non-negative number or null"
            )

        with _OVERRIDE_LOCK:
            overrides = self._load_overrides()
            overrides[model_id] = {
                "available": available,
                "measured_latency_ms": measured_latency_ms,
            }
            self._write_overrides(overrides)

    def clear_override(self, model_id: str) -> None:
        if model_id not in {model["id"] for model in self.models}:
            raise KeyError(model_id)

        with _OVERRIDE_LOCK:
            overrides = self._load_overrides()
            overrides.pop(model_id, None)
            self._write_overrides(overrides)

    @staticmethod
    def _write_overrides(overrides: dict[str, Any]) -> None:
        temporary_path = None
        try:
            MODEL_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=MODEL_OVERRIDES_PATH.parent,
                prefix=f".{MODEL_OVERRIDES_PATH.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = file.name
                json.dump(overrides, file, ensure_ascii=False, indent=2)
                file.write("\n")
            os.replace(temporary_path, MODEL_OVERRIDES_PATH)
        except OSError:
            logger.exception(
                "Unable to write model overrides: %s", MODEL_OVERRIDES_PATH
            )
            raise
        finally:
            if temporary_path and os.path.exists(temporary_path):
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        model = next(
            (item for item in self.models if item["id"] == model_id),
            None,
        )
        return dict(model) if model else None
