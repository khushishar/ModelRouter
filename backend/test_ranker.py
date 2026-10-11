
import json

from app.config import CATALOGUE_PATH
from app.constraints import run_pipeline
from app.ranker import rank_models

with CATALOGUE_PATH.open("r", encoding="utf-8") as f:
    models = json.load(f)["data"]

result = run_pipeline(
    models,
    task="general",
    min_capability=None,
    input_modality="text",
    output_modality="text",
    free_only=True,
    max_cost_usd=0.02,
    required_context=4096,
    max_latency_ms=None,
)

eligible = result["eligible"]

print("Eligible models:", len(eligible))

for mode in ("fast", "thinking"):
    ranked = rank_models(eligible, mode=mode)

    print(f"\nTop 10 for {mode.upper()}:")
    for model in ranked[:10]:
        print(
            model["id"],
            "| latency:", model.get("measured_latency_ms"),
            "| reasoning:", (
                "reasoning" in (model.get("supported_parameters") or [])
                or "include_reasoning" in (
                    model.get("supported_parameters") or []
                )
            ),
        )
