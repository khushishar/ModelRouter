
import json
from app.config import CATALOGUE_PATH
from app.constraints import run_pipeline

with CATALOGUE_PATH.open("r", encoding="utf-8") as f:
    payload = json.load(f)

models = payload["data"]

result = run_pipeline(
    models,
    task="general",
    min_capability=None,
    input_modality="text",
    output_modality="text",
    free_only=True,
    max_cost_usd=0.02,
    required_context=4096,

    # Temporarily disable latency filtering until measurements exist.
    max_latency_ms=None,
)

print("\nStage counts:")
for stage, count in result["stage_counts"].items():
    print(f"{stage}: {count}")

print("\nEligible models:")
for model in result["eligible"][:10]:
    print(
        model["id"],
        "| context:", model.get("context_length"),
        "| estimated cost:",
        model.get("_router_estimated_cost_usd"),
    )

print("\nRejection counts:")
for stage, rejected in result["rejections_by_stage"].items():
    print(f"{stage}: {len(rejected)}")
