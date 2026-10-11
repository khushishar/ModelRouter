
import asyncio
import json

from app.config import CATALOGUE_PATH
from app.constraints import run_pipeline
from app.latency import benchmark_models

from app.latency import benchmark_models, save_latency_to_catalogue
async def main():
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

    # Benchmark only a small initial subset to limit API requests.
    candidates = result["eligible"]

    results = await benchmark_models(candidates, repetitions=3)
    print("Benchmark results:", results)
    save_latency_to_catalogue(results)

    print("\nLatency benchmark results:")
    for item in results:
        print(
            item["id"],
            "| median:", item["median_latency_ms"], "ms",
            "| runs:", item["successful_runs"],
        )


if __name__ == "__main__":
    asyncio.run(main())
