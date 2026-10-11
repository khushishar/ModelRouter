import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from app import api, inference, model_registry
from app.constraints import run_pipeline
from app.estimator import analyze_prompt, estimate_tokens
from app.ranker import rank_models


def model(
    model_id="test/free",
    *,
    price="0",
    context=16_384,
    latency=None,
    modalities=None,
    reasoning=False,
):
    result = {
        "id": model_id,
        "name": model_id,
        "context_length": context,
        "architecture": {
            "input_modalities": modalities or ["text"],
            "output_modalities": ["text"],
        },
        "pricing": {"prompt": price, "completion": price},
        "supported_parameters": ["reasoning"] if reasoning else [],
    }
    if latency is not None:
        result["modelrouter_metrics"] = {"median_latency_ms": latency}
    return result


class EstimatorTests(unittest.TestCase):
    def test_counts_non_whitespace_input_and_classifies_complexity(self):
        self.assertGreater(estimate_tokens("hello, model router!"), 0)
        result = analyze_prompt("Compare these designs step by step.")
        self.assertEqual(result["complexity"], "medium")
        self.assertEqual(result["task"], "reasoning")

    def test_pdf_attachment_selects_file_capability_without_text_prompt(self):
        attachment = api.Attachment(
            name="brief.pdf",
            mime_type="application/pdf",
            data="cGRm",
        )
        message, modalities, file_tokens = api._build_user_message(
            "",
            [(attachment, "application/pdf", b"pdf")],
        )
        self.assertEqual(modalities, ["pdf"])
        self.assertEqual(message["content"][0]["type"], "file")
        self.assertGreaterEqual(file_tokens, 1024)


class ConstraintTests(unittest.TestCase):
    def test_unavailable_models_are_hard_rejected(self):
        candidate = model()
        candidate["available"] = False
        pipeline = run_pipeline([candidate])
        self.assertEqual(pipeline["eligible"], [])
        self.assertEqual(
            pipeline["rejected"][0]["reasons"][0]["constraint"],
            "availability",
        )

    def test_rejects_every_hard_constraint_with_reasons(self):
        candidate = model(
            "test/too-small",
            price="0.001",
            context=512,
            latency=5000,
            modalities=["text"],
        )
        pipeline = run_pipeline(
            [candidate],
            input_modalities=["text", "image"],
            required_context=2048,
            max_latency_ms=1000,
            max_cost_usd=0.001,
            estimated_input_tokens=1000,
            estimated_output_tokens=1000,
        )
        self.assertEqual(pipeline["eligible"], [])
        self.assertEqual(
            {reason["constraint"] for reason in pipeline["rejected"][0]["reasons"]},
            {"modality", "cost", "context", "latency"},
        )

    def test_unknown_latency_is_enforced_by_call_budget_not_assumed_fast(self):
        pipeline = run_pipeline(
            [model()],
            required_context=1024,
            max_latency_ms=500,
        )
        self.assertEqual(len(pipeline["eligible"]), 1)

    def test_capability_floor_and_pdf_file_modality_are_hard_filters(self):
        capable_pdf_model = model(
            "test/pdf",
            modalities=["text", "file"],
        )
        capable_pdf_model["benchmarks"] = {
            "artificial_analysis": {"intelligence_index": 50}
        }
        too_weak = model("test/weak")
        too_weak["benchmarks"] = {
            "artificial_analysis": {"intelligence_index": 20}
        }
        pipeline = run_pipeline(
            [capable_pdf_model, too_weak],
            input_modalities=["text", "pdf"],
            min_capability=40,
            required_context=1024,
            max_latency_ms=500,
        )
        self.assertEqual([item["id"] for item in pipeline["eligible"]], ["test/pdf"])
        self.assertTrue(any(
            reason["constraint"] == "capability"
            for reason in pipeline["rejected"][0]["reasons"]
        ))


class RankerTests(unittest.TestCase):
    def test_speed_is_primary_in_fast_and_thinking_modes(self):
        fast = model("test/fast", latency=100, reasoning=False)
        thinking = model("test/thinking", latency=300, reasoning=True)
        self.assertEqual(
            rank_models([thinking, fast], "fast")[0]["id"],
            "test/fast",
        )
        self.assertEqual(
            rank_models([fast, thinking], "thinking")[0]["id"],
            "test/fast",
        )

    def test_speed_has_more_weight_in_fast_mode_than_model_quality(self):
        fast = model("test/fast", latency=100)
        fast["benchmarks"] = {
            "artificial_analysis": {"intelligence_index": 20}
        }
        capable = model("test/capable", latency=300)
        capable["benchmarks"] = {
            "artificial_analysis": {"intelligence_index": 100}
        }

        self.assertEqual(
            rank_models([capable, fast], "fast")[0]["id"],
            "test/fast",
        )

    def test_thinking_uses_reasoning_and_task_capability_with_speed(self):
        fast = model("test/fast", latency=100)
        fast["benchmarks"] = {
            "artificial_analysis": {"coding_index": 20}
        }
        thinking = model("test/thinking", latency=150, reasoning=True)
        thinking["benchmarks"] = {
            "artificial_analysis": {"coding_index": 100}
        }
        slow = model("test/slow", latency=300)

        self.assertEqual(
            rank_models([fast, slow, thinking], "thinking", "coding")[0]["id"],
            "test/thinking",
        )


class RegistryOverrideTests(unittest.TestCase):
    def test_override_file_is_reloaded_for_each_model_list(self):
        with tempfile.TemporaryDirectory() as directory:
            catalogue_path = Path(directory) / "catalogue.json"
            overrides_path = Path(directory) / "overrides.json"
            catalogue_path.write_text(
                json.dumps({"data": [{
                    "id": "test/model",
                    "context_length": 4096,
                    "architecture": {
                        "input_modalities": ["text"],
                        "output_modalities": ["text"],
                    },
                    "pricing": {"prompt": "0", "completion": "0"},
                }]}),
                encoding="utf-8",
            )
            with (
                patch.object(model_registry, "CATALOGUE_PATH", catalogue_path),
                patch.object(model_registry, "MODEL_OVERRIDES_PATH", overrides_path),
            ):
                registry = model_registry.ModelRegistry()
                overrides_path.write_text(
                    json.dumps({"test/model": {"available": False}}),
                    encoding="utf-8",
                )
                self.assertFalse(registry.list_models()[0]["available"])
                overrides_path.write_text(
                    json.dumps({
                        "test/model": {
                            "available": True,
                            "measured_latency_ms": 125,
                        }
                    }),
                    encoding="utf-8",
                )
                updated = registry.list_models()[0]
                self.assertTrue(updated["available"])
                self.assertEqual(updated["measured_latency_ms"], 125)


class InferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_failover_uses_next_candidate_after_timeout(self):
        class Response:
            status_code = 200
            is_error = False

            @staticmethod
            def json():
                return {
                    "choices": [{"message": {"content": "done"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }

        class Client:
            calls = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def post(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise httpx.TimeoutException("timed out")
                return Response()

        client = Client()
        with (
            patch.object(inference, "OPENROUTER_API_KEY", "test-key"),
            patch.object(inference.httpx, "AsyncClient", return_value=client),
        ):
            result = await inference.generate_response(
                [{"role": "user", "content": "hello"}],
                [{"id": "test/first"}, {"id": "test/second"}],
                latency_budget_ms=1000,
            )

        self.assertEqual(result["model"], "test/second")
        self.assertEqual(result["content"], "done")
        self.assertFalse(result["attempts"][0].get("success", False))
        self.assertTrue(result["attempts"][1]["success"])


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        api.app.state.registry = type(
            "Registry",
            (),
            {"models": [model()], "list_models": lambda self: list(self.models)},
        )()
        self.log_patch = patch.object(api, "_write_log")
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://test",
        )
        self.addAsyncCleanup(self.client.aclose)

    async def test_chat_returns_answer_and_explanation(self):
        generate = AsyncMock(return_value={
            "model": "test/free",
            "content": "Hello.",
            "latency_ms": 24,
            "total_latency_ms": 24,
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            "attempts": [{"model": "test/free", "success": True}],
        })
        with patch.object(api, "generate_response", generate):
            response = await self.client.post(
                "/api/chat",
                json={"prompt": "Say hello"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Hello.")
        self.assertEqual(response.json()["selected_model"], "test/free")
        self.assertIn("analysis", response.json())

    async def test_chat_uses_temperature_for_selected_mode(self):
        for mode, temperature in (("fast", 0.7), ("thinking", 0.3)):
            with self.subTest(mode=mode):
                generate = AsyncMock(return_value={
                    "model": "test/free",
                    "content": "Hello.",
                    "latency_ms": 24,
                    "total_latency_ms": 24,
                    "usage": {},
                    "attempts": [],
                })
                with patch.object(api, "generate_response", generate):
                    response = await self.client.post(
                        "/api/chat",
                        json={"prompt": "Say hello", "mode": mode},
                    )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    generate.await_args.kwargs["temperature"],
                    temperature,
                )

    async def test_only_paid_candidates_require_consent(self):
        api.app.state.registry.models = [model("test/paid", price="0.000001")]
        response = await self.client.post(
            "/api/chat",
            json={"prompt": "Say hello"},
        )
        self.assertEqual(response.status_code, 402)
        self.assertTrue(response.json()["paywall_required"])
        self.assertEqual(
            response.json()["paid_alternatives"][0]["id"],
            "test/paid",
        )

    async def test_no_valid_model_returns_constraint_explanation(self):
        api.app.state.registry.models = [model("test/short", context=128)]
        response = await self.client.post(
            "/api/chat",
            json={"prompt": "Say hello"},
        )
        self.assertEqual(response.status_code, 409)
        self.assertTrue(response.json()["rejected"])


if __name__ == "__main__":
    unittest.main()
