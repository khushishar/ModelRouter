import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

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
        "benchmarks": {
            "artificial_analysis": {"intelligence_index": 35}
        },
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

    def test_ambiguous_short_prompt_defaults_to_medium_safety_tier(self):
        result = analyze_prompt("What is a quasar?")
        self.assertEqual(result["complexity"], "medium")
        self.assertEqual(result["complexity_confidence"], "low")
        self.assertEqual(result["automatic_min_capability"], 30)
        self.assertIn(
            "ambiguous_prompt_defaulted_to_medium",
            result["complexity_signals"],
        )

    def test_long_or_complex_input_uses_higher_capability_floor(self):
        result = analyze_prompt("Review the entire contract carefully.")
        self.assertEqual(result["complexity"], "high")
        self.assertEqual(result["automatic_min_capability"], 40)

    def test_unambiguously_simple_greeting_has_no_automatic_floor(self):
        result = analyze_prompt("Hello!")
        self.assertEqual(result["complexity"], "low")
        self.assertIsNone(result["automatic_min_capability"])


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
    def test_fast_prefers_known_low_latency_and_thinking_prefers_reasoning(self):
        fast = model("test/fast", latency=100, reasoning=False)
        thinking = model("test/thinking", latency=300, reasoning=True)
        self.assertEqual(
            rank_models([thinking, fast], "fast")[0]["id"],
            "test/fast",
        )
        self.assertEqual(
            rank_models([fast, thinking], "thinking")[0]["id"],
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

    def test_registry_can_save_and_reset_judge_overrides(self):
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
                registry.set_override(
                    "test/model",
                    available=False,
                    measured_latency_ms=9000,
                )
                overridden = registry.list_models()[0]
                self.assertFalse(overridden["available"])
                self.assertEqual(overridden["measured_latency_ms"], 9000)
                registry.clear_override("test/model")
                self.assertTrue(registry.list_models()[0]["available"])
                self.assertEqual(registry.get_overrides(), {})

    def test_registry_rejects_invalid_latency_override(self):
        with tempfile.TemporaryDirectory() as directory:
            catalogue_path = Path(directory) / "catalogue.json"
            overrides_path = Path(directory) / "overrides.json"
            catalogue_path.write_text(
                json.dumps({"data": [{
                    "id": "test/model",
                    "context_length": 4096,
                    "architecture": {},
                    "pricing": {"prompt": "0", "completion": "0"},
                }]}),
                encoding="utf-8",
            )
            with (
                patch.object(model_registry, "CATALOGUE_PATH", catalogue_path),
                patch.object(model_registry, "MODEL_OVERRIDES_PATH", overrides_path),
            ):
                registry = model_registry.ModelRegistry()
                with self.assertRaises(ValueError):
                    registry.set_override(
                        "test/model",
                        available=True,
                        measured_latency_ms=float("inf"),
                    )



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

    async def test_judge_endpoints_list_update_and_reset_models(self):
        registry = type("Registry", (), {})()
        registry.models = [model("test/free", latency=100)]
        registry.list_models = lambda: registry.models
        registry.get_overrides = lambda: {}
        registry.set_override = Mock()
        registry.clear_override = Mock()
        api.app.state.registry = registry

        listed = await self.client.get("/api/judge/models")
        saved = await self.client.post(
            "/api/judge/overrides",
            json={
                "model_id": "test/free",
                "available": False,
                "measured_latency_ms": 9000,
            },
        )
        reset = await self.client.post(
            "/api/judge/overrides/reset",
            json={"model_id": "test/free"},
        )

        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["models"][0]["measured_latency_ms"], 100)
        self.assertEqual(saved.status_code, 200)
        registry.set_override.assert_called_once_with(
            "test/free",
            available=False,
            measured_latency_ms=9000,
        )
        self.assertEqual(reset.status_code, 200)
        registry.clear_override.assert_called_once_with("test/free")

    async def test_judge_mutations_reject_non_loopback_clients(self):
        remote_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=api.app,
                client=("192.0.2.10", 12345),
            ),
            base_url="http://test",
        )
        try:
            response = await remote_client.post(
                "/api/judge/overrides",
                json={
                    "model_id": "test/free",
                    "available": False,
                    "measured_latency_ms": 9000,
                },
            )
        finally:
            await remote_client.aclose()

        self.assertEqual(response.status_code, 403)

    async def test_chat_applies_automatic_capability_floor_for_ambiguous_prompt(self):
        weak = model("test/weak")
        weak["benchmarks"]["artificial_analysis"]["intelligence_index"] = 29
        strong = model("test/strong")
        strong["benchmarks"]["artificial_analysis"]["intelligence_index"] = 30
        api.app.state.registry.models = [weak, strong]
        generate = AsyncMock(return_value={
            "model": "test/strong",
            "content": "A quasar is a bright active galactic nucleus.",
            "latency_ms": 24,
            "total_latency_ms": 24,
            "usage": {"prompt_tokens": 6, "completion_tokens": 10},
            "attempts": [{"model": "test/strong", "success": True}],
        })
        with patch.object(api, "generate_response", generate):
            response = await self.client.post(
                "/api/chat",
                json={"prompt": "What is a quasar?"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["constraints"]["min_capability"], 30)
        self.assertEqual(payload["analysis"]["complexity_confidence"], "low")
        self.assertEqual(payload["selected_model"], "test/strong")
        rejected_weak = next(item for item in payload["rejected"] if item["id"] == "test/weak")
        self.assertTrue(any(
            reason["constraint"] == "capability"
            for reason in rejected_weak["reasons"]
        ))

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
