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

    def test_image_generation_is_text_input_and_image_output(self):
        result = analyze_prompt("Make an image of a cat.")
        self.assertEqual(result["task"], "image_generation")
        self.assertEqual(result["output_modality"], "image")
        self.assertEqual(
            analyze_prompt("What is in this picture?")["output_modality"],
            "text",
        )

    def test_pdf_text_is_extracted_before_building_model_message(self):
        attachment = api.Attachment(
            name="brief.pdf",
            mime_type="application/pdf",
            data="dGVzdA==",
        )
        reader = type(
            "Reader",
            (),
            {
                "is_encrypted": False,
                "pages": [
                    type("Page", (), {"extract_text": lambda self: "Extracted PDF text"})()
                ],
            },
        )()
        with patch.object(api, "PdfReader", return_value=reader):
            mime_type, extracted = api._decode_attachment(attachment)

        self.assertEqual(mime_type, "text/plain")
        self.assertEqual(extracted, b"Extracted PDF text")
        message, modalities, _ = api._build_user_message(
            "",
            [(attachment, mime_type, extracted)],
        )
        self.assertEqual(modalities, ["text"])
        self.assertIn("Extracted PDF text", message["content"])

    def test_scanned_pdf_without_extractable_text_returns_clear_error(self):
        attachment = api.Attachment(
            name="scan.pdf",
            mime_type="application/pdf",
            data="dGVzdA==",
        )
        reader = type(
            "Reader",
            (),
            {
                "is_encrypted": False,
                "pages": [
                    type("Page", (), {"extract_text": lambda self: ""})()
                ],
            },
        )()
        with patch.object(api, "PdfReader", return_value=reader):
            with self.assertRaises(api.HTTPException) as error:
                api._decode_attachment(attachment)

        self.assertEqual(error.exception.status_code, 422)
        self.assertIn("OCR", error.exception.detail)

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

    def test_reasoning_support_is_fallback_when_task_benchmark_is_missing(self):
        reasoning_model = model("test/reasoning", reasoning=True)
        reasoning_model.pop("benchmarks")
        pipeline = run_pipeline(
            [reasoning_model],
            task="reasoning",
            min_capability=30,
            required_context=1024,
        )

        self.assertEqual([item["id"] for item in pipeline["eligible"]], [
            "test/reasoning",
        ])
        self.assertIsNone(pipeline["eligible"][0]["_router_capability"])

    def test_reasoning_fallback_does_not_bypass_low_numeric_benchmark(self):
        reasoning_model = model("test/weak-reasoning", reasoning=True)
        reasoning_model["benchmarks"]["artificial_analysis"][
            "intelligence_index"
        ] = 20
        pipeline = run_pipeline(
            [reasoning_model],
            task="reasoning",
            min_capability=30,
            required_context=1024,
        )

        self.assertEqual(pipeline["eligible"], [])
        self.assertIn(
            "Capability 20 is below minimum 30",
            [
                reason["reason"]
                for reason in pipeline["rejected"][0]["reasons"]
            ],
        )

    def test_image_generation_requires_supported_and_priced_image_output(self):
        text_only = model("test/text-only")
        image_capable = model("test/image", price="0")
        image_capable["architecture"]["output_modalities"] = ["text", "image"]
        image_capable["pricing"]["image_output"] = "0.001"
        pipeline = run_pipeline(
            [text_only, image_capable],
            input_modalities=["text"],
            output_modality="image",
            estimated_output_images=1,
            free_only=False,
            paid_consent=True,
            max_cost_usd=0.01,
            required_context=1024,
        )
        self.assertEqual([item["id"] for item in pipeline["eligible"]], ["test/image"])
        self.assertAlmostEqual(
            pipeline["eligible"][0]["_router_estimated_cost_usd"],
            0.001,
        )
        self.assertTrue(any(
            reason["constraint"] == "modality"
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

    def test_registry_can_save_and_reset_preference_overrides(self):
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

    def test_registry_persists_preferred_model_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            catalogue_path = Path(directory) / "catalogue.json"
            preferences_path = Path(directory) / "preferences.json"
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
                patch.object(
                    model_registry,
                    "MODEL_PREFERENCES_PATH",
                    preferences_path,
                ),
            ):
                registry = model_registry.ModelRegistry()
                self.assertIsNone(registry.get_preferred_model_id())
                registry.set_preferred_model_id("test/model")
                self.assertEqual(
                    registry.get_preferred_model_id(),
                    "test/model",
                )
                registry.set_preferred_model_id(None)
                self.assertIsNone(registry.get_preferred_model_id())

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

    async def test_image_only_response_is_success_and_requests_image_modality(self):
        image_url = "data:image/png;base64,aW1hZ2U="

        class Response:
            status_code = 200
            is_error = False

            @staticmethod
            def json():
                return {
                    "choices": [{
                        "message": {
                            "content": None,
                            "images": [{
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            }],
                        },
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 0},
                }

        class Client:
            payload = None
            headers = None

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def post(self, *_args, **kwargs):
                self.payload = kwargs["json"]
                self.headers = kwargs["headers"]
                return Response()

        client = Client()
        with (
            patch.object(inference, "OPENROUTER_API_KEY", "test-key"),
            patch.object(inference.httpx, "AsyncClient", return_value=client),
        ):
            result = await inference.generate_response(
                [{"role": "user", "content": "Make an image."}],
                [{"id": "test/image"}],
                latency_budget_ms=1000,
                output_modality="image",
            )

        self.assertEqual(result["images"], [image_url])
        self.assertEqual(result["content"], "")
        self.assertEqual(client.payload["modalities"], ["image", "text"])
        self.assertEqual(client.headers["Authorization"], "Bearer test-key")

    async def test_provider_rejection_reports_status_and_provider_message(self):
        response = httpx.Response(
            401,
            json={
                "error": {
                    "code": 401,
                    "message": "Invalid API key",
                }
            },
        )

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def post(self, *_args, **_kwargs):
                return response

        with (
            patch.object(inference, "OPENROUTER_API_KEY", "test-key"),
            patch.object(inference.httpx, "AsyncClient", return_value=Client()),
        ):
            with self.assertRaises(inference.InferenceError) as error:
                await inference.generate_response(
                    [{"role": "user", "content": "hello"}],
                    [{"id": "test/first"}, {"id": "test/second"}],
                    latency_budget_ms=5000,
                )

        self.assertIn("provider", str(error.exception).lower())
        self.assertIn("HTTP 401", str(error.exception))
        self.assertIn("Invalid API key", str(error.exception))
        self.assertNotIn("latency budget expired", str(error.exception).lower())
        self.assertEqual(len(error.exception.attempts), 2)


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        api.app.state.registry = type(
            "Registry",
            (),
            {
                "models": [model()],
                "list_models": lambda self: list(self.models),
                "get_preferred_model_id": lambda self: None,
            },
        )()
        api.app.state.conversations = {}
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

    async def test_preference_endpoints_list_update_and_reset_models(self):
        registry = type("Registry", (), {})()
        registry.models = [model("test/free", latency=100)]
        registry.list_models = lambda: registry.models
        registry.get_overrides = lambda: {}
        registry.get_preferred_model_id = lambda: None
        registry.set_override = Mock()
        registry.clear_override = Mock()
        registry.set_preferred_model_id = Mock()
        api.app.state.registry = registry

        listed = await self.client.get("/api/preferences/models")
        saved = await self.client.post(
            "/api/preferences/overrides",
            json={
                "model_id": "test/free",
                "available": False,
                "measured_latency_ms": 9000,
            },
        )
        reset = await self.client.post(
            "/api/preferences/overrides/reset",
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

    async def test_preferred_model_is_saved_globally(self):
        registry = Mock()
        registry.models = [model("test/selected")]
        registry.set_preferred_model_id.return_value = None
        api.app.state.registry = registry

        response = await self.client.post(
            "/api/preferences/model",
            json={"model_id": "test/selected"},
        )

        self.assertEqual(response.status_code, 200)
        registry.set_preferred_model_id.assert_called_once_with("test/selected")

    async def test_chat_uses_only_the_globally_preferred_model(self):
        preferred = model("test/preferred")
        fallback = model("test/fallback", latency=1)
        api.app.state.registry.models = [preferred, fallback]
        api.app.state.registry.get_preferred_model_id = lambda: "test/preferred"
        generate = AsyncMock(return_value={
            "model": "test/preferred",
            "content": "Hello.",
            "latency_ms": 24,
            "total_latency_ms": 24,
            "usage": {},
            "attempts": [],
        })

        with patch.object(api, "generate_response", generate):
            response = await self.client.post(
                "/api/chat",
                json={"prompt": "Say hello"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item["id"] for item in generate.await_args.args[1]],
            ["test/preferred"],
        )
        self.assertEqual(
            response.json()["constraints"]["preferred_model_id"],
            "test/preferred",
        )

    async def test_chat_does_not_fallback_when_preferred_model_is_ineligible(self):
        preferred = model("test/preferred", context=128)
        fallback = model("test/fallback")
        api.app.state.registry.models = [preferred, fallback]
        api.app.state.registry.get_preferred_model_id = lambda: "test/preferred"
        generate = AsyncMock()

        with patch.object(api, "generate_response", generate):
            response = await self.client.post(
                "/api/chat",
                json={"prompt": "Say hello"},
            )

        self.assertEqual(response.status_code, 409)
        generate.assert_not_awaited()

    async def test_preferences_are_available_to_remote_clients(self):
        registry = Mock()
        registry.models = [model("test/free", latency=100)]
        registry.list_models.return_value = registry.models
        registry.get_overrides.return_value = {}
        registry.get_preferred_model_id.return_value = None
        registry.set_preferred_model_id.return_value = None
        api.app.state.registry = registry
        remote_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=api.app,
                client=("192.0.2.10", 12345),
            ),
            base_url="http://test",
        )
        try:
            listed = await remote_client.get("/api/preferences/models")
            saved = await remote_client.post(
                "/api/preferences/overrides",
                json={
                    "model_id": "test/free",
                    "available": False,
                    "measured_latency_ms": 9000,
                },
            )
            reset = await remote_client.post(
                "/api/preferences/overrides/reset",
                json={"model_id": "test/free"},
            )
        finally:
            await remote_client.aclose()

        self.assertEqual(listed.status_code, 200)
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(reset.status_code, 200)
        registry.set_override.assert_called_once_with(
            "test/free",
            available=False,
            measured_latency_ms=9000,
        )
        registry.clear_override.assert_called_once_with("test/free")

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
        rejected_weak = next(
            item for item in payload["rejected"] if item["id"] == "test/weak"
        )
        self.assertTrue(any(
            reason["constraint"] == "capability"
            for reason in rejected_weak["reasons"]
        ))

    async def test_image_request_filters_for_image_output_and_returns_images(self):
        image_model = model("test/image", price="0")
        image_model["architecture"]["output_modalities"] = ["text", "image"]
        image_model["pricing"]["image_output"] = "0.001"
        api.app.state.registry.models = [image_model]
        image_url = "data:image/png;base64,aW1hZ2U="
        generate = AsyncMock(return_value={
            "model": "test/image",
            "content": "",
            "images": [image_url],
            "latency_ms": 24,
            "total_latency_ms": 24,
            "usage": {"prompt_tokens": 2, "completion_tokens": 0},
            "attempts": [{"model": "test/image", "success": True}],
        })
        with patch.object(api, "generate_response", generate):
            response = await self.client.post(
                "/api/chat",
                json={"prompt": "Make an image of a cat", "allow_paid": True},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["constraints"]["input_modalities"], ["text"])
        self.assertEqual(payload["constraints"]["output_modality"], "image")
        self.assertEqual(payload["images"], [image_url])
        self.assertEqual(
            generate.await_args.kwargs["output_modality"],
            "image",
        )

    async def test_uploaded_image_is_input_modality_not_requested_output(self):
        image_model = model("test/image-input", price="0", modalities=["text", "image"])
        api.app.state.registry.models = [image_model]
        generate = AsyncMock(return_value={
            "model": "test/image-input",
            "content": "It is a cat.",
            "images": [],
            "latency_ms": 24,
            "total_latency_ms": 24,
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            "attempts": [{"model": "test/image-input", "success": True}],
        })
        with patch.object(api, "generate_response", generate):
            response = await self.client.post(
                "/api/chat",
                json={
                    "prompt": "What is in this picture?",
                    "attachments": [{
                        "name": "cat.png",
                        "mime_type": "image/png",
                        "data": "data:image/png;base64,aW1hZ2U=",
                    }],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["constraints"]["input_modalities"],
            ["text", "image"],
        )
        self.assertEqual(response.json()["constraints"]["output_modality"], "text")
        self.assertEqual(generate.await_args.kwargs["output_modality"], "text")

    async def test_chat_sends_previous_turns_for_same_conversation(self):
        generate = AsyncMock(side_effect=[
            {
                "model": "test/free",
                "content": "The document says revenue grew.",
                "latency_ms": 24,
                "total_latency_ms": 24,
                "usage": {"prompt_tokens": 8, "completion_tokens": 6},
                "attempts": [{"model": "test/free", "success": True}],
            },
            {
                "model": "test/free",
                "content": "It grew by 12%.",
                "latency_ms": 24,
                "total_latency_ms": 24,
                "usage": {"prompt_tokens": 20, "completion_tokens": 5},
                "attempts": [{"model": "test/free", "success": True}],
            },
        ])
        conversation_id = "5a515808-59cc-4d26-8ee7-c8f4e9b7e979"
        with patch.object(api, "generate_response", generate):
            first = await self.client.post(
                "/api/chat",
                json={
                    "conversation_id": conversation_id,
                    "prompt": "The report says revenue grew by 12%.",
                },
            )
            second = await self.client.post(
                "/api/chat",
                json={
                    "conversation_id": conversation_id,
                    "prompt": "How much did it grow?",
                },
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["conversation_id"], conversation_id)
        second_request_messages = generate.await_args_list[1].args[0]
        self.assertEqual(
            [message["role"] for message in second_request_messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertIn(
            "revenue grew by 12%",
            second_request_messages[1]["content"],
        )
        self.assertEqual(
            second_request_messages[2]["content"],
            "The document says revenue grew.",
        )
        self.assertGreater(
            second.json()["analysis"]["estimated_input_tokens"],
            estimate_tokens("How much did it grow?"),
        )

    async def test_distinct_conversation_ids_do_not_share_history(self):
        generate = AsyncMock(return_value={
            "model": "test/free",
            "content": "Answer.",
            "latency_ms": 24,
            "total_latency_ms": 24,
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            "attempts": [{"model": "test/free", "success": True}],
        })
        with patch.object(api, "generate_response", generate):
            await self.client.post(
                "/api/chat",
                json={
                    "conversation_id": "a9179f00-897e-4437-b83b-46b8d3f96c3b",
                    "prompt": "first chat secret",
                },
            )
            await self.client.post(
                "/api/chat",
                json={
                    "conversation_id": "ba148adb-d94c-472f-aaf1-6f432e827840",
                    "prompt": "second chat question",
                },
            )

        second_messages = generate.await_args_list[1].args[0]
        self.assertEqual(len(second_messages), 2)
        self.assertNotIn("first chat secret", str(second_messages))

    async def test_chat_uses_extracted_pdf_as_text_model_input(self):
        api.app.state.registry.models[0]["benchmarks"]["artificial_analysis"][
            "intelligence_index"
        ] = 45
        generate = AsyncMock(return_value={
            "model": "test/free",
            "content": "PDF summary.",
            "latency_ms": 24,
            "total_latency_ms": 24,
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            "attempts": [{"model": "test/free", "success": True}],
        })
        with (
            patch.object(
                api,
                "_decode_attachment",
                return_value=("text/plain", b"Contract clause text"),
            ),
            patch.object(api, "generate_response", generate),
        ):
            response = await self.client.post(
                "/api/chat",
                json={
                    "prompt": "Summarize this",
                    "attachments": [{
                        "name": "contract.pdf",
                        "mime_type": "application/pdf",
                        "data": "cGRm",
                    }],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["constraints"]["input_modalities"],
            ["text"],
        )
        call_messages = generate.await_args.args[0]
        self.assertIn("Contract clause text", call_messages[1]["content"])
        self.assertNotIn('"type": "file"', str(call_messages[1]["content"]))
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

    async def test_unhandled_chat_error_keeps_cors_headers(self):
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=api.app,
                raise_app_exceptions=False,
            ),
            base_url="http://test",
        )
        await self.enterAsyncContext(client)

        with patch.object(
            api,
            "generate_response",
            side_effect=RuntimeError("Unexpected inference failure"),
        ):
            response = await client.post(
                "/api/chat",
                headers={"Origin": "https://modelrouter-1.onrender.com"},
                json={"prompt": "Say hello"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "https://modelrouter-1.onrender.com",
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
