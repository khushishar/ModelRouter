# ModelRouter backend

The gateway uses `data/openrouter_models.json` as its local model catalogue.
Refresh that snapshot with:

```powershell
.\.venv\Scripts\python.exe -m app.catalogue
```

Set `OPENROUTER_API_KEY` in `backend/.env`, then start the API from the
`backend` directory:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api:app --reload
```

The frontend expects the API at `http://localhost:8000`; override that with
`VITE_API_BASE_URL` when needed. The `/api/chat` request accepts `prompt`,
`mode` (`fast` or `thinking`), optional attachments, `allow_paid` (false by
default), `max_cost_usd`, `max_latency_ms`, and `max_output_tokens`.
Callers can also supply `min_capability` (when catalogue benchmark data is
available) and `required_context_tokens`; estimated input plus requested output
tokens always sets the minimum required context.

Each chat request may include a UUID `conversation_id`. The frontend creates
one per chat and reuses it for subsequent turns; the backend sends up to the
last 10 successful turns back to the model. Conversation history is kept only
in memory, is removed after an hour of inactivity, and is cleared when the
backend restarts. Memory is bounded by conversation count and size. Start a
new chat to get an independent conversation. History tokens are included in
context and estimated cost checks.

The gateway estimates tokens and task complexity locally before routing. Token
counts are approximate because provider tokenizers differ. Ambiguous short
prompts default to medium complexity instead of being assumed simple. Medium
and high complexity apply minimum Artificial Analysis benchmark floors of 30
and 40 respectively; a caller-specified floor can only raise that minimum.
Models without the relevant benchmark are rejected when a floor applies. These
heuristics are conservative estimates, not a substitute for an explicit
capability requirement. The router rejects models that fail modality, price,
context, capability, availability, or measured-latency constraints, ranks the
survivors, and shares one latency deadline across inference attempts. Unknown
catalogue latency is not presumed fast; the live deadline remains a hard limit.
Supported attachments are text, PDF, and image files. PDFs are converted to
text in the backend before routing, so a text-capable model can act on their
contents. Image-only/scanned PDFs are rejected with an explicit message because
OCR is not enabled. Images remain multimodal inputs and are routed only to
models whose catalogue advertises image input support.
Prompts that ask to create or draw an image are routed as text input with image
output; only models advertising image output and a known image-output price can
qualify. Image generation requires enabling paid models when the image-output
price is non-zero. Generated images are returned in the chat response and
displayed in the conversation.

Ranking uses a weighted score normalized across the eligible models for each
request. Fast mode weights latency 75%, estimated cost 15%, and task capability
10%. Thinking mode weights latency 55%, reasoning support 25%, task capability
15%, and estimated cost 5%. Missing metric values receive no utility for that
metric. Temperature is 0.7 in fast mode and 0.3 in thinking mode.

Routing decisions, rejection reasons, model attempts, and cost/latency
metadata are appended to `data/routing_logs.jsonl`. User prompt and response
contents are deliberately not written to the routing log.

For routing demonstrations, use the frontend's **Judge controls** panel to
change a model's availability or simulated latency. Overrides persist in
`data/model_overrides.json`, are reloaded for every chat request, and can be
reset to catalogue defaults from the panel. The judge-control endpoints accept
changes only from loopback clients; do not remove that restriction when
exposing the backend to a network.

Overrides can also be edited directly in `data/model_overrides.json`:

```json
{
  "provider/model-id": {
    "available": false,
    "measured_latency_ms": 12000
  }
}
```

Set `available` to `false` to hard-reject a model, or set a latency above a
request's budget to force selection of another valid candidate. Use `null` for
`measured_latency_ms` to remove a latency measurement.

Assistant responses are rendered as Markdown with GitHub-flavored tables and
lists. Raw HTML is not enabled.

Run the focused offline routing tests from this directory:

```powershell
.\.venv\Scripts\python.exe -m unittest test_router
```
