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

The gateway estimates tokens and task complexity locally before routing. Token
counts are approximate because provider tokenizers differ. It rejects models
that fail modality, price, context, or measured-latency constraints, ranks the
survivors, and shares one latency deadline across inference attempts. Unknown
catalogue latency is not presumed fast; the live deadline remains a hard limit.
Supported attachments are text, PDF, and image files. PDFs and images are only
sent to models whose catalogue advertises the required input modality.

Routing decisions, rejection reasons, model attempts, and cost/latency
metadata are appended to `data/routing_logs.jsonl`. User prompt and response
contents are deliberately not written to the routing log.

For routing demonstrations, create or edit `data/model_overrides.json`.
Overrides are reloaded for every chat request, without restarting the server:

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
`measured_latency_ms` to remove a latency measurement. Overrides are local
operator controls and are not exposed as an unauthenticated API.

Run the focused offline routing tests from this directory:

```powershell
.\.venv\Scripts\python.exe -m unittest test_router
```
