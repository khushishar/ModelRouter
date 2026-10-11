
from app.requirements_analyzer import analyze_requirements

request = (
    "Summarise this document into five bullet points. "
    "Keep the answer under 200 words."
)

result = analyze_requirements(
    request,
    input_modalities=["pdf"],
    input_token_count=5000,
    mode="fast",
)

print(result.model_dump_json(indent=2))
