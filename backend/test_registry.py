from app.model_registry import ModelRegistry

registry = ModelRegistry()
models = registry.list_models()

print("\nTotal models:", len(models))
print("\nFirst 5 normalized models:")

for model in models[:5]:
    print({
        "id": model["id"],
        "context_length": model["context_length"],
        "input_modalities": model["input_modalities"],
        "output_modalities": model["output_modalities"],
        "is_free": model["is_free"],
    })

print("\nFree models:", sum(m["is_free"] for m in models))
print("Text-input models:", sum(
    "text" in m["input_modalities"] for m in models
))
print("Image-input models:", sum(
    "image" in m["input_modalities"] for m in models
))