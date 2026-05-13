from pathlib import Path

from transformers import pipeline


BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODELS = {
    "bert": BASE_DIR / "models" / "bert_ner",
    "roberta": BASE_DIR / "models" / "roberta_ner",
    "deberta": BASE_DIR / "models" / "deberta_ner",
}


print("\nAvailable Models:")
print("1. bert")
print("2. roberta")
print("3. deberta")

selected_model = input("\nSelect model: ").strip().lower()

if selected_model not in MODELS:
    print("\nInvalid model selected.")
    exit()

model_path = str(MODELS[selected_model])

print(f"\nLoading {selected_model} model...")

ner_pipeline = pipeline(
    "ner",
    model=model_path,
    tokenizer=model_path,
    aggregation_strategy="simple"
)

print("\nModel loaded successfully!")

text = input("\nEnter text:\n\n")

results = ner_pipeline(text)

print("\nNER Predictions:\n")

for entity in results:
    print(entity)