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


selected_model = input(
    "\nSelect model: "
).strip().lower()


if selected_model not in MODELS:

    print("\nInvalid model selected.")
    exit()


model_path = str(
    MODELS[selected_model]
)


print(f"\nLoading {selected_model} model...\n")


ner_pipeline = pipeline(
    "ner",
    model=model_path,
    tokenizer=model_path,
    aggregation_strategy="simple"
)


text = input(
    "\nEnter text for visualization:\n\n"
)


results = ner_pipeline(text)


print("\n========== ENTITY RESULTS ==========\n")


if len(results) == 0:

    print("No entities detected.")

else:

    for entity in results:

        word = entity["word"]

        label = entity["entity_group"]

        score = round(
            float(entity["score"]),
            4
        )

        print(
            f"{word:<20} → {label:<5} | Confidence: {score}"
        )