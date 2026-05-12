from transformers import pipeline

MODEL_PATH = "../training/models/bert_ner"

ner_pipeline = pipeline(
    "ner",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    aggregation_strategy="simple"
)

text = """
Apple hired Sundar Pichai in California and opened a new office in Bangalore.
"""

results = ner_pipeline(text)

print("\n========== ENTITY RESULTS ==========\n")

for entity in results:
    word = entity["word"]
    label = entity["entity_group"]
    score = round(float(entity["score"]), 4)

    print(f"{word:<20} → {label:<5} | Confidence: {score}")