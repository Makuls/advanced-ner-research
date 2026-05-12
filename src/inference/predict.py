from transformers import pipeline

MODEL_PATH = "../training/models/bert_ner"

ner_pipeline = pipeline(
    "ner",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    aggregation_strategy="simple"
)

text = """
Apple is opening a new office in Bangalore and Sundar Pichai will visit India next month.
"""

results = ner_pipeline(text)

print("\nNER Predictions:\n")

for entity in results:
    print(entity)