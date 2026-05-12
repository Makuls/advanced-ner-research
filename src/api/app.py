from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

MODEL_PATH = "../training/models/bert_ner"

ner_pipeline = pipeline(
    "ner",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    aggregation_strategy="simple"
)

class TextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "NER API is running"}


@app.post("/predict")
def predict(request: TextRequest):

    results = ner_pipeline(request.text)

    formatted_results = []

    for entity in results:
        formatted_results.append({
            "entity": entity["word"],
            "label": entity["entity_group"],
            "score": float(entity["score"])
        })

    return {
        "text": request.text,
        "entities": formatted_results
    }