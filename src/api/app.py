from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline


app = FastAPI()


BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATHS = {
    "bert": BASE_DIR / "models" / "bert_ner",
    "roberta": BASE_DIR / "models" / "roberta_ner",
    "deberta": BASE_DIR / "models" / "deberta_ner",
}


loaded_pipelines = {}


class TextRequest(BaseModel):
    text: str
    model: str


@app.get("/")
def home():

    return {
        "message": "Advanced NER API is running"
    }


def get_pipeline(model_name: str):

    if model_name not in loaded_pipelines:

        model_path = str(
            MODEL_PATHS[model_name]
        )

        loaded_pipelines[model_name] = pipeline(
            "ner",
            model=model_path,
            tokenizer=model_path,
            aggregation_strategy="simple"
        )

    return loaded_pipelines[model_name]


@app.post("/predict")
def predict(request: TextRequest):

    selected_model = request.model.lower()

    if selected_model not in MODEL_PATHS:

        return {
            "error": "Invalid model selected"
        }

    ner_pipeline = get_pipeline(
        selected_model
    )

    results = ner_pipeline(
        request.text
    )

    formatted_results = []

    for entity in results:

        formatted_results.append({
            "entity": entity["word"],
            "label": entity["entity_group"],
            "score": float(entity["score"])
        })

    return {
        "model": selected_model,
        "text": request.text,
        "entities": formatted_results
    }