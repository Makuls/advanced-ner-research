```python
"""Train a DeBERTa-v3-Small baseline for CoNLL-2003 Named Entity Recognition."""

from __future__ import annotations

import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from datasets import DatasetDict, load_dataset

from seqeval.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

LOGGER = logging.getLogger(__name__)

IGNORE_INDEX = -100


@dataclass(frozen=True)
class TrainingConfig:

    model_name: str = "microsoft/deberta-v3-small"

    dataset_name: str = "conll2003"

    output_dir: Path = Path("models/deberta_small_ner")

    logging_dir: Path = Path("logs/deberta_small_ner")

    max_length: int = 64

    num_train_epochs: float = 5.0

    train_batch_size: int = 1

    eval_batch_size: int = 1

    learning_rate: float = 5e-5

    weight_decay: float = 0.01

    warmup_ratio: float = 0.06

    seed: int = 42

    label_all_tokens: bool = False


def parse_args() -> TrainingConfig:

    parser = argparse.ArgumentParser(
        description="Train DeBERTa-v3-Small on CoNLL2003 NER."
    )

    parser.add_argument(
        "--model-name",
        default="microsoft/deberta-v3-small"
    )

    parser.add_argument(
        "--dataset-name",
        default="conll2003"
    )

    parser.add_argument(
        "--output-dir",
        default="models/deberta_small_ner"
    )

    parser.add_argument(
        "--logging-dir",
        default="logs/deberta_small_ner"
    )

    args = parser.parse_args()

    return TrainingConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        output_dir=Path(args.output_dir),
        logging_dir=Path(args.logging_dir),
    )


def setup_logging() -> None:

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )


def get_device_name() -> str:

    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def build_b_to_i_label_map(
    label_list: list[str]
) -> list[int]:

    label_to_id = {
        label: idx
        for idx, label in enumerate(label_list)
    }

    b_to_i_label = []

    for label in label_list:

        if label.startswith("B-"):

            i_label = f"I-{label[2:]}"

            b_to_i_label.append(
                label_to_id.get(
                    i_label,
                    label_to_id[label]
                )
            )

        else:
            b_to_i_label.append(
                label_to_id[label]
            )

    return b_to_i_label


def load_dataset_data(
    dataset_name: str
) -> tuple[DatasetDict, list[str]]:

    LOGGER.info("Loading dataset...")

    dataset = load_dataset(dataset_name)

    label_feature = dataset["train"].features[
        "ner_tags"
    ].feature

    label_list = list(label_feature.names)

    LOGGER.info(
        "Loaded dataset with splits: %s",
        {split: len(records) for split, records in dataset.items()},
    )

    LOGGER.info("NER labels: %s", label_list)

    return dataset, label_list


def tokenize_and_align_labels(
    examples: dict[str, list[Any]],
    tokenizer,
    b_to_i_label,
    max_length: int,
    label_all_tokens: bool,
):

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )

    labels = []

    for i, label in enumerate(examples["ner_tags"]):

        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )

        previous_word_idx = None

        label_ids = []

        for word_idx in word_ids:

            if word_idx is None:

                label_ids.append(
                    IGNORE_INDEX
                )

            elif word_idx != previous_word_idx:

                label_ids.append(
                    label[word_idx]
                )

            elif label_all_tokens:

                label_ids.append(
                    b_to_i_label[label[word_idx]]
                )

            else:

                label_ids.append(
                    IGNORE_INDEX
                )

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


def compute_metrics(
    eval_prediction: EvalPrediction,
    label_list: list[str]
):

    predictions, labels = eval_prediction

    predictions = np.argmax(
        predictions,
        axis=2
    )

    true_predictions = []

    true_labels = []

    for prediction, label in zip(
        predictions,
        labels
    ):

        current_predictions = []
        current_labels = []

        for p, l in zip(prediction, label):

            if l != IGNORE_INDEX:

                current_predictions.append(
                    label_list[p]
                )

                current_labels.append(
                    label_list[l]
                )

        true_predictions.append(
            current_predictions
        )

        true_labels.append(
            current_labels
        )

    return {
        "precision": precision_score(
            true_labels,
            true_predictions
        ),
        "recall": recall_score(
            true_labels,
            true_predictions
        ),
        "f1": f1_score(
            true_labels,
            true_predictions
        ),
        "accuracy": accuracy_score(
            true_labels,
            true_predictions
        ),
    }


def train_model(
    config: TrainingConfig
):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_seed(config.seed)

    config.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    config.logging_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    LOGGER.info(
        "Using device: %s",
        get_device_name()
    )

    raw_dataset, label_list = load_dataset_data(
        config.dataset_name
    )

    id2label = {
        idx: label
        for idx, label in enumerate(label_list)
    }

    label2id = {
        label: idx
        for idx, label in id2label.items()
    }

    b_to_i_label = build_b_to_i_label_map(
        label_list
    )

    LOGGER.info(
        "Loading tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=False
    )

    LOGGER.info(
        "Loading model..."
    )

    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    model.gradient_checkpointing_enable()

    LOGGER.info(
        "Tokenizing dataset..."
    )

    tokenized_dataset = raw_dataset.map(
        lambda examples: tokenize_and_align_labels(
            examples,
            tokenizer,
            b_to_i_label,
            config.max_length,
            config.label_all_tokens,
        ),
        batched=True,
        remove_columns=raw_dataset[
            "train"
        ].column_names,
    )

    data_collator = (
        DataCollatorForTokenClassification(
            tokenizer=tokenizer
        )
    )

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),

        logging_dir=str(config.logging_dir),

        eval_strategy="epoch",

        save_strategy="epoch",

        learning_rate=config.learning_rate,

        per_device_train_batch_size=config.train_batch_size,

        per_device_eval_batch_size=config.eval_batch_size,

        gradient_accumulation_steps=4,

        num_train_epochs=config.num_train_epochs,

        weight_decay=config.weight_decay,

        warmup_ratio=config.warmup_ratio,

        logging_steps=50,

        save_total_limit=2,

        load_best_model_at_end=True,

        metric_for_best_model="f1",

        greater_is_better=True,

        report_to="none",

        fp16=False,
    )

    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=tokenized_dataset["train"],

        eval_dataset=tokenized_dataset["validation"],

        data_collator=data_collator,

        compute_metrics=lambda p: compute_metrics(
            p,
            label_list
        ),
    )

    LOGGER.info("Starting training...")

    trainer.train()

    LOGGER.info("Evaluating model...")

    validation_metrics = trainer.evaluate(
        metric_key_prefix="validation"
    )

    LOGGER.info(validation_metrics)

    LOGGER.info("Saving model...")

    trainer.save_model(
        str(config.output_dir)
    )

    tokenizer.save_pretrained(
        str(config.output_dir)
    )

    trainer.save_metrics(
        "validation",
        validation_metrics
    )

    trainer.save_state()

    LOGGER.info("Training complete.")


def main():

    setup_logging()

    config = parse_args()

    LOGGER.info(config)

    train_model(config)


if __name__ == "__main__":

    main()