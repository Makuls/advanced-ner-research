"""Train a BERT baseline for CoNLL-2003 Named Entity Recognition.

This script is intentionally structured as a clean research baseline:
it is easy to run from VS Code, easy to extend for multilingual NER, and
uses standard Hugging Face + PyTorch tooling throughout.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    BertForTokenClassification,
    BertTokenizerFast,
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
    """Configuration for the baseline BERT NER experiment."""

    model_name: str = "bert-base-cased"
    dataset_name: str = "conll2003"
    output_dir: Path = Path("models/bert_ner")
    logging_dir: Path = Path("logs/bert_ner")
    max_length: int = 128
    num_train_epochs: float = 3.0
    train_batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    label_all_tokens: bool = False


def parse_args() -> TrainingConfig:
    """Parse command-line arguments into a typed training configuration."""

    parser = argparse.ArgumentParser(
        description="Train a bert-base-cased token classification model on CoNLL-2003."
    )
    parser.add_argument("--model-name", default="bert-base-cased")
    parser.add_argument("--dataset-name", default="conll2003")
    parser.add_argument("--output-dir", default="models/bert_ner")
    parser.add_argument("--logging-dir", default="logs/bert_ner")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label-all-tokens",
        action="store_true",
        help=(
            "Propagate labels to all subword pieces. By default, only the first "
            "piece of each word contributes to the loss."
        ),
    )

    args = parser.parse_args()
    return TrainingConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        output_dir=Path(args.output_dir),
        logging_dir=Path(args.logging_dir),
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        label_all_tokens=args.label_all_tokens,
    )


def setup_logging() -> None:
    """Configure console logging for training and debugging."""

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def get_device_name() -> str:
    """Return a human-readable accelerator name for logging."""

    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def build_b_to_i_label_map(label_list: list[str]) -> list[int]:
    """Map B-* labels to their I-* counterpart for continued subword pieces."""

    label_to_id = {label: index for index, label in enumerate(label_list)}
    b_to_i_label: list[int] = []

    for label in label_list:
        if label.startswith("B-"):
            i_label = f"I-{label[2:]}"
            b_to_i_label.append(label_to_id.get(i_label, label_to_id[label]))
        else:
            b_to_i_label.append(label_to_id[label])

    return b_to_i_label


def load_dataset_data(dataset_name: str = "conll2003") -> tuple[DatasetDict, list[str]]:
    """Load CoNLL-style data and return the dataset with its NER label names."""

    LOGGER.info("Loading dataset: %s", dataset_name)
    dataset = load_dataset(dataset_name)

    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected a DatasetDict from {dataset_name}, got {type(dataset)!r}.")

    label_feature = dataset["train"].features["ner_tags"].feature
    label_list = list(label_feature.names)

    LOGGER.info(
        "Loaded %s with splits: %s",
        dataset_name,
        {split: len(records) for split, records in dataset.items()},
    )
    LOGGER.info("NER labels: %s", label_list)
    return dataset, label_list


def tokenize_and_align_labels(
    examples: dict[str, list[Any]],
    tokenizer: BertTokenizerFast,
    label_all_tokens: bool = False,
    max_length: int = 128,
    b_to_i_label: list[int] | None = None,
) -> dict[str, Any]:
    """Tokenize word-level examples and align NER tags to BERT subword tokens.

    Special tokens and ignored subword pieces receive IGNORE_INDEX so that the
    loss function and evaluation do not treat them as true labels.
    """

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )

    aligned_labels: list[list[int]] = []
    for example_index, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=example_index)
        previous_word_id: int | None = None
        label_ids: list[int] = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(IGNORE_INDEX)
            elif word_id != previous_word_id:
                label_ids.append(labels[word_id])
            elif label_all_tokens:
                label_id = labels[word_id]
                label_ids.append(b_to_i_label[label_id] if b_to_i_label else label_id)
            else:
                label_ids.append(IGNORE_INDEX)

            previous_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


def compute_metrics(eval_prediction: EvalPrediction, label_list: list[str]) -> dict[str, float]:
    """Compute seqeval entity-level metrics for Trainer evaluation."""

    predictions, labels = eval_prediction
    predicted_label_ids = np.argmax(predictions, axis=-1)

    true_predictions: list[list[str]] = []
    true_labels: list[list[str]] = []

    for prediction_row, label_row in zip(predicted_label_ids, labels, strict=True):
        sentence_predictions: list[str] = []
        sentence_labels: list[str] = []

        for predicted_id, label_id in zip(prediction_row, label_row, strict=True):
            if label_id == IGNORE_INDEX:
                continue

            sentence_predictions.append(label_list[int(predicted_id)])
            sentence_labels.append(label_list[int(label_id)])

        true_predictions.append(sentence_predictions)
        true_labels.append(sentence_labels)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }


def train_model(config: TrainingConfig) -> Trainer:
    """Build, train, evaluate, and save the BERT NER model."""

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.logging_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Using device: %s", get_device_name())

    raw_dataset, label_list = load_dataset_data(config.dataset_name)
    id2label = {index: label for index, label in enumerate(label_list)}
    label2id = {label: index for index, label in id2label.items()}
    b_to_i_label = build_b_to_i_label_map(label_list)

    LOGGER.info("Loading tokenizer and model: %s", config.model_name)
    tokenizer = BertTokenizerFast.from_pretrained(config.model_name)
    model = BertForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    LOGGER.info("Tokenizing dataset and aligning labels.")
    tokenized_dataset = raw_dataset.map(
        lambda examples: tokenize_and_align_labels(
            examples=examples,
            tokenizer=tokenizer,
            label_all_tokens=config.label_all_tokens,
            max_length=config.max_length,
            b_to_i_label=b_to_i_label,
        ),
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
        desc="Tokenizing and aligning labels",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        logging_dir=str(config.logging_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=lambda prediction: compute_metrics(prediction, label_list),
    )

    LOGGER.info("Starting training.")
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    LOGGER.info("Evaluating on the validation split.")
    validation_metrics = trainer.evaluate(metric_key_prefix="validation")
    trainer.log_metrics("validation", validation_metrics)
    trainer.save_metrics("validation", validation_metrics)

    LOGGER.info("Evaluating on the test split.")
    test_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"], metric_key_prefix="test")
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    LOGGER.info("Saving model, tokenizer, and trainer state to %s", config.output_dir)
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    trainer.save_state()

    return trainer


def main() -> None:
    """Entry point for command-line training."""

    setup_logging()
    config = parse_args()
    LOGGER.info("Training configuration: %s", config)
    train_model(config)
    LOGGER.info("Training complete. Model saved to %s", config.output_dir)


if __name__ == "__main__":
    main()
