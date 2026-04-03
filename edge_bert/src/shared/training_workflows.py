from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from codecarbon import EmissionsTracker
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, get_scheduler

from shared.experiment_settings import (
    DATASET_SETTINGS,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_LABELS,
    MODELS_DIR,
    RESULTS_DIR,
    save_json,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(slots=True)
class TrainingRunConfig:
    regimen_name: str
    model_label: str
    checkpoint_name: str
    history_file_name: str
    max_epochs: int
    early_stopping_patience: int | None = None
    freeze_embeddings: bool = False
    freeze_transformer_layers: int = 0
    batch_size: int = 16
    learning_rate: float = 2e-5
    seed: int = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_train_and_validation_loaders(batch_size: int) -> tuple[DataLoader, DataLoader, int]:
    dataset = load_dataset(DATASET_SETTINGS["name"], DATASET_SETTINGS["config"])
    tokenizer = DistilBertTokenizerFast.from_pretrained(DEFAULT_MODEL_NAME)

    def tokenize(example: dict[str, list[str]]) -> dict[str, list[int]]:
        return tokenizer(
            example["sentence"],
            truncation=True,
            padding="max_length",
            max_length=DATASET_SETTINGS["max_length"],
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size)
    return train_loader, val_loader, len(dataset["validation"])


def build_model(*, freeze_embeddings: bool, freeze_transformer_layers: int) -> DistilBertForSequenceClassification:
    model = DistilBertForSequenceClassification.from_pretrained(
        DEFAULT_MODEL_NAME,
        num_labels=DEFAULT_NUM_LABELS,
    )

    if freeze_embeddings:
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = False

    if freeze_transformer_layers > 0:
        for layer in model.distilbert.transformer.layer[:freeze_transformer_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    return model


def evaluate_model(
    model: DistilBertForSequenceClassification,
    val_loader: DataLoader,
) -> tuple[float, float]:
    model.eval()
    predictions: list[int] = []
    true_labels: list[int] = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())

    return accuracy_score(true_labels, predictions), f1_score(true_labels, predictions)


def build_training_payload(
    *,
    config: TrainingRunConfig,
    checkpoint_path: Path,
    best_accuracy: float,
    best_f1: float,
    best_epoch: int,
    best_global_step: int,
    completed_epochs: int,
    total_validation_passes: int,
    validation_size: int,
    stopped_early: bool,
    emissions: float | None,
    history: list[dict[str, float | int | bool]],
    steps_per_epoch: int,
) -> dict[str, object]:
    checkpoint_relative = checkpoint_path.relative_to(MODELS_DIR.parent)
    return {
        "regimen_name": config.regimen_name,
        "model_label": config.model_label,
        "max_epochs": config.max_epochs,
        "completed_epochs": completed_epochs,
        "steps_per_epoch": steps_per_epoch,
        "best_epoch": best_epoch,
        "best_global_step": best_global_step,
        "best_validation_accuracy": round(best_accuracy, 4),
        "best_validation_f1": round(best_f1, 4),
        "early_stopping_patience": config.early_stopping_patience,
        "used_early_stopping": config.early_stopping_patience is not None,
        "stopped_early": stopped_early,
        "validation_strategy": "epoch_end",
        "total_validation_passes": total_validation_passes,
        "validation_samples_per_pass": validation_size,
        "total_validation_examples_processed": total_validation_passes * validation_size,
        "checkpoint_path": str(checkpoint_relative).replace("\\", "/"),
        "emissions_kg_co2eq": emissions,
        "history": history,
    }


def run_training_experiment(config: TrainingRunConfig) -> dict[str, object]:
    seed_everything(config.seed)

    print(f"Training on: {DEVICE}")
    print(f"Regimen: {config.regimen_name}")
    print(f"Model   : {config.model_label}")

    train_loader, val_loader, validation_size = load_train_and_validation_loaders(config.batch_size)
    steps_per_epoch = len(train_loader)

    model = build_model(
        freeze_embeddings=config.freeze_embeddings,
        freeze_transformer_layers=config.freeze_transformer_layers,
    )
    if config.freeze_embeddings or config.freeze_transformer_layers > 0:
        print(
            "Freeze setup:"
            f" embeddings={config.freeze_embeddings},"
            f" frozen_transformer_layers={config.freeze_transformer_layers}"
        )

    model.to(DEVICE)
    optimizer_params = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = AdamW(optimizer_params, lr=config.learning_rate)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=config.max_epochs * steps_per_epoch,
    )

    checkpoint_path = MODELS_DIR / config.regimen_name / config.checkpoint_name
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = RESULTS_DIR / "training_histories" / config.regimen_name / config.history_file_name
    history_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker()
    tracker.start()

    best_accuracy = float("-inf")
    best_f1 = 0.0
    best_epoch = 0
    best_global_step = 0
    early_stop_counter = 0
    stopped_early = False
    global_step = 0
    total_validation_passes = 0
    history: list[dict[str, float | int | bool]] = []

    for epoch_index in range(config.max_epochs):
        epoch_number = epoch_index + 1
        print(f"\nEpoch {epoch_number}/{config.max_epochs}")

        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader):
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1

        train_loss = total_loss / steps_per_epoch
        acc, f1 = evaluate_model(model, val_loader)
        total_validation_passes += 1

        is_best = acc > best_accuracy
        if is_best:
            best_accuracy = acc
            best_f1 = f1
            best_epoch = epoch_number
            best_global_step = global_step
            torch.save(model.state_dict(), checkpoint_path)
            early_stop_counter = 0
            print(f"Best model saved to: {checkpoint_path}")
        elif config.early_stopping_patience is not None:
            early_stop_counter += 1

        history.append(
            {
                "epoch": epoch_number,
                "global_step": global_step,
                "train_loss": round(train_loss, 4),
                "validation_accuracy": round(acc, 4),
                "validation_f1": round(f1, 4),
                "is_best_checkpoint": is_best,
            }
        )

        print(f"Train Loss          : {train_loss:.4f}")
        print(f"Validation Accuracy : {acc:.4f}")
        print(f"Validation F1       : {f1:.4f}")
        print(f"Validation Passes   : {total_validation_passes}")

        if config.early_stopping_patience is not None and early_stop_counter >= config.early_stopping_patience:
            stopped_early = True
            print("Early stopping triggered.")
            break

    emissions = tracker.stop()
    payload = build_training_payload(
        config=config,
        checkpoint_path=checkpoint_path,
        best_accuracy=best_accuracy,
        best_f1=best_f1,
        best_epoch=best_epoch,
        best_global_step=best_global_step,
        completed_epochs=len(history),
        total_validation_passes=total_validation_passes,
        validation_size=validation_size,
        stopped_early=stopped_early,
        emissions=round(emissions, 6) if emissions is not None else None,
        history=history,
        steps_per_epoch=steps_per_epoch,
    )
    save_json(history_path, payload)

    print("\nRun Summary")
    print(f"Best Epoch                 : {best_epoch}")
    print(f"Best Validation Accuracy   : {best_accuracy:.4f}")
    print(f"Best Validation F1         : {best_f1:.4f}")
    print(f"Total Validation Passes    : {total_validation_passes}")
    print(f"Validation Samples / Pass  : {validation_size}")
    print(f"Total Validation Examples  : {total_validation_passes * validation_size}")
    print(f"Checkpoint                 : {checkpoint_path}")
    print(f"History JSON               : {history_path}")
    print(f"Training Emissions (kg)    : {payload['emissions_kg_co2eq']}")

    return payload
