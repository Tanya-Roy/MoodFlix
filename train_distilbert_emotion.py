"""Fine-tune DistilBERT on the public dair-ai/emotion dataset.

Training uses the dataset's train and validation splits. The held-out test
split is evaluated only after selecting the best validation macro-F1 model.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "models" / "distilbert-emotion"
BASE_MODEL = "distilbert-base-uncased"
DATASET_NAME = "dair-ai/emotion"
DATASET_CONFIG = "split"
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for six-way emotion classification."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed-precision training, even when a CUDA GPU is available.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_dataset(dataset) -> None:
    required_splits = {"train", "validation", "test"}
    missing = required_splits.difference(dataset.keys())
    if missing:
        raise ValueError(f"{DATASET_NAME} is missing splits: {sorted(missing)}")
    for split in required_splits:
        if not {"text", "label"}.issubset(dataset[split].column_names):
            raise ValueError(
                f"{DATASET_NAME}/{split} must contain text and label columns."
            )
    label_feature = dataset["train"].features["label"]
    names = getattr(label_feature, "names", None)
    if names is None or set(names) != set(LABEL_NAMES):
        raise ValueError(
            "Expected dair-ai/emotion to provide exactly these named labels: "
            + ", ".join(LABEL_NAMES)
        )


def prepare_loaders(dataset, tokenizer, batch_size: int, max_length: int):
    label_names = dataset["train"].features["label"].names
    label_to_id = {name: index for index, name in enumerate(LABEL_NAMES)}

    def tokenize_batch(batch):
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )
        encoded["labels"] = [
            label_to_id[label_names[label_id]] for label_id in batch["label"]
        ]
        return encoded

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing emotion text",
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loaders = {
        split: DataLoader(
            tokenized[split],
            batch_size=batch_size,
            shuffle=split == "train",
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )
        for split in ("train", "validation", "test")
    }
    return loaders, label_to_id


def calculate_metrics(labels: list[int], predictions: list[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
    }


def evaluate(model, loader, device, amp_enabled: bool):
    model.eval()
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(**batch, labels=labels)
            total_loss += outputs.loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
    metrics = calculate_metrics(all_labels, all_predictions)
    metrics["loss"] = total_loss / len(all_labels)
    return metrics, all_labels, all_predictions


def main() -> None:
    args = parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.max_length < 1
        or args.patience < 1
        or args.learning_rate <= 0
    ):
        raise ValueError(
            "Epochs, batch size, max length, patience, and learning rate "
            "must be positive."
        )
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    print(f"Device: {device}")
    if device.type == "cpu":
        print("CPU training may take a while; CUDA is used automatically if available.")
    print(f"Loading {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    check_dataset(dataset)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    loaders, label_to_id = prepare_loaders(
        dataset,
        tokenizer,
        args.batch_size,
        args.max_length,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL_NAMES),
        id2label={index: name for name, index in label_to_id.items()},
        label2id=label_to_id,
    ).to(device)

    train_labels = np.asarray(loaders["train"].dataset["labels"], dtype=np.int64)
    class_counts = np.bincount(train_labels, minlength=len(LABEL_NAMES))
    class_weights = 1 / np.sqrt(class_counts.clip(min=1))
    class_weights = class_weights / class_weights.mean()
    loss_function = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    steps_per_epoch = len(loaders["train"])
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(loaders["train"], start=1):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(**batch).logits
                loss = loss_function(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()
            if step % 100 == 0 or step == steps_per_epoch:
                print(
                    f"Epoch {epoch}/{args.epochs} "
                    f"step {step}/{steps_per_epoch} "
                    f"train_loss={running_loss / step:.4f}"
                )

        validation_metrics, _, _ = evaluate(
            model,
            loaders["validation"],
            device,
            amp_enabled,
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": running_loss / steps_per_epoch,
            "validation": validation_metrics,
        }
        history.append(epoch_result)
        print(
            f"Epoch {epoch}: validation accuracy={validation_metrics['accuracy']:.4f}, "
            f"macro_f1={validation_metrics['macro_f1']:.4f}, "
            f"weighted_f1={validation_metrics['weighted_f1']:.4f}"
        )

        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    best_model = AutoModelForSequenceClassification.from_pretrained(
        args.output_dir
    ).to(device)
    test_metrics, test_labels, test_predictions = evaluate(
        best_model,
        loaders["test"],
        device,
        amp_enabled,
    )
    majority_label = int(np.bincount(train_labels).argmax())
    majority_predictions = [majority_label] * len(test_labels)
    report = {
        "dataset": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "split_sizes": {split: len(dataset[split]) for split in dataset},
        "base_model": BASE_MODEL,
        "label_names": LABEL_NAMES,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "class_weights": {
            name: float(class_weights[label_to_id[name]])
            for name in LABEL_NAMES
        },
        "epoch_history": history,
        "test": {
            **test_metrics,
            "majority_class_baseline": calculate_metrics(
                test_labels,
                majority_predictions,
            ),
            "classification_report": classification_report(
                test_labels,
                test_predictions,
                labels=list(range(len(LABEL_NAMES))),
                target_names=LABEL_NAMES,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                test_labels,
                test_predictions,
                labels=list(range(len(LABEL_NAMES))),
            ).tolist(),
        },
    }
    report_path = args.output_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "Held-out test metrics (best validation macro-F1 checkpoint): "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"macro_f1={test_metrics['macro_f1']:.4f}, "
        f"weighted_f1={test_metrics['weighted_f1']:.4f}"
    )
    print(f"Saved fine-tuned model and metrics to: {args.output_dir}")
    print(f"Metrics report: {report_path}")


if __name__ == "__main__":
    main()
