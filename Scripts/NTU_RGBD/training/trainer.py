# Scripts/NTU_RGBD/training/trainer.py

from __future__ import annotations

from pathlib import Path
from typing import Optional
import csv
import time

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


# ============================================================
# 1. Train one epoch
# ============================================================

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    print_every: int = 50,
) -> float:
    model.train()

    running_loss = 0.0
    total_samples = 0

    for batch_index, batch in enumerate(
        dataloader,
        start=1,
    ):
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        target_heatmaps = batch["heatmaps"].to(
            device,
            non_blocking=True,
        )

        visibility = batch["visibility"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(images)

        loss = criterion(
            predictions,
            target_heatmaps,
            visibility,
        )

        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]

        running_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        if (
            print_every > 0
            and batch_index % print_every == 0
        ):
            average_loss = (
                running_loss
                / max(total_samples, 1)
            )

            print(
                f"Epoch {epoch} | "
                f"Batch {batch_index}/{len(dataloader)} | "
                f"Train Loss: {average_loss:.6f}"
            )

    return (
        running_loss
        / max(total_samples, 1)
    )


# ============================================================
# 2. Validate one epoch
# ============================================================

@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
) -> float:
    model.eval()

    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        target_heatmaps = batch["heatmaps"].to(
            device,
            non_blocking=True,
        )

        visibility = batch["visibility"].to(
            device,
            non_blocking=True,
        )

        predictions = model(images)

        loss = criterion(
            predictions,
            target_heatmaps,
            visibility,
        )

        batch_size = images.shape[0]

        running_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

    return (
        running_loss
        / max(total_samples, 1)
    )


# ============================================================
# 3. Save checkpoint
# ============================================================

def save_checkpoint(
    output_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    history: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> None:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    if history is not None:
        checkpoint["history"] = history

    if extra is not None:
        checkpoint.update(extra)

    torch.save(
        checkpoint,
        output_path,
    )


# ============================================================
# 4. Save history CSV
# ============================================================

def save_training_history_csv(
    history: dict[str, list],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "learning_rate",
        "epoch_seconds",
    ]

    number_of_epochs = len(
        history["epoch"]
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index in range(number_of_epochs):
            writer.writerow({
                "epoch": history["epoch"][index],
                "train_loss": (
                    history["train_loss"][index]
                ),
                "val_loss": (
                    history["val_loss"][index]
                ),
                "learning_rate": (
                    history["learning_rate"][index]
                ),
                "epoch_seconds": (
                    history["epoch_seconds"][index]
                ),
            })


# ============================================================
# 5. Plot loss curves
# ============================================================

def plot_loss_curves(
    history: dict[str, list],
    output_dir: str | Path,
    title: str = "NTU RGB+D ResNet50 Heatmap Loss",
) -> None:
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = history["epoch"]
    train_losses = history["train_loss"]
    val_losses = history["val_loss"]

    if not epochs:
        raise ValueError(
            "Training history is empty"
        )

    best_epoch_index = int(
        min(
            range(len(val_losses)),
            key=val_losses.__getitem__,
        )
    )

    best_epoch = epochs[
        best_epoch_index
    ]

    best_val_loss = val_losses[
        best_epoch_index
    ]

    figure = plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        epochs,
        train_losses,
        marker="o",
        markersize=3,
        linewidth=1.8,
        label="Train Loss",
    )

    plt.plot(
        epochs,
        val_losses,
        marker="o",
        markersize=3,
        linewidth=1.8,
        label="Validation Loss",
    )

    plt.scatter(
        [best_epoch],
        [best_val_loss],
        s=60,
        zorder=5,
        label=(
            f"Best Validation "
            f"(Epoch {best_epoch})"
        ),
    )

    plt.axvline(
        x=best_epoch,
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Heatmap MSE Loss")
    plt.grid(
        True,
        alpha=0.3,
    )
    plt.legend()
    plt.tight_layout()

    png_path = (
        output_dir
        / "loss_curve.png"
    )

    pdf_path = (
        output_dir
        / "loss_curve.pdf"
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Loss curve PNG: {png_path}"
    )

    print(
        f"Loss curve PDF: {pdf_path}"
    )


# ============================================================
# 6. Run complete training
# ============================================================

def run_training(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    epochs: int,
    output_dir: str | Path,
    scheduler=None,
    print_every: int = 50,
    model_name: str = (
        "NTU RGB+D ResNet50 Heatmap"
    ),
) -> dict[str, list]:
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = model.to(device)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
        "epoch_seconds": [],
    }

    best_val_loss = float("inf")

    history_csv_path = (
        output_dir
        / "training_history.csv"
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        epoch_start_time = time.time()

        print()
        print("=" * 70)
        print(f"Epoch {epoch}/{epochs}")
        print("=" * 70)

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            print_every=print_every,
        )

        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        epoch_seconds = (
            time.time()
            - epoch_start_time
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(
            float(train_loss)
        )
        history["val_loss"].append(
            float(val_loss)
        )
        history["learning_rate"].append(
            learning_rate
        )
        history["epoch_seconds"].append(
            float(epoch_seconds)
        )

        print(
            f"Train loss:    {train_loss:.6f}"
        )

        print(
            f"Val loss:      {val_loss:.6f}"
        )

        print(
            f"Learning rate: {learning_rate:.8f}"
        )

        print(
            f"Epoch time:    "
            f"{epoch_seconds:.2f}s"
        )

        if scheduler is not None:
            if isinstance(
                scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        save_checkpoint(
            output_path=(
                output_dir
                / "last_model.pt"
            ),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            history=history,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                output_path=(
                    output_dir
                    / "best_model.pt"
                ),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                history=history,
            )

            print(
                "Saved new best model."
            )

        # 每个 epoch 都更新 CSV 和曲线。
        # 即使训练中途停止，已有记录也不会丢失。
        save_training_history_csv(
            history=history,
            output_path=history_csv_path,
        )

        plot_loss_curves(
            history=history,
            output_dir=output_dir,
            title=f"{model_name} Loss",
        )

    print()
    print("=" * 70)
    print("Training finished")
    print("=" * 70)

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"History CSV: "
        f"{history_csv_path}"
    )

    return history