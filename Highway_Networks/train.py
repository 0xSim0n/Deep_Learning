import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import CIFAR10Net


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
WEIGHT_DECAY = 5e-4


def get_loaders(batch_size, seed):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_data = datasets.CIFAR10(DATA_DIR, train=True, transform=train_transform)
    val_data = datasets.CIFAR10(DATA_DIR, train=True, transform=eval_transform)
    test_data = datasets.CIFAR10(DATA_DIR, train=False, transform=eval_transform)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_data), generator=generator).tolist()
    train_indices, val_indices = indices[:45000], indices[45000:]

    train_loader = DataLoader(Subset(train_data, train_indices), batch_size, shuffle=True)
    val_loader = DataLoader(Subset(val_data, val_indices), batch_size)
    test_loader = DataLoader(test_data, batch_size)
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    highway_blocks = model.highway_blocks()
    gate_sums = torch.zeros(len(highway_blocks), device=device)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            if highway_blocks:
                gate_sums += torch.stack([b.last_gate_mean for b in highway_blocks]) * labels.size(0)

    gate_means = (gate_sums / total).tolist()
    return total_loss / total, 100.0 * correct / total, gate_means


def save_plots(history, test_gates, output_dir):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in [(axes[0], "loss", "Cross-entropy loss"), (axes[1], "acc", "Accuracy (%)")]:
        ax.plot(epochs, [row[f"train_{metric}"] for row in history], label="Train")
        ax.plot(epochs, [row[f"val_{metric}"] for row in history], label="Validation")
        ax.set(xlabel="Epoch", title=title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    if test_gates:
        blocks = list(range(1, len(test_gates) + 1))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(blocks, test_gates)
        ax.set(xlabel="Highway block", ylabel="Mean T(x)", title="Transform gate activity (test set)")
        ax.set_xticks(blocks)
        fig.tight_layout()
        fig.savefig(output_dir / "gate_means.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["highway", "plain"], default="highway")
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = RUNS_DIR / f"{args.model}_{args.blocks}_blocks_lr{args.lr}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_loaders(args.batch_size, args.seed)
    model = CIFAR10Net(args.model, args.blocks, args.channels).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    weights = [p for p in model.parameters() if p.ndim > 1]
    biases = [p for p in model.parameters() if p.ndim <= 1]
    optimizer = torch.optim.SGD(
        [{"params": weights, "weight_decay": WEIGHT_DECAY}, {"params": biases, "weight_decay": 0.0}],
        lr=args.lr,
        momentum=0.9,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    history = []
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc, val_gates = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        row.update({f"gate_{i}": g for i, g in enumerate(val_gates, start=1)})
        history.append(row)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, checkpoint_path)

        gate_info = f" | mean T {sum(val_gates) / len(val_gates):.3f}" if val_gates else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.2f}% | "
            f"val loss {val_loss:.4f}, acc {val_acc:.2f}%{gate_info}"
        )

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_gates = run_epoch(model, test_loader, criterion, device)
    save_plots(history, test_gates, output_dir)

    results = {
        **vars(args),
        "best_val_acc": best_val_acc,
        "best_epoch": checkpoint["epoch"],
        "test_acc": test_acc,
        "test_loss": test_loss,
        "test_gate_means": test_gates,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {checkpoint['epoch']})")
    print(f"Test accuracy: {test_acc:.2f}% | Test loss: {test_loss:.4f}")
    if test_gates:
        print("Mean T(x) per block: " + ", ".join(f"{g:.3f}" for g in test_gates))
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
