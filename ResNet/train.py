import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets

from models import CIFARResNet


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9


def augment(images):
    """4-pixel zero padding, random 32x32 crop and random horizontal flip, per image."""
    n = images.size(0)
    padded = F.pad(images, (4, 4, 4, 4))
    offsets = torch.arange(32, device=images.device)
    rows = torch.randint(0, 9, (n, 1), device=images.device) + offsets
    cols = torch.randint(0, 9, (n, 1), device=images.device) + offsets
    batch = torch.arange(n, device=images.device)[:, None, None, None]
    channels = torch.arange(images.size(1), device=images.device)[None, :, None, None]
    crops = padded[batch, channels, rows[:, None, :, None], cols[:, None, None, :]]
    flip = torch.rand(n, device=images.device) < 0.5
    return torch.where(flip[:, None, None, None], crops.flip(3), crops)


class GPULoader:
    def __init__(self, images, labels, mean, batch_size, train=False):
        self.images = images
        self.labels = labels
        self.mean = mean
        self.batch_size = batch_size
        self.train = train

    def __iter__(self):
        n = self.labels.size(0)
        order = torch.randperm(n, device=self.labels.device) if self.train else torch.arange(n, device=self.labels.device)
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            images = self.images[idx].float().div(255) - self.mean
            if self.train:
                images = augment(images)
            yield images, self.labels[idx]


def load_cifar(train, device):
    data = datasets.CIFAR10(DATA_DIR, train=train)
    images = torch.from_numpy(data.data).permute(0, 3, 1, 2).contiguous().to(device)
    labels = torch.tensor(data.targets, device=device)
    return images, labels


def get_loaders(batch_size, seed, val_split, device):
    images, labels = load_cifar(True, device)
    test_images, test_labels = load_cifar(False, device)

    if val_split:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(labels.size(0), generator=generator).to(device)
        train_idx, val_idx = indices[:45000], indices[45000:]
        images, labels, val_images, val_labels = images[train_idx], labels[train_idx], images[val_idx], labels[val_idx]

    mean = images.float().div(255).mean(0)
    train_loader = GPULoader(images, labels, mean, batch_size, train=True)
    test_loader = GPULoader(test_images, test_labels, mean, 500)

    if val_split:
        return train_loader, GPULoader(val_images, val_labels, mean, 500), test_loader, "val"
    return train_loader, test_loader, test_loader, "test"


def learning_rate(iteration, args):
    if iteration < args.warmup_iters:
        return args.lr * 0.1
    milestones = (args.iterations // 2, args.iterations * 3 // 4)
    return args.lr * 0.1 ** sum(iteration >= m for m in milestones)


def train_epoch(model, loader, criterion, optimizer, iteration, args):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        if iteration >= args.iterations:
            break
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(iteration, args)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iteration += 1

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 - 100.0 * correct / total, iteration


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    highway_blocks = model.highway_blocks()
    gate_sums = torch.zeros(len(highway_blocks), device=next(model.parameters()).device)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        if highway_blocks:
            gate_sums += torch.stack([b.last_gate_mean for b in highway_blocks]) * labels.size(0)

    gate_means = (gate_sums / total).tolist()
    return total_loss / total, 100.0 - 100.0 * correct / total, gate_means


def save_plots(history, eval_name, output_dir):
    iterations = [row["iteration"] / 1e4 for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in [(axes[0], "loss", "Cross-entropy loss"), (axes[1], "err", "Error (%)")]:
        ax.plot(iterations, [row[f"train_{metric}"] for row in history], "--", label="Train")
        ax.plot(iterations, [row[f"{eval_name}_{metric}"] for row in history], label=eval_name.capitalize())
        ax.set(xlabel="Iteration (1e4)", title=title)
        ax.legend()
    axes[1].set_ylim(0, 20)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet", "plain", "highway"], default="resnet")
    parser.add_argument("--depth", type=int, default=56)
    parser.add_argument("--iterations", type=int, default=64000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--val-split", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.warmup_iters is None:
        args.warmup_iters = 400 if args.depth >= 110 else 0

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = RUNS_DIR / f"{args.model}_{args.depth}{'_val' if args.val_split else ''}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    train_loader, eval_loader, test_loader, eval_name = get_loaders(
        args.batch_size, args.seed, args.val_split, device
    )
    model = CIFARResNet(args.model, args.depth).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    history = []
    iteration = 0
    epoch = 0
    while iteration < args.iterations:
        epoch += 1
        lr = learning_rate(iteration, args)
        train_loss, train_err, iteration = train_epoch(
            model, train_loader, criterion, optimizer, iteration, args
        )
        eval_loss, eval_err, eval_gates = evaluate(model, eval_loader, criterion)

        row = {
            "epoch": epoch,
            "iteration": iteration,
            "lr": lr,
            "train_loss": train_loss,
            "train_err": train_err,
            f"{eval_name}_loss": eval_loss,
            f"{eval_name}_err": eval_err,
        }
        row.update({f"gate_{i}": g for i, g in enumerate(eval_gates, start=1)})
        history.append(row)

        gate_info = f" | mean T {sum(eval_gates) / len(eval_gates):.3f}" if eval_gates else ""
        print(
            f"Epoch {epoch:3d} | iter {iteration:5d}/{args.iterations} | lr {lr:.4f} | "
            f"train loss {train_loss:.4f}, err {train_err:.2f}% | "
            f"{eval_name} loss {eval_loss:.4f}, err {eval_err:.2f}%{gate_info}"
        )

    torch.save({"model_state_dict": model.state_dict(), "iteration": iteration}, output_dir / "model.pt")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    save_plots(history, eval_name, output_dir)

    test_loss, test_err, test_gates = evaluate(model, test_loader, criterion)
    results = {
        **vars(args),
        "epochs": epoch,
        "final_train_err": history[-1]["train_err"],
        "test_err": test_err,
        "test_acc": 100.0 - test_err,
        "test_loss": test_loss,
    }
    if test_gates:
        results["test_gate_means"] = test_gates
    if args.val_split:
        results["final_val_err"] = history[-1]["val_err"]
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Test error: {test_err:.2f}% | Test accuracy: {100.0 - test_err:.2f}% | Test loss: {test_loss:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
