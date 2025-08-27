import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from dataset import SuperconductingChipDataset
from model import ChipPerformanceNet


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def train(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = SuperconductingChipDataset(args.csv, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = get_device()
    model = ChipPerformanceNet(pretrained=True).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                preds = model(images)
                val_loss += criterion(preds, targets).item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch + 1}: val_loss={val_loss:.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Train chip performance predictor")
    parser.add_argument("--csv", type=str, required=True, help="CSV with tem_path, afm_path, performance")
    parser.add_argument("--output", type=str, default="model.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
